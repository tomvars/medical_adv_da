import nibabel as nib
import numpy as np
import torch
from functools import partial
from collections import defaultdict
from pairwise_measures import PairwiseMeasures
from utils import apply_transform, non_geometric_augmentations, generate_affine, to_var_gpu, batch_adaptation, soft_dice

def evaluate(args, preds, targets, prefix,
             metrics=['dice', 'jaccard', 'sensitivity', 'specificity', 'soft_dice',
                      'loads', 'haus_dist', 'vol_diff', 'ppv', 'connected_elements']):
    output_dict = defaultdict(list)
    nifty_metrics = ['dice', 'jaccard', 'sensitivity', 'specificity',
                     'haus_dist', 'vol_diff', 'ppv', 'connected_elements']
    for pred, target in zip(preds, targets):
        seg = np.where(pred > 0.5, np.ones_like(pred, dtype=np.int64), np.zeros_like(pred, dtype=np.int64))
        ref = np.where(target > 0.5, np.ones_like(target, dtype=np.int64), np.zeros_like(target, dtype=np.int64))
        pairwise = PairwiseMeasures(seg, ref)
        for metric in nifty_metrics:
            if metric in metrics:
                if metric == 'connected_elements':
                    TPc, FPc, FNc = pairwise.m_dict[metric][0]()
                    output_dict[prefix + 'TPc'].append(TPc)
                    output_dict[prefix + 'FPc'].append(FPc)
                    output_dict[prefix + 'FNc'].append(FNc)
                else:
                    output_dict[prefix + metric].append(pairwise.m_dict[metric][0]())
        if 'soft_dice' in metrics:
            output_dict[prefix + 'soft_dice'].append(soft_dice(pred, ref, args.labels))
        if 'loads' in metrics:
            output_dict[prefix + 'loads'].append(np.sum(pred))
        if 'per_pixel_diff' in metrics:
            output_dict[prefix + 'per_pixel_diff'].append(np.mean(np.abs(ref - pred)))
    return output_dict

def inference_tumour(args, p, model, whole_volume_dataset, iteration=0, prefix='', infer_on=None):
    """
    This function should run inference on a set of volumes, save the results, calculate the dice
    """
    def save_img(format_spec, identifier, array):
        img = nib.Nifti1Image(array, np.eye(4))
        fn = format_spec.format(identifier)
        nib.save(img, fn)
        return fn

    with torch.set_grad_enabled(False):
        model.eval()
        preds_0, preds_ema = [], []
        preds, targets = [], []
        predsAug, predsT = [], []
        range_of_volumes = range(len(whole_volume_dataset)) if infer_on is None else infer_on
        print('Evaluating on {} subjects'.format(len(range_of_volumes)))
        for index in range(len(range_of_volumes)):
            print('Evaluating on subject {}'.format(str(index)))
            inputs, labels = whole_volume_dataset[index]
            #TODO: inputs is of size (4, 170, 240, 160), need to change inference values accordingly.
            subj_id = whole_volume_dataset.get_subject_id_from_index(index)
            targetL = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[-1]))
            outputS = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[-1]))
            inputsS = np.zeros(shape=(inputs.shape[0], args.paddtarget, args.paddtarget, inputs.shape[-1]))
            outputsT = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[-1]))
            outputsAug = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[-1]))
            for slice_index in np.arange(0, inputs.shape[-1], step=args.batch_size):
                index_start = slice_index
                index_end = min(slice_index+args.batch_size, inputs.shape[-1])
                batch_input = np.einsum('ijkl->lijk', inputs[:, :, :, index_start:index_end])
                batch_labels = np.einsum('ijk->kij', labels[:, :, index_start:index_end])
                batch_input = torch.tensor(batch_input)
                batch_labels = torch.tensor(np.expand_dims(batch_labels, axis=1))
                batch_input, batch_labels = batch_adaptation(batch_input, batch_labels, args.paddtarget)
                batch_input, batch_labels = to_var_gpu(batch_input), to_var_gpu(batch_labels)
                outputs, _, _, _, _, _, _, _, _, _ = model(batch_input)
                outputs = torch.sigmoid(outputs)
                if args.method == 'A2':
                    Theta, Theta_inv = generate_affine(batch_input, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(batch_input, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                elif args.method == 'A4':
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                elif args.method in ['A3', 'adversarial', 'mean_teacher']:
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(inputstaug, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                outputS[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                 np.squeeze(outputs.detach().cpu().numpy()))
                targetL[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                 np.squeeze(batch_labels.detach().cpu().numpy()))
                inputsS[:, :, :, index_start:index_end] = np.einsum('ijkl->jkli', np.squeeze(batch_input.detach().cpu().numpy()))
                if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                    outputsAug[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                      np.squeeze(outputstaug.detach().cpu().numpy()))
                    if args.method in ['A3', 'A2', 'adversarial', 'mean_teacher']:
                        outputsT[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                          np.squeeze(outputs_t.detach().cpu().numpy()))
            format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source, args.target, args.tag,
                                                      iteration) + \
                          '_{}_' + f'{str(subj_id)}.nii.gz'
            ema_format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source,
                                                          args.target, args.tag, 'EMA') + \
                              '_{}_' + f'{str(subj_id)}.nii.gz'
            if iteration == 0:
                fn = save_img(format_spec=ema_format_spec, identifier='Prediction', array=outputS)
            else:
                pred_zero = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                            f'_{args.tag}_0__Prediction_{str(subj_id)}.nii.gz'
                outputs_0 = nib.load(pred_zero).get_data()
                preds_0.append(outputs_0)
                alpha = 0.9
                pred_ema_filename = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                                    f'_{args.tag}_EMA__Prediction_{str(subj_id)}.nii.gz'
                pred_ema_t_minus_one = nib.load(pred_ema_filename).get_data()
                pred_ema = alpha * outputS + (1 - alpha) * pred_ema_t_minus_one
                preds_ema.append(pred_ema)
                save_img(format_spec=ema_format_spec, identifier='Prediction', array=pred_ema)
            save_img(format_spec=format_spec, identifier='Prediction', array=outputS)
            save_img(format_spec=format_spec, identifier='target', array=targetL)
            for idx, modality in enumerate(['flair', 't1c', 't1', 't2']):
                save_img(format_spec=format_spec, identifier='{}_mri'.format(modality), array=inputsS[idx, ...])
            preds.append(outputS)
            targets.append(targetL)
            if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                predsAug.append(outputsAug)
                save_img(format_spec=format_spec, identifier='Aug', array=outputsAug)
                if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                    predsT.append(outputsT)
                    save_img(format_spec=format_spec, identifier='Transformed', array=outputsT)
        performance_supervised = evaluate(args=args, preds=preds, targets=targets, prefix='supervised_')
        performance_i = None
        if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
            if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                performance_i = evaluate(args=args, preds=predsAug, targets=predsT, prefix='consistency_')
            else:
                performance_i = evaluate(args=args, preds=predsAug, targets=preds, prefix='consistency_')
        if iteration == 0:
            return performance_supervised, performance_i, None, None
        else:
            performance_compared_to_0 = evaluate(args=args, preds=preds, targets=preds_0, prefix='diff_to_0_',
                                                 metrics=['per_pixel_diff'])
            performance_compared_to_ema = evaluate(args=args, preds=preds, targets=preds_ema, prefix='diff_to_ema_',
                                                 metrics=['per_pixel_diff'])
            return performance_supervised, performance_i, performance_compared_to_0, performance_compared_to_ema


def inference_ms(args, p, model, whole_volume_dataset, iteration=0, prefix='', infer_on=None, eval_diff=True):
    """
    This function should run inference on a set of volumes, save the results, calculate the dice
    """
    def save_img(format_spec, identifier, array):
        img = nib.Nifti1Image(array, np.eye(4))
        fn = format_spec.format(identifier)
        nib.save(img, fn)
        return fn

    with torch.set_grad_enabled(False):
        model.eval()
        preds_0, preds_ema = [], []
        preds, targets = [], []
        predsAug, predsT = [], []
        print('Evaluating on {} subjects'.format(len(whole_volume_dataset)))
        range_of_volumes = range(len(whole_volume_dataset)) if infer_on is None else infer_on
        for index in range_of_volumes:
            print('Evaluating on subject {}'.format(str(index)))
            inputs, labels = whole_volume_dataset[index]
            subj_id = whole_volume_dataset.get_subject_id_from_index(index)
            targetL = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputS = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            inputsS = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputsT = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputsAug = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            for slice_index in np.arange(0, inputs.shape[2], step=args.batch_size):
                index_start = slice_index
                index_end = min(slice_index+args.batch_size, inputs.shape[2])
                batch_input = np.einsum('ijk->kij', inputs[:, :, index_start:index_end])
                batch_labels = np.einsum('ijk->kij', labels[:, :, index_start:index_end])
                batch_input = torch.tensor(np.expand_dims(batch_input, axis=1).astype(np.float32))
                batch_labels = torch.tensor(np.expand_dims(batch_labels, axis=1))
                batch_input, batch_labels = batch_adaptation(batch_input, batch_labels, args.paddtarget)
                batch_input, batch_labels = to_var_gpu(batch_input), to_var_gpu(batch_labels)
                outputs, _, _, _, _, _, _, _, _, _ = model(batch_input)
                outputs = torch.sigmoid(outputs)
                if args.method == 'A2':
                    Theta, Theta_inv = generate_affine(batch_input, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(batch_input, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                elif args.method == 'A4':
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                elif args.method in ['A3', 'adversarial', 'mean_teacher']:
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(inputstaug, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                outputS[:, :, index_start:index_end] = np.einsum('ijk->jki', outputs.detach().cpu().numpy()[:, 0, ...])
                targetL[:, :, index_start:index_end] = np.einsum('ijk->jki', batch_labels.detach().cpu().numpy()[:, 0, ...])
                inputsS[:, :, index_start:index_end] = np.einsum('ijk->jki', batch_input.detach().cpu().numpy()[:, 0, ...])
                if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                    outputsAug[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                        outputstaug.detach().cpu().numpy()[:, 0, ...])
                    if args.method in ['A3', 'A2', 'adversarial', 'mean_teacher']:
                        outputsT[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                          outputs_t.detach().cpu().numpy()[:, 0, ...])
            format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source, args.target, args.tag, iteration) +\
                      '_{}_' + f'{str(subj_id)}.nii.gz'
            ema_format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source,
                                                          args.target, args.tag, 'EMA') + \
                              '_{}_' + f'{str(subj_id)}.nii.gz'
            if iteration == 0:
                save_img(format_spec=ema_format_spec, identifier='Prediction', array=outputS)
            elif eval_diff and iteration > 0:
                pred_zero = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                            f'_{args.tag}_{0}__Prediction_{str(subj_id)}.nii.gz'
                outputs_0 = nib.load(pred_zero).get_data()
                preds_0.append(outputs_0)
                alpha = 0.9
                pred_ema_filename = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                                    f'_{args.tag}_EMA__Prediction_{str(subj_id)}.nii.gz'
                print(pred_ema_filename)
                pred_ema_t_minus_one = nib.load(pred_ema_filename).get_data()
                pred_ema = alpha * outputS + (1 - alpha) * pred_ema_t_minus_one
                preds_ema.append(pred_ema)
                save_img(format_spec=ema_format_spec, identifier='Prediction', array=pred_ema)
            else:
                print('Not computing diff')
            save_img(format_spec=format_spec, identifier='Prediction', array=outputS)
            save_img(format_spec=format_spec, identifier='target', array=targetL)
            save_img(format_spec=format_spec, identifier='mri', array=inputsS)
            preds.append(outputS)
            targets.append(targetL)
            if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                predsAug.append(outputsAug)
                save_img(format_spec=format_spec, identifier='Aug', array=outputsAug)
                if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                    predsT.append(outputsT)
                    save_img(format_spec=format_spec, identifier='Transformed', array=outputsT)
        performance_supervised = evaluate(args=args, preds=preds, targets=targets, prefix='supervised_')
        performance_i = None
        if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
            if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                performance_i = evaluate(args=args, preds=predsAug, targets=predsT, prefix='consistency_')
            else:
                performance_i = evaluate(args=args, preds=predsAug, targets=preds, prefix='consistency_')
        if iteration == 0:
            return performance_supervised, performance_i, None, None
        else:
            performance_compared_to_0 = evaluate(args=args, preds=preds, targets=preds_0, prefix='diff_to_0_',
                                                 metrics=['per_pixel_diff'])
            performance_compared_to_ema = evaluate(args=args, preds=preds, targets=preds_ema, prefix='diff_to_ema_',
                                                 metrics=['per_pixel_diff'])
            return performance_supervised, performance_i, performance_compared_to_0, performance_compared_to_ema
        
def inference_crossmoda(args, p, model, whole_volume_dataset, iteration=0, prefix='', infer_on=None, eval_diff=True):
    """
    This function should run inference on a set of volumes, save the results, calculate the dice
    """
    def save_img(format_spec, identifier, array):
        img = nib.Nifti1Image(array, np.eye(4))
        fn = format_spec.format(identifier)
        nib.save(img, fn)
        return fn

    with torch.set_grad_enabled(False):
        model.eval()
        preds_0, preds_ema = [], []
        preds, targets = [], []
        predsAug, predsT = [], []
        print('Evaluating on {} subjects'.format(len(whole_volume_dataset)))
        range_of_volumes = range(len(whole_volume_dataset)) if infer_on is None else infer_on
        for index in range_of_volumes:
            print('Evaluating on subject {}'.format(str(index)))
            inputs, labels = whole_volume_dataset[index]
            subj_id = whole_volume_dataset.get_subject_id_from_index(index)
            targetL = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputS = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            inputsS = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputsT = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            outputsAug = np.zeros(shape=(args.paddtarget, args.paddtarget, inputs.shape[2]))
            for slice_index in np.arange(0, inputs.shape[2], step=args.batch_size):
                index_start = slice_index
                index_end = min(slice_index+args.batch_size, inputs.shape[2])
                batch_input = np.einsum('ijk->kij', inputs[:, :, index_start:index_end])
                batch_labels = np.einsum('ijk->kij', labels[:, :, index_start:index_end])
                batch_input = torch.tensor(np.expand_dims(batch_input, axis=1).astype(np.float32))
                batch_labels = torch.tensor(np.expand_dims(batch_labels, axis=1))
                batch_input, batch_labels = batch_adaptation(batch_input, batch_labels, args.paddtarget)
                batch_input, batch_labels = to_var_gpu(batch_input), to_var_gpu(batch_labels)
                outputs, _, _, _, _, _, _, _, _, _, _ = model(batch_input)
                outputs = torch.sigmoid(outputs)
                if args.method == 'A2':
                    Theta, Theta_inv = generate_affine(batch_input, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(batch_input, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                elif args.method == 'A4':
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    outputstaug, _, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                elif args.method in ['A3', 'adversarial', 'mean_teacher']:
                    batch_trs = batch_input.cpu().numpy()
                    batch_trs = p.map(partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                      np.copy(batch_trs))
                    batch_trs = p.map(partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                      np.copy(batch_trs))
                    inputstaug = torch.Tensor(batch_trs).cuda()
                    Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                                      scale=args.affine_scale,
                                                      shearingScale=args.affine_shearing)
                    inputstaug = apply_transform(inputstaug, Theta)
                    outputstaug, _, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                    outputstaug = torch.sigmoid(outputstaug)
                    outputs_t = apply_transform(outputs, Theta)
                outputS[:, :, index_start:index_end] = np.einsum('ijk->jki', outputs.detach().cpu().numpy()[:, 0, ...])
                targetL[:, :, index_start:index_end] = np.einsum('ijk->jki', batch_labels.detach().cpu().numpy()[:, 0, ...])
                inputsS[:, :, index_start:index_end] = np.einsum('ijk->jki', batch_input.detach().cpu().numpy()[:, 0, ...])
                if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                    outputsAug[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                        outputstaug.detach().cpu().numpy()[:, 0, ...])
                    if args.method in ['A3', 'A2', 'adversarial', 'mean_teacher']:
                        outputsT[:, :, index_start:index_end] = np.einsum('ijk->jki',
                                                                          outputs_t.detach().cpu().numpy()[:, 0, ...])
            format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source, args.target, args.tag, iteration) +\
                      '_{}_' + f'{str(subj_id)}.nii.gz'
            ema_format_spec = '{}_{}_{}_{}_{}_{}_'.format(prefix, args.method, args.source,
                                                          args.target, args.tag, 'EMA') + \
                              '_{}_' + f'{str(subj_id)}.nii.gz'
            if iteration == 0:
                save_img(format_spec=ema_format_spec, identifier='Prediction', array=outputS)
            elif eval_diff and iteration > 0:
                pred_zero = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                            f'_{args.tag}_{0}__Prediction_{str(subj_id)}.nii.gz'
                outputs_0 = nib.load(pred_zero).get_data()
                preds_0.append(outputs_0)
                alpha = 0.9
                pred_ema_filename = f'{prefix}_{args.method}_{args.source}_{args.target}' \
                                    f'_{args.tag}_EMA__Prediction_{str(subj_id)}.nii.gz'
                print(pred_ema_filename)
                pred_ema_t_minus_one = nib.load(pred_ema_filename).get_data()
                pred_ema = alpha * outputS + (1 - alpha) * pred_ema_t_minus_one
                preds_ema.append(pred_ema)
                save_img(format_spec=ema_format_spec, identifier='Prediction', array=pred_ema)
            else:
                print('Not computing diff')
            save_img(format_spec=format_spec, identifier='Prediction', array=outputS)
            save_img(format_spec=format_spec, identifier='target', array=targetL)
            save_img(format_spec=format_spec, identifier='mri', array=inputsS)
            preds.append(outputS)
            targets.append(targetL)
            if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
                predsAug.append(outputsAug)
                save_img(format_spec=format_spec, identifier='Aug', array=outputsAug)
                if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                    predsT.append(outputsT)
                    save_img(format_spec=format_spec, identifier='Transformed', array=outputsT)
        performance_supervised = evaluate(args=args, preds=preds, targets=targets, prefix='supervised_')
        performance_i = None
        if args.method in ['A2', 'A3', 'A4', 'adversarial', 'mean_teacher']:
            if args.method in ['A2', 'A3', 'adversarial', 'mean_teacher']:
                performance_i = evaluate(args=args, preds=predsAug, targets=predsT, prefix='consistency_')
            else:
                performance_i = evaluate(args=args, preds=predsAug, targets=preds, prefix='consistency_')
        if iteration == 0:
            return performance_supervised, performance_i, None, None
        else:
            performance_compared_to_0 = evaluate(args=args, preds=preds, targets=preds_0, prefix='diff_to_0_',
                                                 metrics=['per_pixel_diff'])
            performance_compared_to_ema = evaluate(args=args, preds=preds, targets=preds_ema, prefix='diff_to_ema_',
                                                 metrics=['per_pixel_diff'])
            return performance_supervised, performance_i, performance_compared_to_0, performance_compared_to_ema