import nibabel as nib
import numpy as np
import sys
import torch
from functools import partial
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from pairwise_measures import PairwiseMeasures
from src.utils import apply_transform,\
non_geometric_augmentations, \
generate_affine, to_var_gpu, \
batch_adaptation, soft_dice, \
collate_patches_object_detection
import src.medicaldetectiontoolkit.model_utils as mutils
from monai.data import GridPatchDataset, PatchIter, DataLoader, decollate_batch
from monai.data.nifti_saver import NiftiSaver
from monai.transforms import (LoadImaged,
                              Lambdad,
                              Orientationd,
                              SqueezeDimd,
                              ToTensord,
                              Compose,
                              AddChanneld,
                              Invertd,
                              ResizeWithPadOrCropd,
                              ScaleIntensityd,
                              SpatialCropd,
                              BatchInverseTransform,
                              RandWeightedCropd,
                              MapLabelValued,
                              CopyItemsd,
                              RandAffined,
                              Spacingd
                             )
from monai.inferers.utils import sliding_window_inference

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
        
def patch_based_inference(args, model, inference_dir,
                          source_dataset, target_dataset):
    # There needs to be a mapping from args to a transforms list
    datasets = [ds for ds in [source_dataset, target_dataset] if ds is not None]
    if args.task == 'object_detection':
        # Trying out training inference.
        val_org_loader = DataLoader(datasets[0], batch_size=1, num_workers=1)
        nifti_saver_pred = NiftiSaver(output_dir=inference_dir)
        nifti_saver_img = NiftiSaver(output_dir=inference_dir, output_postfix='img')
        nifti_saver_lab = NiftiSaver(output_dir=inference_dir, output_postfix='lab')
        with torch.no_grad():
            for val_data in val_org_loader:
                val_data["pred"] = partial(model.inference_func, mode='box_out')(val_data['inputs'].to(model.device))
                val_data = [i for i in decollate_batch(val_data)]
                print(val_data[0]['pred'].sum())
                nifti_saver_pred.save(val_data[0]['pred'],
                                 meta_data={
                                     'filename_or_obj': Path(val_data[0]
                                                             ['inputs_meta_dict'][
                                                                 'filename_or_obj']).name})
                print(val_data[0]['labels'].sum())
                nifti_saver_lab.save(val_data[0]['labels'],
                                 meta_data={
                                     'filename_or_obj': Path(val_data[0]
                                                             ['inputs_meta_dict'][
                                                                 'filename_or_obj']).name})
                nifti_saver_img.save(val_data[0]['inputs'],
                                 meta_data={
                                     'filename_or_obj': Path(val_data[0]
                                                             ['inputs_meta_dict'][
                                                                 'filename_or_obj']).name})
        print('finished processing')
        exit()
#     if args.task == 'object_detection':
#         for dataset in datasets:
#             with torch.no_grad():
#                 for idx in range(len(dataset)):
#                     # Get preds on source
#                     patch_iter = PatchIter(patch_size=args.spatial_size, start_pos=(0, 0, 0), mode='edge')
#                     grid_patch_dataset = GridPatchDataset(dataset=np.expand_dims(dataset[idx]['inputs'], axis=0),
#                                                           patch_iter=patch_iter)
#                     inference_dl = DataLoader(grid_patch_dataset, batch_size=args.batch_size, shuffle=False)
#                     inverse_op = BatchInverseTransform(transform=dataset.transform, loader=inference_dl)
#                     # inference loop
#                     patch_detections = []
#                     output_img = np.zeros_like(dataset[idx]['inputs'][0])
#                     output_detections = np.zeros_like(dataset[idx]['inputs'][0])
#                     output_seg = np.zeros_like(dataset[idx]['inputs'][0])
#                     with tqdm(total=len(source_dataset), file=sys.stdout) as pbar:
#                         for grid_inputs, grid_coords in inference_dl:
#                             pbar.update(1)
#                             output_dict = model.inference_func(grid_inputs.to(model.device))
#                             boxes = output_dict['results_dict']['boxes']
#                             print(inverse_op(grid_inputs).shape)
#                             for box, grid_coord, inputs in zip(boxes, grid_coords, output_dict['source_inputs']):
#                                 grid_coord = grid_coord.cpu().numpy()[1:]
#                                 print(output_img.shape)
#                                 print(grid_coord)
#                                 print(inputs.cpu().numpy().shape)
#                                 output_img[
#                                     grid_coord[0][0]:grid_coord[0][1],
#                                     grid_coord[1][0]:grid_coord[1][1],
#                                     grid_coord[2][0]:grid_coord[2][1]] = inputs.cpu().numpy()
#                                 detections = [(b['box_coords'], b['box_score']) for b in box]
#                                 patch_detections += [(grid_coord, detection, box_score) for
#                                                      detection, box_score in detections]
#                     detections, scores = collate_patches_object_detection(patch_detections)
#                     # Now we do NMS!
# #                     post_nms_detections = mutils.nms_numpy(detections[order], scores[order], 0.5)
#                     print(dataset[idx]['inputs_meta_dict'])
#                     print(detections)
#                     exit()
    if args.task == 'object_detection':
        
        for dataset in datasets:
            post_transforms = Compose([Invertd(
            keys="pred",
            transform=dataset.transform,
            orig_keys="inputs",
            meta_keys="pred_meta_dict",
            orig_meta_keys="inputs_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True)])
            val_org_loader = DataLoader(dataset, batch_size=1, num_workers=1)
            nifti_saver_pred = NiftiSaver(output_dir=inference_dir)
            nifti_saver_img = NiftiSaver(output_dir=inference_dir, output_postfix='img')
            nifti_saver_lab = NiftiSaver(output_dir=inference_dir, output_postfix='lab')
            with torch.no_grad():
                for val_data in val_org_loader:
                    val_data["pred"] = sliding_window_inference(
                        inputs=val_data['inputs'].to(model.device),
                        roi_size=args.spatial_size,
                        sw_batch_size=args.batch_size,
                        predictor=partial(model.inference_func, mode='box_out'),
                    )
                    val_data = [i for i in decollate_batch(val_data)]
                    print(val_data[0]['pred'].sum())
                    nifti_saver_pred.save(val_data[0]['pred'],
                                     meta_data={
                                         'filename_or_obj': Path(val_data[0]
                                                                 ['inputs_meta_dict'][
                                                                     'filename_or_obj']).name})
                    print(val_data[0]['labels'].sum())
                    nifti_saver_lab.save(val_data[0]['labels'],
                                     meta_data={
                                         'filename_or_obj': Path(val_data[0]
                                                                 ['inputs_meta_dict'][
                                                                     'filename_or_obj']).name})
                    nifti_saver_img.save(val_data[0]['inputs'],
                                     meta_data={
                                         'filename_or_obj': Path(val_data[0]
                                                                 ['inputs_meta_dict'][
                                                                     'filename_or_obj']).name})    
    # Here we call a postprocess_and_save function which has a 2D mode, 3D mode, object detection, segmentation
    # Each model can have a postprocess_and_save function?
    
    pass

def slice_based_inference(model, output_path, whole_volume_path, files_df, subject_id, batch_size=10):
    """
    Inner loop function to run inference on a single subject
    """
    subject_files_df = files_df[files_df['subject_id'] == subject_id]
    subject_files_df = subject_files_df.sort_values(by='slice_index')
    monai_data_list = [{'inputs': row['flair_path'],
                    'labels': row['label_path']}
                    for _, row in subject_files_df.iterrows()]
    transforms = Compose([LoadImaged(keys=['inputs', 'labels']),
                      Orientationd(keys=['inputs', 'labels'], axcodes='RAS'),
                      AddChanneld(keys=['inputs', 'labels']),
                      ToTensord(keys=['inputs', 'labels']),
                      ResizeWithPadOrCropd(keys=['inputs', 'labels'],
                                           spatial_size=(256, 256)),
                      SpatialCropd(keys=['inputs', 'labels'], roi_center=(127, 138), roi_size=(96, 96)),
                      ScaleIntensityd(keys=['inputs'], minv=0.0, maxv=1.0)
                     ])
    individual_subject_dataset = MonaiDataset(data=monai_data_list, transform=transforms)
    inference_dl = DataLoader(individual_subject_dataset, batch_size=batch_size, shuffle=False)
    inverse_op = BatchInverseTransform(transform=individual_subject_dataset.transform, loader=inference_dl)
    final_output = []
    nifti_saver = NiftiSaver(resample=False)
    img = nib.load(whole_volume_path)
    for idx, batch in enumerate(inference_dl):
        # Need to run inference here
        outputs, outputs2 = model.inference_func(batch['inputs'].to(model.device))
        tumour_preds = (torch.sigmoid(outputs) > 0.5).float().detach().cpu()
        cochlea_preds = (torch.sigmoid(outputs2) > 0.5).float().detach().cpu() * 2.0
        batch['inputs'] = torch.clamp(tumour_preds + cochlea_preds, min=0, max=2) # hack
        final_output.append(np.stack(
            [f['inputs'] for f in inverse_op(batch)]
        ))
    volume = np.einsum('ijkl->jkli', np.concatenate(final_output))[:, ::-1, ::-1, ...]
    nifti_saver = NiftiSaver(output_dir = Path(output_path).parent / 'mni_preds')
    nifti_saver.save(volume, meta_data={'affine': img.affine, 'filename_or_obj': Path(output_path).name})