import numpy as np
import torch

class BaseModel:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        
    def inference(self):
        pass
#         # Dump of code to potentially re-introduce
#         performance_target_val_s, \
#         performance_target_val, \
#         performance_target_val_0, \
#         performance_target_val_ema = inference_func(args, p, model, target_val_dataset,
#                                                     prefix=os.path.join(results_folder, 'target_val'),
#                                                     iteration=iteration, infer_on=[0])
#         performance_source_val_s, \
#         performance_source_val, \
#         performance_source_val_0, \
#         performance_source_val_ema = inference_func(args, p, model, source_val_dataset,
#                                                     prefix=os.path.join(results_folder, 'source_val'),
#                                                     iteration=iteration, infer_on=[0])
#         performance_target_train_s, \
#         performance_target_train, \
#         performance_target_train_0, \
#         performance_target_train_ema = inference_func(args, p, model, target_test_dataset,
#                                                       prefix=os.path.join(results_folder,
#                                                                           'target_train'),
#                                                       iteration=iteration, infer_on=[0])  # hack
#         for key, value in postfix_dict.items():
#             writer.add_scalar('{}/train'.format(key), value, iteration)
#         for key, value in performance_source_val_s.items():
#             writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
#         if performance_source_val is not None:
#             for key, value in performance_source_val.items():
#                 writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
#         for key, value in performance_target_val_s.items():
#             writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
#         if performance_target_val is not None:
#             for key, value in performance_target_val.items():
#                 writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
#         for key, value in performance_target_train_s.items():
#             writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
#         if performance_target_train is not None:
#             for key, value in performance_target_train.items():
#                 writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
#         if performance_source_val_0 is not None:
#             for key, value in performance_source_val_0.items():
#                 writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
#         if performance_source_val_ema is not None:
#             for key, value in performance_source_val_ema.items():
#                 writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
#         if performance_target_val_0 is not None:
#             for key, value in performance_target_val_0.items():
#                 writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
#         if performance_target_val_ema is not None:
#             for key, value in performance_target_val_ema.items():
#                 writer.add_scalar('ema_{}/val_target'.format(key), np.mean(value), iteration)
#         if performance_target_train_0 is not None:
#             for key, value in performance_target_train_0.items():
#                 writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
#         if performance_target_train_ema is not None:
#             for key, value in performance_target_train_ema.items():
#                 writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
#         # writer.add_hparams(vars(args), {'hparam/' + key: value for key, value in postfix_dict.items()})
#         source_inputs_numpy = outputs.cpu().detach().numpy().flatten()
#         source_inputs_numpy = np.random.choice(source_inputs_numpy, size=1000)
#         source_inputs_numpy = 1 / (1 + np.exp(-source_inputs_numpy))  # Manual sigmoid
#         writer.add_histogram('outputs_source_hist', source_inputs_numpy, iteration)
#         if outputst is not None:
#             target_outputs_numpy = outputst.cpu().detach().numpy().flatten()
#             target_outputs_numpy = np.random.choice(target_outputs_numpy, size=1000)
#             target_outputs_numpy = 1 / (1 + np.exp(-target_outputs_numpy))  # Manual sigmoid
#             writer.add_histogram('outputs_target_hist', target_outputs_numpy, iteration)
#                         print('*** Target Val Performance ***')
#                         print(performance_target_val_s, performance_target_val,
#                               performance_target_val_0, performance_target_val_ema)
#                         print('*** Source Val Performance ***')
#                         print(performance_source_val_s, performance_source_val,
#                               performance_source_val_0, performance_source_val_ema)
#                         print('*** Target Train Performance ***')
#                         print(performance_target_train_s, performance_target_train,
#                               performance_target_val_0, performance_target_val_ema)

##### Discriminator validation ######
#         if args.method == 'adversarial':
#             correct_val, num_of_subjects_val = 0, 0
#             for i in range(20):
#                 source_val_slice_batch = next(source_val_slice_dl)
#                 source_val_slice_inputs = source_val_slice_batch['inputs'].to(device)
#                 source_val_slice_labels = source_val_slice_batch['labels'].to(device)
#                 target_val_slice_batch = next(target_val_slice_dl)
#                 target_val_slice_inputs = target_val_slice_batch['inputs'].to(device)
#                 target_val_slice_labels = target_val_slice_batch['labels'].to(device)
#                 discriminator.eval()
#                 inputs_source_discriminator = source_val_slice_inputs
#                 batch_trs = target_val_slice_inputs.cpu().numpy()
#                 batch_trs = p.map(
#                     partial(non_geometric_augmentations, method='bias', norm_training_images=None),
#                     np.copy(batch_trs))
#                 batch_trs = p.map(
#                     partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
#                     np.copy(batch_trs))
#                 inputs_target_discriminator_aug = torch.Tensor(batch_trs).cuda()

#                 Theta, Theta_inv = generate_affine(inputs_target_discriminator_aug,
#                                                   degreeFreedom=args.affine_rot_degree, scale=args.affine_scale,
#                                                   shearingScale=args.affine_shearing)
#                 inputs_target_discriminator_aug = apply_transform(inputs_target_discriminator_aug, Theta)

#                 inputs_models_discriminator = torch.cat(
#                     (inputs_source_discriminator, inputs_target_discriminator_aug), 0)
#                 labels_discriminator = to_var_gpu(
#                     torch.cat((torch.zeros(inputs_source_discriminator.size(0)),
#                                torch.ones(inputs_target_discriminator_aug.size(0))), 0).type(torch.LongTensor))
#                 _, _, _, _, _, _, dec4, dec3, dec2, dec1 = model(inputs_models_discriminator)
#                 dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
#                 dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
#                 dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
#                 dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')
#                 inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)
#                 outputs_discriminator = discriminator(inputs_discriminator)
#                 correct_val += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
#                 num_of_subjects_val += int(outputs_discriminator.size(0))
#             discriminator_val_accuracy = (correct_val / num_of_subjects_val).item()
#             writer.add_scalar('discriminator_acc_val', discriminator_val_accuracy, iteration)
#             ##############################################################
    
    def training_loop(self):
        pass
    
    def tensorboard_logging(self):
        pass
    
    def load(self):
        pass
    
    def save(self):
        pass
    
    def epoch_reset(self):
        pass