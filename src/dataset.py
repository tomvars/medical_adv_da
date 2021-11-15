import os
import nibabel as nib
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.utils import batch_adaptation, to_var_gpu
from src.custom_monai_transforms import BoundingBoxesd
from monai.data.dataset import PersistentDataset as MonaiDataset
from monai.data.nifti_saver import NiftiSaver
from monai.transforms import (LoadImaged,
                              Lambdad,
                              Orientationd,
                              SqueezeDimd,
                              ToTensord,
                              Compose,
                              AddChanneld,
                              ResizeWithPadOrCropd,
                              ScaleIntensityd,
                              SpatialCropd,
                              CropForegroundd,
                              CenterSpatialCropd,
                              BatchInverseTransform,
                              RandWeightedCropd,
                              MapLabelValued,
                              CopyItemsd,
                              RandAffined,
                              Spacingd,
                              RandSpatialCropd,
                              RandSpatialCropSamplesd,
                              RandBiasFieldd,
                              RandAdjustContrastd
                             )
import ants


def get_monai_slice_dataset(data_dir, paddtarget, slice_selection_method, dataset_split_csv, split,
                            spatial_dims=2, spatial_size=[256, 256], exclude_slices=None,
                            synthesis=True, tumour_only=False, bounding_boxes=False, 
                            return_aug=False, label_mapping=None):
    """
    This function object expects a data_dir which contains the following structure:
    data_dir
    |
    slices
        ----flair
              |
              ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
              ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
              ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
        ----labels
              |
              ----- wmh_<subject id as int>_slice_<slice id as int>.nii.gz
              ----- wmh_<subject id as int>_slice_<slice id as int>.nii.gz
              ----- wmh_<subject id as int>_slice_<slice id as int>.nii.gz
    """
    data_dir = os.path.join(data_dir, 'slices')
    assert slice_selection_method in ['mask', 'none']
    assert isinstance(paddtarget, int)
    assert 'flair' in os.listdir(data_dir)
    assert 'labels' in os.listdir(data_dir)
    dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'])
    flair_filenames = os.listdir(os.path.join(data_dir, 'flair'))
    subject_ids = [x.split('_slice')[0] for x in flair_filenames]
    slice_idx_arr = [int(x.split('_')[3].replace('.nii.gz', '')) for x in flair_filenames]
    label_paths = [os.path.join(data_dir, 'labels', x.replace('FLAIR', 'wmh')) for x in flair_filenames]
    flair_paths = [os.path.join(data_dir, 'flair', x) for x in flair_filenames]
    assert all([isinstance(x, int) for x in slice_idx_arr])
    files_df = pd.DataFrame(
        data=[(subj, slice_idx, fp, lp) for subj, slice_idx, fp, lp in zip(subject_ids, slice_idx_arr,
                                                                           label_paths, flair_paths)],
        columns=['subject_id', 'slice_index', 'label_path', 'flair_path']
    )
    if exclude_slices is not None:
        files_df = files_df[~files_df['slice_index'].isin(exclude_slices)]
    # Apply split filter to images
    images_to_use = dataset_split_df[dataset_split_df['split'] == split]['subject_id'].values
    files_df = files_df[files_df['subject_id'].isin(images_to_use)]
    monai_data_list = [{'inputs': row['flair_path'],
                        'labels': row['label_path']}
                        for _, row in files_df.iterrows()]
    transforms_list = [
        LoadImaged(keys=['inputs', 'labels']),
        Orientationd(keys=['inputs', 'labels'], axcodes='RAS'),
        AddChanneld(keys=['inputs', 'labels']),
        ToTensord(keys=['inputs', 'labels']),
        ResizeWithPadOrCropd(keys=['inputs', 'labels'], spatial_size=spatial_size),
        #SpatialCropd(keys=['inputs', 'labels'], roi_center=(127, 138), roi_size=(96, 96)),
#         ScaleIntensityd(keys=['inputs'], minv=0.0, maxv=1.0)
    ]
    if isinstance(label_mapping, dict):
        transforms_list.append(MapLabelValued(keys=['labels'],
                                              orig_labels=label_mapping.keys(),
                                              target_labels=label_mapping.values()))
    if bounding_boxes:
        transforms_list.append(BoundingBoxesd(keys=['labels']))
    if return_aug:
        transforms_list.append(CopyItemsd(keys=["inputs"], names=["inputs_aug"], times=1))
        transforms_list.append(RandAffined(
            keys=["inputs_aug"],
            allow_missing_keys=True,
            spatial_size=spatial_size,
            prob=1.0,
            rotate_range=0.1,
            shear_range=0.0,
            translate_range=(0.1, 0.1, 0.1),
            scale_range=[-0.1, 0.1],
        ))
    transforms = Compose(transforms_list)
    # Should normalise at the volume level
    return MonaiDataset(data=monai_data_list, transform=transforms)

def get_monai_patch_dataset(data_dir, cf, slice_selection_method, dataset_split_csv, split,
                            spatial_size=[128, 128, 24], spatial_dims=3, exclude_slices=None,
                            synthesis=True, bounding_boxes=False,
                            return_aug=False, label_mapping=None):
    """
    This function object expects a data_dir which contains the following structure:
    data_dir
    |
    whole/
        ----flair
              |
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
        ----labels
              |
              ----- wmh_<subject id as int>.nii.gz
              ----- wmh_<subject id as int>.nii.gz
              ----- wmh_<subject id as int>.nii.gz
    """
    data_dir = os.path.join(data_dir, 'whole')
    assert slice_selection_method in ['mask', 'none']
    assert 'flair' in os.listdir(data_dir)
    assert 'labels' in os.listdir(data_dir)
    dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'])
    flair_filenames = os.listdir(os.path.join(data_dir, 'flair'))
    subject_ids = np.array([x.replace('.nii.gz', '') for x in flair_filenames])
    flair_paths = [os.path.join(data_dir, 'flair', x) for x in flair_filenames]
    if split in ['train', 'val']:
        label_paths = [os.path.join(data_dir, 'labels', x.replace('FLAIR', 'wmh')) for x in flair_filenames]
        files_df = pd.DataFrame(
            data=[(subj, fp, lp) for subj, fp, lp in zip(subject_ids, label_paths, flair_paths)],
            columns=['subject_id', 'label_path', 'flair_path']
        )
        images_to_use = dataset_split_df[dataset_split_df['split'] == split]['subject_id'].values
        files_df = files_df[files_df['subject_id'].isin(images_to_use)]
        monai_data_list = [{'inputs': row['flair_path'],
                            'labels': row['label_path']}
                            for _, row in files_df.iterrows()]
        transforms_list = [
            LoadImaged(keys=['inputs', 'labels']),
            Orientationd(keys=['inputs', 'labels'], axcodes='RAS'),
            AddChanneld(keys=['inputs', 'labels']),
            Spacingd(keys=['inputs'], pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
            Spacingd(keys=['labels'], pixdim=[1.0, 1.0, 1.0], mode='nearest'),
            CopyItemsd(keys=['labels'], times=1, names=['weight_map']),
            CropForegroundd(keys=['inputs', 'labels'], source_key='inputs', margin=(20, 20, 0))
        ]
        if split == 'train' and cf.training_aug:
            # RandAffine when not doing ADA
            transforms_list.append(RandAffined(
                    keys=['inputs', 'labels'],
                    allow_missing_keys=True,
                    mode=['bilinear', 'nearest'],
#                     spatial_size=spatial_size,
                    prob=1.0,
                    rotate_range=[0, 0, 1.57],
                    shear_range=0.0,
                    translate_range=[10, 10, 10],
                    scale_range=[0.1, 0.1, 0.1],
            ))
#         transforms_list.append(
#             ResizeWithPadOrCropd(keys=['inputs', 'labels'], spatial_size=spatial_size, method="end", mode="constant", constant_values=(0,))
#         )
        if isinstance(cf.spatial_crop_center, list) and isinstance(cf.spatial_crop_roi, list):
            transforms_list.append(
                SpatialCropd(keys=['inputs', 'labels'],
                             roi_center=cf.spatial_crop_center,
                             roi_size=cf.spatial_crop_roi))
        transforms_list.append(
            RandSpatialCropSamplesd(keys=['inputs', 'labels'],
                          roi_size=spatial_size, num_samples=1, random_size=False))
        if isinstance(label_mapping, dict):
            transforms_list.append(MapLabelValued(keys=['labels'],
                                                  orig_labels=label_mapping.keys(),
                                                  target_labels=label_mapping.values()))
        if bounding_boxes:
            transforms_list.append(BoundingBoxesd(keys=['labels'], pad_bbox=0)) #2 for EPAD_SWI 
            #transforms_list.append(AddChanneld(keys=['inputs', 'labels']))
        if return_aug:
            transforms_list.append(CopyItemsd(keys=["inputs"], names=["inputs_aug"], times=1))
            transforms_list.append(RandBiasFieldd(keys=['inputs_aug'],
                                                  degree=3,
                                                  coeff_range=(0.0, 0.1),
                                                  prob=0.5))
            transforms_list.append(RandAdjustContrastd(keys=['inputs_aug'],
                                                       prob=0.5,
                                                       gamma=(0.5, 1.5)))
            transforms_list.append(RandAffined(
                keys=["inputs_aug"],
                allow_missing_keys=True,
                spatial_size=spatial_size,
                prob=1.0,
                rotate_range=0.1,
                shear_range=0.0,
                translate_range=(0.1, 0.1, 0.1),
                scale_range=[-0.1, 0.1],
            ))
            transforms_list.append(ScaleIntensityd(keys=['inputs_aug'], minv=0.0, maxv=1.0))
        transforms_list.append(ScaleIntensityd(keys=['inputs'], minv=0.0, maxv=1.0))
        transforms_list.append(ToTensord(keys=['inputs', 'labels']))
    else:
        files_df = pd.DataFrame(
            data=[(subj, lp) for subj, lp in zip(subject_ids, flair_paths)],
            columns=['subject_id', 'flair_path']
        )
        images_to_use = dataset_split_df['subject_id'].values
        files_df = files_df[files_df['subject_id'].isin(images_to_use)]
        monai_data_list = [{'inputs': row['flair_path']} for _, row in files_df.iterrows()]
        transforms_list = [
            LoadImaged(keys=['inputs']),
            Orientationd(keys=['inputs'], axcodes='RAS'),
            AddChanneld(keys=['inputs']),
            Spacingd(keys=['inputs'], pixdim=[1.0, 1.0, 1.0], mode='bilinear'),
            ScaleIntensityd(keys=['inputs'], minv=0.0, maxv=1.0),
            ToTensord(keys=['inputs'])
        ]

    
    # Should normalise at the volume level
    return MonaiDataset(data=monai_data_list, transform=Compose(transforms_list), cache_dir='/data2/tom/cache_dir')

def infer_on_subject(model, output_path, whole_volume_path, files_df, subject_id, batch_size=10):
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
    print(Path(output_path).parent / 'mni_preds')
    print(Path(output_path).parent)
    print(Path(output_path).name)
    nifti_saver.save(volume, meta_data={'affine': img.affine, 'filename_or_obj': Path(output_path).name})
    # Now save it in the original space
    seg_path = str(Path(output_path).parent / 'mni_preds'/ subject_id / f"{subject_id}_seg.nii.gz")
    img_path = whole_volume_path
    orig_img_path = whole_volume_path.replace('FLAIR_', 'orig_FLAIR_')
    orig_img = ants.image_read(orig_img_path)
    img = ants.image_read(img_path)
    seg = ants.image_read(seg_path)
    resampled_img = ants.resample_image_to_target(image=img, target=orig_img, interp_type='linear')
    resampled_seg = ants.resample_image_to_target(image=seg, target=orig_img, interp_type='nearestNeighbor')
    transforms = ants.registration(fixed=orig_img, moving=resampled_img, type_of_transform='SyN')
    transformed_seg = ants.apply_transforms(fixed=orig_img, moving=resampled_seg,
                                            transformlist=transforms['fwdtransforms'], interpolator='nearestNeighbor')
    transformed_img = ants.apply_transforms(fixed=orig_img, moving=resampled_img,
                                            transformlist=transforms['fwdtransforms'], interpolator='linear')
    subject_id = subject_id.split('_')[-1]
    submission_filename = f"crossmoda_{subject_id}_Label.nii.gz"
    output_path = Path(output_path).parent / 'submission_folder'
    output_path.mkdir(parents=True, exist_ok=True)
    if not (output_path / submission_filename).exists():
        transformed_seg.to_file(str(output_path / submission_filename))
    else:
        'File at {} already exists'.format(str(output_path / submission_filename))
        
def reweight_map(weight_map):
    weight_map[weight_map == 1] = 0.5
    weight_map[weight_map == 0] = 0.5/weight_map.size
    return weight_map