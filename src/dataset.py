import os
import nibabel as nib
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.utils import batch_adaptation, to_var_gpu
from monai.data.dataset import Dataset as MonaiDataset
from monai.data.nifti_saver import NiftiSaver
from monai.transforms import (LoadImaged,
                              Orientationd,
                              ToTensord,
                              Compose,
                              AddChanneld,
                              ResizeWithPadOrCropd,
                              ScaleIntensityd,
                              SpatialCropd,
                              BatchInverseTransform
                             )
import ants


class SliceDataset(Dataset):
    """
    This Dataset object expects a data_dir which contains the following structure:
    data_dir
    |
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

    def __init__(self, data_dir, paddtarget, slice_selection_method, dataset_split_csv, split,
                 exclude_slices=None, synthesis=True, tumour_only=False):
        assert slice_selection_method in ['mask', 'none']
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'])
        self.split = split
        self.paddtarget = paddtarget
        self.slice_selection_method = slice_selection_method
        self.synthesis = synthesis
        flair_filenames = os.listdir(os.path.join(data_dir, 'flair'))
        subject_ids = [x.split('_slice')[0] for x in flair_filenames]
        slice_idx_arr = [int(x.split('_')[3].replace('.nii.gz', '')) for x in flair_filenames]
        label_paths = [os.path.join(data_dir, 'labels', x.replace('FLAIR', 'wmh')) for x in flair_filenames]
        flair_paths = [os.path.join(data_dir, 'flair', x) for x in flair_filenames]
        assert all([isinstance(x, int) for x in slice_idx_arr])
        self.files_df = pd.DataFrame(
            data=[(subj, slice_idx, fp, lp) for subj, slice_idx, fp, lp in zip(subject_ids, slice_idx_arr,
                                                                               label_paths, flair_paths)],
            columns=['subject_id', 'slice_index', 'label_path', 'flair_path']
        )
        if exclude_slices is not None:
            self.files_df = self.files_df[~self.files_df['slice_index'].isin(exclude_slices)]
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        flair_slice = nib.load(flair_filepath).get_data()
        label_slice = nib.load(label_filepath).get_data()
        
        batch = {'inputs': torch.tensor(flair_slice).unsqueeze(dim=0).unsqueeze(dim=1).to(torch.float),
                 'labels': torch.tensor(label_slice).unsqueeze(dim=0).unsqueeze(dim=1).to(torch.float)}
        batch['inputs'] = to_var_gpu(batch['inputs'][:, 0, ...])
        batch['labels'] = to_var_gpu(batch['labels'][:, 0, ...])
        return batch

    def __len__(self):
        return len(self.files_df)

    def get_slice_indices_for_subject_ids(self, subject_ids):
        return self.files_df[self.files_df['subject_id'].isin(subject_ids)].index.values


class WholeVolumeDataset(Dataset):
    """
        This Dataset object expects a data_dir which contains the following structure:
        data_dir
        |
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

    def __init__(self, data_dir, paddtarget, dataset_split_csv, split, synthesis=True, tumour_only=False):
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'])
        self.split = split
        flair_filenames = os.listdir(os.path.join(data_dir, 'flair'))
        subject_ids = np.array([x.replace('.nii.gz', '') for x in flair_filenames])
        flair_paths = [os.path.join(data_dir, 'flair', x) for x in flair_filenames]
        label_paths = [os.path.join(data_dir, 'labels', x.replace('FLAIR', 'wmh')) for x in flair_filenames]
        self.files_df = pd.DataFrame(
            data=[(subj, fp, lp) for subj, fp, lp in zip(subject_ids, label_paths, flair_paths)],
            columns=['subject_id', 'label_path', 'flair_path']
        )
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        inputs = nib.load(flair_filepath).get_data()
        labels = nib.load(label_filepath).get_data()
#        batch = {'inputs': inputs, 'labels': labels}
        return inputs, labels

    def __len__(self):
        return len(self.files_df)

    def get_subject_id_from_index(self, index):
        return self.files_df[
            self.files_df['flair_path'] == self.files_df['flair_path'].values[index]]['subject_id'].values[0]

class WholeVolumeDatasetTumour(Dataset):
    """
        This Dataset object expects a data_dir which contains the following structure:
        data_dir
        |
        ----flair
              |
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
        ----T1c
              |
              ----- T1c_<subject id as int>.nii.gz
              ----- T1c_<subject id as int>.nii.gz
              ----- T1c_<subject id as int>.nii.gz
        ----T1
              |
              ----- T1_<subject id as int>.nii.gz
              ----- T1_<subject id as int>.nii.gz
              ----- T1_<subject id as int>.nii.gz
        ----T2
              |
              ----- T2_<subject id as int>.nii.gz
              ----- T2_<subject id as int>.nii.gz
              ----- T2_<subject id as int>.nii.gz
        ----labels
              |
              ----- bin_<subject id as int>.nii.gz
              ----- bin_<subject id as int>.nii.gz
              ----- bin_<subject id as int>.nii.gz
        """
    def __init__(self, data_dir, paddtarget, dataset_split_csv, split, tumour_only=False):
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        subject_id_arr = np.array(['_'.join(x.split('_')[1:]).replace('.nii.gz', '') for x in os.listdir(os.path.join(data_dir, 'flair'))])
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'], dtype={'subject_id': str})
        self.split = split
        self.tumour_only = tumour_only
        flair_paths = [os.path.join(data_dir, 'flair', 'FLAIR_' + str(id) + '.nii.gz') for id in subject_id_arr]
        t1c_paths = [os.path.join(data_dir, 't1c', 'T1c_' + str(id) + '.nii.gz') for id in subject_id_arr]
        t1_paths = [os.path.join(data_dir, 't1', 'T1_' + str(id) + '.nii.gz') for id in subject_id_arr]
        t2_paths = [os.path.join(data_dir, 't2', 'T2_' + str(id) + '.nii.gz') for id in subject_id_arr]
        label_dir = 'labels' if not tumour_only else 'labels_tumour_only'
        label_paths = [os.path.join(data_dir, label_dir, 'bin_' + str(id) + '.nii.gz') for id in subject_id_arr]

        self.files_df = pd.DataFrame(
            data=[tup for tup in zip(subject_id_arr, label_paths, flair_paths, t1c_paths, t1_paths, t2_paths)],
            columns=['subject_id', 'label_path', 'flair_path', 't1c_path', 't1_path', 't2_path']
        )
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        t1c_filepath = self.files_df['t1c_path'].values[index]
        t1_filepath = self.files_df['t1_path'].values[index]
        t2_filepath = self.files_df['t2_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        flair = nib.load(flair_filepath).get_data()
        t1c = nib.load(t1c_filepath).get_data()
        t1 = nib.load(t1_filepath).get_data()
        t2 = nib.load(t2_filepath).get_data()
        labels = nib.load(label_filepath).get_data()
        inputs = np.stack([flair, t1c, t1, t2], axis=0)
        batch = {'inputs': inputs, 'labels': labels}
        batch['inputs'] = to_var_gpu(batch['inputs'][:, 0, ...])
        batch['labels'] = to_var_gpu(batch['labels'][:, 0, ...])
        return batch

    def __len__(self):
        return len(self.files_df)

    def get_subject_id_from_index(self, index):
        return self.files_df[
            self.files_df['flair_path'] == self.files_df['flair_path'].values[index]]['subject_id'].values[0]


class SliceDatasetTumour(Dataset):
    """
    This Dataset object expects a data_dir which contains the following structure:
    data_dir
    |
    ----flair
          |
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
    ----T1c
          |
          ----- T1c_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1c_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1c_<subject id as int>_slice_<slice id as int>.nii.gz
    ----T1
          |
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
    ----T2
          |
          ----- T2_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T2_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T2_<subject id as int>_slice_<slice id as int>.nii.gz
    ----labels
          |
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
    """
    def __init__(self, data_dir, paddtarget, slice_selection_method, dataset_split_csv, split, tumour_only=False):
        assert slice_selection_method in ['mask', 'none']
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        self.paddtarget = paddtarget
        self.slice_selection_method = slice_selection_method
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'], dtype={'subject_id': str})
        self.split = split
        self.tumour_only = tumour_only
        subject_basenames = ['_'.join(p.split('_')[1:]) for p in os.listdir(os.path.join(data_dir, 'flair'))]
        label_dir = 'labels' if not tumour_only else 'labels_tumour_only'
        label_paths = [os.path.join(data_dir, label_dir, 'bin_' + x) for x in subject_basenames]
        flair_paths = [os.path.join(data_dir, 'flair', 'FLAIR_' + x) for x in subject_basenames]
        t1c_paths = [os.path.join(data_dir, 't1c', 'T1c_' + x) for x in subject_basenames]
        t1_paths = [os.path.join(data_dir, 't1', 'T1_' + x) for x in subject_basenames]
        t2_paths = [os.path.join(data_dir, 't2', 'T2_' + x) for x in subject_basenames]
        subject_id_arr = [x.split('_')[0] for x in subject_basenames]
        slice_idx_arr = [int(x.split('_')[2].replace('.nii.gz', '')) for x in subject_basenames]
        assert all([isinstance(x, int) for x in slice_idx_arr])
        self.files_df = pd.DataFrame(
            data=[tp for tp in zip(subject_id_arr, slice_idx_arr, label_paths,
                                   flair_paths,t1c_paths, t1_paths, t2_paths)],
            columns=['subject_id', 'slice_index', 'label_path', 'flair_path', 't1c_path', 't1_path', 't2_path']
        )
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        t1c_filepath = self.files_df['t1c_path'].values[index]
        t1_filepath = self.files_df['t1_path'].values[index]
        t2_filepath = self.files_df['t2_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        flair_slice = nib.load(flair_filepath).get_data()
        t1c_slice = nib.load(t1c_filepath).get_data()
        t1_slice = nib.load(t1_filepath).get_data()
        t2_slice = nib.load(t2_filepath).get_data()
        label_slice = nib.load(label_filepath).get_data()
        image_slice = np.stack([flair_slice, t1c_slice, t1_slice, t2_slice], axis=0)
        inputs, labels = batch_adaptation(torch.tensor(image_slice).unsqueeze(dim=0),
                                          torch.tensor(label_slice), self.paddtarget)
        batch = {'inputs': inputs, 'labels': labels} 
        batch['inputs'] = to_var_gpu(batch['inputs'][:, 0, ...])
        batch['labels'] = to_var_gpu(batch['labels'][:, 0, ...])
        return batch

    def __len__(self):
        return len(self.files_df)

    def get_slice_indices_for_subject_ids(self, subject_ids):
        return self.files_df[self.files_df['subject_id'].isin(subject_ids)].index.values

class WholeVolumeDatasetMS(Dataset):
    """
        This Dataset object expects a data_dir which contains the following structure:
        data_dir
        |
        ----flair
              |
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
              ----- FLAIR_<subject id as int>.nii.gz
        ----T1
              |
              ----- T1_<subject id as int>.nii.gz
              ----- T1_<subject id as int>.nii.gz
              ----- T1_<subject id as int>.nii.gz
        ----labels
              |
              ----- bin_<subject id as int>.nii.gz
              ----- bin_<subject id as int>.nii.gz
              ----- bin_<subject id as int>.nii.gz
        """
    def __init__(self, data_dir, paddtarget, dataset_split_csv, split):
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        subject_id_arr = np.array(['_'.join(x.split('_')[1:]).replace('.nii.gz', '') for x in os.listdir(os.path.join(data_dir, 'flair'))])
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'], dtype={'subject_id': str})
        self.split = split
        flair_paths = [os.path.join(data_dir, 'flair', 'FLAIR_' + str(id) + '.nii.gz') for id in subject_id_arr]
        t1_paths = [os.path.join(data_dir, 't1', 'T1_' + str(id) + '.nii.gz') for id in subject_id_arr]
        label_paths = [os.path.join(data_dir, 'labels', 'bin_' + str(id) + '.nii.gz') for id in subject_id_arr]

        self.files_df = pd.DataFrame(
            data=[tup for tup in zip(subject_id_arr, label_paths, flair_paths, t1_paths)],
            columns=['subject_id', 'label_path', 'flair_path', 't1_path']
        )
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        t1_filepath = self.files_df['t1_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        flair = nib.load(flair_filepath).get_data()
        t1 = nib.load(t1_filepath).get_data()
        labels = nib.load(label_filepath).get_data()
        inputs = np.stack([flair, t1], axis=0)
        batch = {'inputs': inputs, 'labels': labels} 
        batch['inputs'] = to_var_gpu(batch['inputs'][:, 0, ...])
        batch['labels'] = to_var_gpu(batch['labels'][:, 0, ...])
        return batch

    def __len__(self):
        return len(self.files_df)

    def get_subject_id_from_index(self, index):
        return self.files_df[
            self.files_df['flair_path'] == self.files_df['flair_path'].values[index]]['subject_id'].values[0]


class SliceDatasetMS(Dataset):
    """
    This Dataset object expects a data_dir which contains the following structure:
    data_dir
    |
    ----flair
          |
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- FLAIR_<subject id as int>_slice_<slice id as int>.nii.gz
    ----T1
          |
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- T1_<subject id as int>_slice_<slice id as int>.nii.gz
    ----labels
          |
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
          ----- bin_<subject id as int>_slice_<slice id as int>.nii.gz
    """
    def __init__(self, data_dir, paddtarget, slice_selection_method, dataset_split_csv, split):
        assert slice_selection_method in ['mask', 'none']
        assert isinstance(paddtarget, int)
        assert 'flair' in os.listdir(data_dir)
        assert 'labels' in os.listdir(data_dir)
        self.paddtarget = paddtarget
        self.slice_selection_method = slice_selection_method
        self.dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'], dtype={'subject_id': str})
        self.split = split
        subject_basenames = ['_'.join(p.split('_')[1:]) for p in os.listdir(os.path.join(data_dir, 'flair'))]
        label_paths = [os.path.join(data_dir, 'labels', 'bin_' + x) for x in subject_basenames]
        flair_paths = [os.path.join(data_dir, 'flair', 'FLAIR_' + x) for x in subject_basenames]
        t1_paths = [os.path.join(data_dir, 't1', 'T1_' + x) for x in subject_basenames]
        subject_id_arr = [x.split('_')[0] for x in subject_basenames]
        slice_idx_arr = [int(x.split('_')[2].replace('.nii.gz', '')) for x in subject_basenames]
        assert all([isinstance(x, int) for x in slice_idx_arr])
        self.files_df = pd.DataFrame(
            data=[tp for tp in zip(subject_id_arr, slice_idx_arr, label_paths,
                                   flair_paths, t1_paths)],
            columns=['subject_id', 'slice_index', 'label_path', 'flair_path', 't1_path']
        )
        # Apply split filter to images
        images_to_use = self.dataset_split_df[self.dataset_split_df['split'] == self.split]['subject_id'].values
        self.files_df = self.files_df[self.files_df['subject_id'].isin(images_to_use)]

    def __getitem__(self, index):
        flair_filepath = self.files_df['flair_path'].values[index]
        t1_filepath = self.files_df['t1_path'].values[index]
        label_filepath = self.files_df['label_path'].values[index]
        flair_slice = nib.load(flair_filepath).get_data()
        t1_slice = nib.load(t1_filepath).get_data()
        label_slice = nib.load(label_filepath).get_data()
        image_slice = np.stack([flair_slice, t1_slice], axis=0)
        inputs, labels = batch_adaptation(torch.tensor(image_slice).unsqueeze(dim=0),
                                          torch.tensor(label_slice), self.paddtarget)
        batch = {'inputs': inputs, 'labels': labels} 
        batch['inputs'] = to_var_gpu(batch['inputs'][:, 0, ...])
        batch['labels'] = to_var_gpu(batch['labels'][:, 0, ...])
        return batch

    def __len__(self):
        return len(self.files_df)

    def get_slice_indices_for_subject_ids(self, subject_ids):
        return self.files_df[self.files_df['subject_id'].isin(subject_ids)].index.values

class SubsetTumour(WholeVolumeDatasetTumour):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class Subset(WholeVolumeDataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_monai_slice_dataset(data_dir, paddtarget, slice_selection_method, dataset_split_csv, split,
                 exclude_slices=None, synthesis=True, tumour_only=False):
    """
    This function object expects a data_dir which contains the following structure:
    data_dir
    |
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
    assert slice_selection_method in ['mask', 'none']
    assert isinstance(paddtarget, int)
    assert 'flair' in os.listdir(data_dir)
    assert 'labels' in os.listdir(data_dir)
    dataset_split_df = pd.read_csv(dataset_split_csv, names=['subject_id', 'split'])
    split = split
    paddtarget = paddtarget
    slice_selection_method = slice_selection_method
    synthesis = synthesis
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
    # Need to return a Monai Dataset object
    # Expects data to be an array of dictionaries e.g
    # [{                            {                            {
    #         'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
    #         'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
    #         'extra': 123                 'extra': 456                 'extra': 789
    #     },
    monai_data_list = [{'inputs': row['flair_path'],
                        'labels': row['label_path']}
                        for _, row in files_df.iterrows()]
    transforms = Compose([LoadImaged(keys=['inputs', 'labels']),
                          Orientationd(keys=['inputs', 'labels'], axcodes='RAS'),
                          AddChanneld(keys=['inputs', 'labels']),
                          ToTensord(keys=['inputs', 'labels']),
                          ResizeWithPadOrCropd(keys=['inputs', 'labels'],
                                               spatial_size=(256, 256)),
                          SpatialCropd(keys=['inputs', 'labels'], roi_center=(127, 138), roi_size=(96, 96)),
                          ScaleIntensityd(keys=['inputs'], minv=0.0, maxv=1.0)
                         ])
    # Should normalise at the volume level
    return MonaiDataset(data=monai_data_list, transform=transforms)

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
    transforms = ants.registration(fixed=orig_img, moving=resampled_img, type_of_transform='Affine')
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