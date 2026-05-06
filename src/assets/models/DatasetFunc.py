#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:04:57 2023

@author: rodrigohadlich
"""

import os
import glob
import h5py as h5
import numpy as np
from torch.utils.data import Dataset


def peek_shapes_hdf5(data_dir):
    files = glob.iglob(os.path.join(data_dir, '*.h5'))
    try:
        first_file = next(files)
    except StopIteration as exc:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}") from exc
    with h5.File(first_file, "r") as fin:
        data_shape = fin['features/data'].shape
        label_shape = fin['labels/pressure'].shape

    return data_shape, label_shape


# Dataset class
class GetDataset(Dataset):

    def _scan_file_metadata(self):
        row_counts = []
        feature_dims = []
        label_dims = []

        for filename in self.all_files:
            with h5.File(filename, "r") as fin:
                data_shape = fin['features/data'].shape
                label_shape = fin['labels/pressure'].shape
                mean_shape = fin['normalization/feature_mean'].shape
                std_shape = fin['normalization/feature_std'].shape

            if len(data_shape) != 2:
                raise ValueError(
                    f"Expected row-major 2D features in {filename}, got shape {data_shape}"
                )
            if len(label_shape) != 2:
                raise ValueError(
                    f"Expected row-major 2D labels in {filename}, got shape {label_shape}"
                )

            n_rows_data, n_features = data_shape
            n_rows_label, n_targets = label_shape
            if n_rows_data != n_rows_label:
                raise ValueError(
                    f"Row mismatch in {filename}: features {data_shape} vs labels {label_shape}"
                )

            mean_size = int(np.prod(mean_shape))
            std_size = int(np.prod(std_shape))
            if mean_size != n_features:
                raise ValueError(
                    f"Normalization mean shape {mean_shape} does not match features "
                    f"shape {data_shape} in {filename}"
                )
            if std_size != n_features:
                raise ValueError(
                    f"Normalization std shape {std_shape} does not match features "
                    f"shape {data_shape} in {filename}"
                )

            row_counts.append(n_rows_data)
            feature_dims.append(n_features)
            label_dims.append(n_targets)

        if not row_counts:
            raise FileNotFoundError(f"No .h5 files found in {self.source}")

        if len(set(feature_dims)) != 1:
            raise ValueError(f"Inconsistent feature dims across files: {feature_dims}")
        if len(set(label_dims)) != 1:
            raise ValueError(f"Inconsistent label dims across files: {label_dims}")

        self.row_counts = np.asarray(row_counts, dtype=np.int64)
        self.row_offsets = np.zeros(len(self.row_counts) + 1, dtype=np.int64)
        self.row_offsets[1:] = np.cumsum(self.row_counts)
        self.total_rows = int(self.row_offsets[-1])
        self.feature_dim = int(feature_dims[0])
        self.label_dim = int(label_dims[0])

    def init_reader(self):
        # shuffle files only, sample-level shuffle remains in DataLoader
        if self.shuffle:
            self.rng.shuffle(self.all_files)

        self.files = self.all_files
        self._scan_file_metadata()

        # shard by sample rows (not files)
        if self.allow_uneven_distribution:
            # covers dataset completely, some workers can have 1 extra sample
            self.local_start = (self.rank * self.total_rows) // self.size
            self.local_end = ((self.rank + 1) * self.total_rows) // self.size
            self.global_size = self.total_rows
        else:
            # equal rows per worker, potentially under-sampling tail rows
            num_rows_local = self.total_rows // self.size
            self.local_start = self.rank * num_rows_local
            self.local_end = self.local_start + num_rows_local
            self.global_size = self.size * num_rows_local

        self.local_size = self.local_end - self.local_start
        print(
            f"Rank {self.rank}: local rows [{self.local_start}, {self.local_end}) "
            f"of total {self.total_rows}"
        )

    def __init__(self,
                 source,
                 allow_uneven_distribution=False,
                 shuffle=False,
                 size=1,
                 rank=0,
                 seed=12345):
        self.source = source
        self.allow_uneven_distribution = allow_uneven_distribution
        self.shuffle = shuffle
        self.size = size
        self.rank = rank
        self.all_files = sorted([os.path.join(self.source, x) for x in os.listdir(self.source) if
                                 x.endswith('.h5')])  # set file format extension here
        if not self.all_files:
            raise FileNotFoundError(f"No .h5 files found in {self.source}")

        # create seed for shuffling files
        self.rng = np.random.RandomState(seed)

        # init reader
        self.init_reader()

        # per-sample feature/label shapes
        self.data_shape = (self.feature_dim,)
        self.label_shape = (self.label_dim,)

        if rank == 0:
            print(f'Initialized dataset with {self.global_size} samples. World size is {size}')

        print(f'Local dataset size in rank {self.rank} is {self.local_size}')

    def __len__(self):
        return self.local_size

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    def _resolve_global_index(self, idx):
        global_idx = self.local_start + int(idx)
        file_idx = int(np.searchsorted(self.row_offsets, global_idx, side='right') - 1)
        row_idx = int(global_idx - self.row_offsets[file_idx])
        return file_idx, row_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.local_size:
            raise IndexError(f"Index {idx} out of range for local size {self.local_size}")

        file_idx, row_idx = self._resolve_global_index(idx)
        global_idx = self.local_start + int(idx)
        filename = self.files[file_idx]

        # load data and project
        with h5.File(filename, "r") as f:
            data = f['features/data'][row_idx]
            label = f['labels/pressure'][row_idx]
            mean = f['normalization/feature_mean'][...].reshape(-1)
            std_dev = f['normalization/feature_std'][...].reshape(-1)

        # pre-process
        data = (data - mean) / std_dev

        return data, label, filename, global_idx
