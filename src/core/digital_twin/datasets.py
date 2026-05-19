import glob
import os

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset


def peek_shapes_hdf5(data_dir):
    files = glob.iglob(os.path.join(data_dir, "*.h5"))
    try:
        first_file = next(files)
    except StopIteration as exc:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}") from exc
    with h5.File(first_file, "r") as fin:
        data_shape = fin["features/data"].shape
        label_shape = fin["labels/pressure"].shape
    return data_shape, label_shape


def list_h5_files(source):
    files = sorted([os.path.join(source, x) for x in os.listdir(source) if x.endswith(".h5")])
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {source}")
    return files


class InMemoryRowDataset(Dataset):
    def __init__(
        self,
        source,
        allow_uneven_distribution=False,
        shuffle=False,
        size=1,
        rank=0,
        seed=12345,
    ):
        self.source = source
        self.allow_uneven_distribution = allow_uneven_distribution
        self.shuffle = shuffle
        self.size = size
        self.rank = rank
        self.rng = np.random.RandomState(seed)
        self.files = list_h5_files(self.source)
        if self.shuffle:
            self.rng.shuffle(self.files)

        data_blocks = []
        label_blocks = []
        feature_dims = []
        label_dims = []
        ref_mean = None
        ref_std = None

        for filename in self.files:
            with h5.File(filename, "r") as fin:
                data = fin["features/data"][...]
                label = fin["labels"][...]
                mean = fin["normalization/feature_mean"][...].reshape(-1)
                std_dev = fin["normalization/feature_std"][...].reshape(-1)

            if data.ndim != 2 or label.ndim != 2:
                raise ValueError(
                    f"Expected row-major 2D features/labels in {filename}, got {data.shape} and {label.shape}"
                )
            if data.shape[0] != label.shape[0]:
                raise ValueError(
                    f"Row mismatch in {filename}: features {data.shape} vs labels {label.shape}"
                )
            if mean.size != data.shape[1] or std_dev.size != data.shape[1]:
                raise ValueError(f"Normalization shape mismatch in {filename} for feature shape {data.shape}")
            if ref_mean is None:
                ref_mean = mean
                ref_std = std_dev
            else:
                if not np.allclose(mean, ref_mean):
                    raise ValueError(f"feature_mean differs across files; mismatch found in {filename}")
                if not np.allclose(std_dev, ref_std):
                    raise ValueError(f"feature_std differs across files; mismatch found in {filename}")

            data_blocks.append(data)
            label_blocks.append(label)
            feature_dims.append(data.shape[1])
            label_dims.append(label.shape[1])

        if len(set(feature_dims)) != 1 or len(set(label_dims)) != 1:
            raise ValueError("Inconsistent feature/label dimensions across files.")

        total_rows = sum(block.shape[0] for block in data_blocks)
        if self.allow_uneven_distribution:
            self.local_start = (self.rank * total_rows) // self.size
            self.local_end = ((self.rank + 1) * total_rows) // self.size
        else:
            num_rows_local = total_rows // self.size
            self.local_start = self.rank * num_rows_local
            self.local_end = self.local_start + num_rows_local

        self.global_size = total_rows if self.allow_uneven_distribution else self.size * (total_rows // self.size)
        self.local_size = self.local_end - self.local_start
        self.feature_dim = int(feature_dims[0])
        self.label_dim = int(label_dims[0])

        data_all = np.concatenate(data_blocks, axis=0)
        label_all = np.concatenate(label_blocks, axis=0)
        if np.any(ref_std == 0):
            raise ValueError("Found zeros in feature_std; cannot normalize safely.")
        data_all = (data_all - ref_mean) / ref_std

        # Materialize contiguous writable arrays and convert once to tensors so
        # DataLoader collation does not need to convert NumPy views.
        self.data = torch.from_numpy(
            np.array(
                data_all[self.local_start : self.local_end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        )
        self.labels = torch.from_numpy(
            np.array(
                label_all[self.local_start : self.local_end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        )
        self.data_shape = (self.feature_dim,)
        self.label_shape = (self.label_dim,)

    def __len__(self):
        return self.local_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.local_size:
            raise IndexError(f"Index {idx} out of range for local size {self.local_size}")
        return self.data[idx], self.labels[idx]
