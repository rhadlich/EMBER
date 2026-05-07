#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:01:59 2023

@author: rodrigohadlich

github version check.
"""

import os
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from DatasetFunc import GetDataset, InMemoryRowDataset, PerFileDataset, peek_shapes_hdf5


# helper function for determining the data shapes
def get_datashapes(root_dir):
    return peek_shapes_hdf5(os.path.join(root_dir, "train"))


# helper function to de-clutter the main script
def _per_file_collate_fn(target_rows):
    def collate(batch):
        data_chunks = []
        label_chunks = []
        for data, label in batch:
            data_chunks.append(torch.as_tensor(data))
            label_chunks.append(torch.as_tensor(label))

        data_all = torch.cat(data_chunks, dim=0)
        label_all = torch.cat(label_chunks, dim=0)

        if data_all.shape[0] < target_rows:
            raise ValueError(
                f"Per-file collate received {data_all.shape[0]} rows, "
                f"which is fewer than target_rows={target_rows}"
            )

        data_all = data_all[:target_rows]
        label_all = label_all[:target_rows]
        return data_all, label_all

    return collate


def get_dataloader(
    root_dir,
    size,
    rank,
    batch_size,
    distributed=False,
    train_data_mode="row",
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    if prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

    # import only what current worker needs
    train_dir = os.path.join(root_dir, "train")
    rows_per_file = None
    files_per_batch = None
    if train_data_mode == "row":
        train_set = GetDataset(
            train_dir,
            allow_uneven_distribution=False,
            shuffle=True,
            size=1,
            rank=0,
        )
        per_file_collate = None
        loader_batch_size = batch_size
    elif train_data_mode == "in_memory_rows":
        train_set = InMemoryRowDataset(
            train_dir,
            allow_uneven_distribution=False,
            shuffle=True,
            size=1,
            rank=0,
        )
        per_file_collate = None
        loader_batch_size = batch_size
    elif train_data_mode == "per_file":
        train_set = PerFileDataset(train_dir, shuffle=True)
        unique_rows = sorted(set(train_set.rows_per_file))
        if len(unique_rows) != 1:
            raise ValueError(
                "Per-file mode requires a constant number of rows per file. "
                f"Found row counts: {unique_rows}"
            )
        rows_per_file = unique_rows[0]
        files_per_batch = math.ceil(batch_size / rows_per_file)
        loader_batch_size = max(1, files_per_batch)
        per_file_collate = _per_file_collate_fn(batch_size)
        print(
            f"Per-file mode: target_rows_per_step={batch_size}, "
            f"rows_per_file={rows_per_file}, files_per_batch={loader_batch_size}"
        )
    else:
        raise ValueError(
            f"Unsupported train_data_mode '{train_data_mode}'. "
            "Choose one of: row, in_memory_rows, per_file."
        )
    if train_set.global_size > train_set.total_rows:
        raise ValueError(
            f"Train dataset global_size={train_set.global_size} exceeds total_rows={train_set.total_rows}"
        )

    distributed_train_sampler = None
    if distributed:
        distributed_train_sampler = DistributedSampler(train_set,
                                                       num_replicas=size,
                                                       rank=rank,
                                                       shuffle=True,
                                                       drop_last=True)

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_set,
                              batch_size=loader_batch_size,
                              num_workers=num_workers,
                              sampler=distributed_train_sampler,
                              shuffle=not distributed,
                              pin_memory=pin_memory,
                              drop_last=True,
                              collate_fn=per_file_collate,
                              **loader_kwargs)
    train_loader.train_data_mode = train_data_mode
    train_loader.target_rows_per_step = int(batch_size)
    train_loader.rows_per_file = rows_per_file
    train_loader.files_per_batch = int(files_per_batch) if files_per_batch is not None else None

    print(f'THE LENGTH OF THE TRAIN LOADER IS {len(train_loader)}.')
    local_rows = getattr(train_set, "local_size", train_set.global_size)
    print(
        f"Train rows (expected): total={train_set.total_rows}, "
        f"used_by_dataset={train_set.global_size}, local={local_rows}"
    )

    train_size = train_set.global_size

    validation_dir = os.path.join(root_dir, "validation")
    validation_set = GetDataset(validation_dir,
                                allow_uneven_distribution=True,
                                shuffle=False,
                                size=size,
                                rank=rank)
    if validation_set.global_size != validation_set.total_rows:
        raise ValueError(
            f"Validation dataset expected full coverage: global_size={validation_set.global_size}, "
            f"total_rows={validation_set.total_rows}"
        )

    # use batch size = 1 here to make sure we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   batch_size=1,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   drop_last=False,
                                   **loader_kwargs)

    validation_size = validation_set.global_size
    print(
        f"Validation rows (expected): total={validation_set.total_rows}, "
        f"global={validation_set.global_size}, local={validation_set.local_size}"
    )

    return train_loader, train_size, validation_loader, validation_size
