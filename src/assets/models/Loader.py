#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:01:59 2023

@author: rodrigohadlich

github version check.
"""

import os
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from DatasetFunc import GetDataset, peek_shapes_hdf5


# helper function for determining the data shapes
def get_datashapes(root_dir):
    return peek_shapes_hdf5(os.path.join(root_dir, "train"))


# helper function to de-clutter the main script
def get_dataloader(root_dir, size, rank, batch_size, distributed=False):
    # import only what current worker needs
    train_dir = os.path.join(root_dir, "train")
    train_set = GetDataset(train_dir,
                           allow_uneven_distribution=False,
                           shuffle=True,
                           size=1,
                           rank=0)

    distributed_train_sampler = None
    if distributed:
        distributed_train_sampler = DistributedSampler(train_set,
                                                       num_replicas=size,
                                                       rank=rank,
                                                       shuffle=True,
                                                       drop_last=True)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=0,
                              sampler=distributed_train_sampler,
                              shuffle=not distributed,
                              pin_memory=False,
                              drop_last=True)

    print(f'THE LENGTH OF THE TRAIN LOADER IS {len(train_loader)}.')

    train_size = train_set.global_size

    validation_dir = os.path.join(root_dir, "validation")
    validation_set = GetDataset(validation_dir,
                                allow_uneven_distribution=True,
                                shuffle=False,
                                size=size,
                                rank=rank)

    # use batch size = 1 here to make sure we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   batch_size=1,
                                   num_workers=0,
                                   pin_memory=False,
                                   drop_last=False)

    validation_size = validation_set.global_size

    return train_loader, train_size, validation_loader, validation_size
