import random
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
    train_dataset,
    validation_dataset,
    batch_size: int,
    size: int,
    rank: int,
    *,
    distributed: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    train_drop_last: bool = True,
    validation_batch_size: int = 1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, int, DataLoader, int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    if num_workers > 0 and prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")

    distributed_train_sampler = None
    if distributed:
        distributed_train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=size,
            rank=rank,
            shuffle=True,
            drop_last=train_drop_last,
            seed=0 if seed is None else seed,
        )

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = partial(_seed_worker, base_seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=distributed_train_sampler,
        shuffle=not distributed,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        worker_init_fn=worker_init_fn,
        generator=generator,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
        **loader_kwargs,
    )

    train_size = len(train_dataset)
    validation_size = len(validation_dataset)
    return train_loader, train_size, validation_loader, validation_size
