from typing import Tuple

from torch.utils.data import DataLoader, DistributedSampler


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
        )

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=distributed_train_sampler,
        shuffle=not distributed,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **loader_kwargs,
    )

    train_size = len(train_dataset)
    validation_size = len(validation_dataset)
    return train_loader, train_size, validation_loader, validation_size
