"""Utility functions for shared memory buffer operations.

These functions are used for managing shared memory buffers between the master
and minion processes.
"""
from multiprocessing import shared_memory
import numpy as np
import torch


def get_indices(buf_arr, f_buf, *, logger=None, lock_index=5) -> tuple:
    """Return (write_idx, read_idx).
    
    Args:
        buf_arr: Buffer array
        f_buf: Flag buffer
        logger: Optional logger
        lock_index: Index in f_buf to use for locking (5 for actor, 6 for filter)
    """
    f_buf[lock_index] = 1  # lock episode buffer
    indices = buf_arr[:2].astype(np.int32)
    return indices


def set_indices(buf_arr: np.ndarray,
                idx: int,
                mode: str,
                f_buf: shared_memory.SharedMemory.buf,
                lock_index=5) -> None:
    """Atomically update either write or read index.
    
    Args:
        buf_arr: Buffer array
        idx: Index to set
        mode: 'w' for write index, 'r' for read index
        f_buf: Flag buffer
        lock_index: Index in f_buf to use for unlocking (5 for actor, 6 for filter)
    """
    # return struct.pack_into("<II", buf, 0, w, r)
    if mode == 'w':
        buf_arr[0] = idx
    elif mode == 'r':
        buf_arr[1] = idx
    else:
        raise ValueError(f'Unknown mode {mode}')
    f_buf[lock_index] = 0  # unlock episode buffer


def flatten_obs_onehot(obs, imep_space, mprr_space) -> torch.Tensor:
    """
    Function that takes in an observation from the environment and flattens to
    the shape required by the ort session.

    The ort session expects a one-hot encoding of the observation. RLlib converts
    observation spaces like the ones here to one-hot representation. There are
    built-in tools to handle this, but decided to with this implementation
    because it is faster and have more control. Also, the built-in tools weren't
    working.
    """
    # imep, mprr = obs["state"]
    tgt = obs

    # imep_idx = np.argmin(np.abs(imep - imep_space))
    # mprr_idx = np.argmin(np.abs(mprr - mprr_space))
    tgt_imep_idx = np.argmin(np.abs(tgt - imep_space))

    # imep_cur = np.eye(len(imep_space), dtype=np.float32)[imep_idx]  # (n_imep,)
    # mprr_cur = np.eye(len(mprr_space), dtype=np.float32)[mprr_idx]  # (n_mprr,)
    tgt_imep = np.eye(len(imep_space), dtype=np.float32)[tgt_imep_idx]  # (n_imep,)

    # return torch.tensor(np.concatenate([imep_cur, mprr_cur, imep_tgt]))
    return torch.tensor(tgt_imep, dtype=torch.float32)
