import os

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset


def peek_hdf5(data_dir: str) -> tuple[int, int, int]:
    files = list_h5_files(data_dir)
    first_file = files[0]
    with h5.File(first_file, "r") as fin:
        states_shape = tuple(fin["features/data"].shape)
        actions_shape = tuple(fin["features/action"].shape)
        next_states, _ = _read_first_available(
            fin,
            ["labels", "labels/next_states", "labels/next_state", "next_states"],
            filename=first_file,
        )
        next_states_shape = tuple(next_states.shape)

    if len(states_shape) != 2 or len(actions_shape) != 2 or len(next_states_shape) != 2:
        raise ValueError(
            f"Expected row-major 2D arrays in {first_file}, "
            f"got states={states_shape}, actions={actions_shape}, next_states={next_states_shape}"
        )

    state_dim = int(states_shape[1])
    action_dim = int(actions_shape[1])
    output_dim = int(next_states_shape[1])
    return state_dim, action_dim, output_dim


def list_h5_files(source: str):
    files = sorted([os.path.join(source, x) for x in os.listdir(source) if x.endswith(".h5")])
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {source}")
    return files


def _read_first_available(fin, keys: list[str], *, filename: str):
    def _read_node(node, key: str):
        if isinstance(node, h5.Dataset):
            return node[...], key
        if isinstance(node, h5.Group):
            dataset_children = [name for name in node.keys() if isinstance(node[name], h5.Dataset)]
            if len(dataset_children) == 1:
                child = dataset_children[0]
                return node[child][...], f"{key}/{child}"
            raise ValueError(
                f"Path '{key}' in {filename} is a group with {len(dataset_children)} dataset children. "
                "Expected a dataset or a group with exactly one dataset child."
            )
        raise TypeError(f"Path '{key}' in {filename} is not a dataset/group.")

    for key in keys:
        if key in fin:
            return _read_node(fin[key], key)
    raise KeyError(f"None of keys {keys} found in {filename}")


class SafetyInMemoryRowDataset(Dataset):
    """
    In-memory row dataset for StatePredictor training from HDF5 files.
    """

    def __init__(
        self,
        source: str,
        *,
        allow_uneven_distribution: bool = False,
        shuffle: bool = False,
        size: int = 1,
        rank: int = 0,
        seed: int = 12345,
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

        states_blocks = []
        actions_blocks = []
        next_states_blocks = []
        state_dims = []
        action_dims = []
        next_state_dims = []

        ref_action_max = None
        ref_action_min = None
        ref_feature_mean = None
        ref_feature_std = None

        for filename in self.files:
            with h5.File(filename, "r") as fin:
                states = fin["features/data"][...]
                actions = fin["features/action"][...]
                next_states, _ = _read_first_available(
                    fin,
                    ["labels", "labels/next_states", "labels/next_state", "next_states"],
                    filename=filename,
                )
                action_max = fin["normalization/action_max"][...].reshape(-1)
                action_min = fin["normalization/action_min"][...].reshape(-1)
                feature_mean = fin["normalization/feature_mean"][...].reshape(-1)
                feature_std = fin["normalization/feature_std"][...].reshape(-1)

            if states.ndim != 2 or actions.ndim != 2 or next_states.ndim != 2:
                raise ValueError(
                    f"Expected row-major 2D arrays in {filename}, "
                    f"got states={states.shape}, actions={actions.shape}, next_states={next_states.shape}"
                )
            if states.shape[0] != actions.shape[0] or states.shape[0] != next_states.shape[0]:
                raise ValueError(
                    f"Row mismatch in {filename}: "
                    f"states={states.shape}, actions={actions.shape}, next_states={next_states.shape}"
                )
            if action_max.size != actions.shape[1] or action_min.size != actions.shape[1]:
                raise ValueError(
                    f"Action normalization shape mismatch in {filename} for action shape {actions.shape}"
                )
            if feature_mean.size != states.shape[1] or feature_std.size != states.shape[1]:
                raise ValueError(
                    f"Feature normalization shape mismatch in {filename} for state shape {states.shape}"
                )

            if ref_action_max is None:
                ref_action_max = action_max
                ref_action_min = action_min
            else:
                if not np.allclose(action_max, ref_action_max):
                    raise ValueError(f"action_max differs across files; mismatch found in {filename}")
                if not np.allclose(action_min, ref_action_min):
                    raise ValueError(f"action_min differs across files; mismatch found in {filename}")
            if ref_feature_mean is None:
                ref_feature_mean = feature_mean
                ref_feature_std = feature_std
            else:
                if not np.allclose(feature_mean, ref_feature_mean):
                    raise ValueError(f"feature_mean differs across files; mismatch found in {filename}")
                if not np.allclose(feature_std, ref_feature_std):
                    raise ValueError(f"feature_std differs across files; mismatch found in {filename}")

            states_blocks.append(states)
            actions_blocks.append(actions)
            next_states_blocks.append(next_states)
            state_dims.append(states.shape[1])
            action_dims.append(actions.shape[1])
            next_state_dims.append(next_states.shape[1])

        if len(set(state_dims)) != 1 or len(set(action_dims)) != 1 or len(set(next_state_dims)) != 1:
            raise ValueError("Inconsistent state/action/next_state dimensions across files.")

        states_all = np.concatenate(states_blocks, axis=0)
        actions_all = np.concatenate(actions_blocks, axis=0)
        next_states_all = np.concatenate(next_states_blocks, axis=0)

        if np.any(ref_feature_std == 0):
            raise ValueError("Found zeros in feature_std; cannot normalize safely.")
        states_all = (states_all - ref_feature_mean) / ref_feature_std

        action_range = ref_action_max - ref_action_min
        if np.any(action_range == 0):
            raise ValueError("Found zeros in (action_max - action_min); cannot normalize safely.")
        actions_all = 2.0 * (actions_all - ref_action_min) / action_range - 1.0

        total_rows = states_all.shape[0]
        if self.allow_uneven_distribution:
            self.local_start = (self.rank * total_rows) // self.size
            self.local_end = ((self.rank + 1) * total_rows) // self.size
        else:
            num_rows_local = total_rows // self.size
            self.local_start = self.rank * num_rows_local
            self.local_end = self.local_start + num_rows_local

        self.global_size = (
            total_rows if self.allow_uneven_distribution else self.size * (total_rows // self.size)
        )
        self.local_size = self.local_end - self.local_start
        self.state_dim = int(state_dims[0])
        self.action_dim = int(action_dims[0])
        self.next_state_dim = int(next_state_dims[0])

        self.states = torch.from_numpy(
            np.array(
                states_all[self.local_start : self.local_end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        )
        self.actions = torch.from_numpy(
            np.array(
                actions_all[self.local_start : self.local_end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        )
        self.next_states = torch.from_numpy(
            np.array(
                next_states_all[self.local_start : self.local_end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        )

    def __len__(self):
        return self.local_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.local_size:
            raise IndexError(f"Index {idx} out of range for local size {self.local_size}")
        return (self.states[idx], self.actions[idx]), self.next_states[idx]
