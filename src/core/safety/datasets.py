import numpy as np
from torch.utils.data import Dataset


class SafetyInMemoryRowDataset(Dataset):
    """
    In-memory dataset for StatePredictor training.

    Subclass this class and override `process_raw_arrays` or `process_row` to add
    model-specific preprocessing from raw data.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        states_key: str = "states",
        actions_key: str = "actions",
        next_states_key: str = "next_states",
        allow_uneven_distribution: bool = False,
        size: int = 1,
        rank: int = 0,
    ):
        payload = np.load(dataset_path)
        states = payload[states_key]
        actions = payload[actions_key]
        next_states = payload[next_states_key]
        states, actions, next_states = self.process_raw_arrays(states, actions, next_states)

        if states.shape[0] != actions.shape[0] or states.shape[0] != next_states.shape[0]:
            raise ValueError("states, actions, and next_states must share the same first dimension.")

        total_rows = int(states.shape[0])
        if allow_uneven_distribution:
            local_start = (rank * total_rows) // size
            local_end = ((rank + 1) * total_rows) // size
            self.global_size = total_rows
        else:
            num_rows_local = total_rows // size
            local_start = rank * num_rows_local
            local_end = local_start + num_rows_local
            self.global_size = size * num_rows_local

        self.states = states[local_start:local_end].astype(np.float32)
        self.actions = actions[local_start:local_end].astype(np.float32)
        self.next_states = next_states[local_start:local_end].astype(np.float32)
        self.local_size = int(self.states.shape[0])

    def process_raw_arrays(self, states, actions, next_states):
        return states, actions, next_states

    def process_row(self, state, action, next_state):
        return state, action, next_state

    def __len__(self):
        return self.local_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.local_size:
            raise IndexError(f"Index {idx} out of range for local size {self.local_size}")
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]
        state, action, next_state = self.process_row(state, action, next_state)
        return (state, action), next_state
