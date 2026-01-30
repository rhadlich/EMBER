import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from typing import Callable, Optional, Tuple, List
import random
import logging
import utils.logging_setup as logging_setup

class StatePredictor(nn.Module):
    def __init__(self,
    state_dim,
    action_dim,
    num_hidden,
    hidden_exp,
    dropout,
    ):
        super(StatePredictor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int((2 ** hidden_exp) / (1-dropout))

        layers = []
        layers += [nn.Linear(state_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), self.dropout]
        layers += [nn.Linear(hidden_dim, state_dim+state_dim*action_dim)]
        self.network = nn.Sequential(*layers)

    def forward(self, x, u):
        # x is of shape (batch_size, state_dim)
        # u is of shape (batch_size, action_dim)
        # return is of shape (batch_size, state_dim)

        # forward pass through the network
        x = self.network(x)

        # split network output into state equation components
        f_x = x[:, :self.state_dim]
        G_x = x[:, self.state_dim:]
        G_x = torch.reshape(G_x, (G_x.shape[0], self.state_dim, self.action_dim))

        # compute state equation
        x_next = f_x + torch.bmm(G_x, u.unsqueeze(-1)).squeeze(-1)

        return x_next, f_x, G_x


class SafetyFilter:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 ort_session: Optional[ort.InferenceSession] = None,
                 input_names: Optional[List[str]] = None,
                 output_names: Optional[List[str]] = None) -> None:
        """
        Initialize SafetyFilter.
        
        - If `ort_session` is provided, this instance can run inference and compute filtered actions.
        - If `ort_session` is None, this instance can still be used for barrier computations
          (e.g., `compute_h`, `_compute_alpha`), but inference-based methods will raise.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            ort_session: ONNXruntime inference session (optional; required for inference)
            input_names: Names of input tensors for ONNX model (required if ort_session is provided)
            output_names: Names of output tensors for ONNX model (required if ort_session is provided)
        """
        self.logger = logging.getLogger("MyRLApp.safety_filter")
        self.logger.info(f"SafetyFilter initialized with state_dim={state_dim}, action_dim={action_dim}")
        
        if ort_session is not None:
            if input_names is None or output_names is None:
                raise ValueError("input_names and output_names must be provided when ort_session is provided")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ort_session = ort_session
        self.input_names = list(input_names) if input_names is not None else []
        self.output_names = list(output_names) if output_names is not None else []

        # define constants for barrier function (using numpy)
        self.a = np.zeros(state_dim, dtype=np.float32)
        self.a[-1] = 1.0
        self.eps = 1e-8  # to avoid division by zero

    def _ort_session_run(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference using ONNXruntime session.
        
        Args:
            x: State array of shape (1, state_dim) or (state_dim,)
            u: Action array of shape (1, action_dim) or (action_dim,)
            
        Returns:
            Tuple of (x_next, f_x, G_x) as numpy arrays
            - x_next: (state_dim,) - predicted next state
            - f_x: (state_dim,) - drift term
            - G_x: (state_dim, action_dim) - control matrix
        """
        if self.ort_session is None:
            raise RuntimeError(
                "SafetyFilter was constructed without an ONNX Runtime session; "
                "inference-based methods are unavailable."
            )
        if len(self.input_names) < 2 or len(self.output_names) < 1:
            raise RuntimeError(
                "SafetyFilter is missing ONNX input/output tensor names; "
                "cannot run inference."
            )

        # Ensure inputs are 2D with batch dimension
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        if u.ndim == 1:
            u = np.expand_dims(u, 0)
        
        # Ensure float32 dtype for ONNX Runtime
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        
        # Run inference
        outputs = self.ort_session.run(
            self.output_names,
            {self.input_names[0]: x, 
             self.input_names[1]: u}
        )
        
        # ONNX model outputs: [x_next, f_x, G_x_flat]
        # G_x_flat needs to be reshaped to (1, state_dim, action_dim)
        x_next = outputs[0]  # (1, state_dim)
        f_x = outputs[1]     # (1, state_dim)
        G_x_flat = outputs[2]  # (1, state_dim * action_dim)
        G_x = G_x_flat.reshape(1, self.state_dim, self.action_dim)
        
        # Remove batch dimension for single-sample case
        x_next = x_next[0]  # (state_dim,)
        f_x = f_x[0]        # (state_dim,)
        G_x = G_x[0]        # (state_dim, action_dim)
        
        return x_next, f_x, G_x

    def compute_h(self, x: np.ndarray) -> float:
        """
        Compute barrier function h(x).
        
        Args:
            x: State array of shape (state_dim,)
            
        Returns:
            Scalar value of barrier function
        """
        return float(10.0 - np.dot(x, self.a))
    
    def _compute_alpha(self, x: np.ndarray) -> float:
        """
        Compute alpha value for safety filter.
        Can change this to make the filter more or less aggressive.
        Requirement is that 0 <= alpha <= h(x)
        
        Args:
            x: State array of shape (state_dim,)
            
        Returns:
            Scalar alpha value
        """
        return self.compute_h(x) * 0.5
    
    def _compute_rho(self, delta: float) -> float:
        """
        Compute rho value based on model error.
        Can be any function, chose log so that it plateaus at some point.
        delta should come from the training error, something like the MSE during training.
        
        Args:
            delta: Model error (scalar, typically MSE loss)
            
        Returns:
            Scalar rho value
        """
        return float(np.log1p(max(delta, 0.0)))

    def compute_filtered_action(self,
                                x: np.ndarray,
                                kn: np.ndarray,
                                model_error: float) -> np.ndarray:
        """
        Compute filtered action using safety filter.
        
        This function currently only supports single batch size. Expected input shapes are:
        - x is of shape (state_dim,) or (1, state_dim)
        - kn is of shape (action_dim,) or (1, action_dim)
        - model_error is a scalar computed from the training error (MSE loss).
        
        Args:
            x: Current state array
            kn: Nominal action array
            model_error: Model prediction error (scalar float)
            
        Returns:
            Filtered action of shape (action_dim,)
        """
        self.logger.debug(f"compute_filtered_action called with x={x}, kn={kn}, model_error={model_error}")
        # Ensure inputs are 1D arrays
        if x.ndim == 2:
            x = x[0]
        if kn.ndim == 2:
            kn = kn[0]
        
        # Get kn data type to match output
        kn_dtype = kn.dtype
        
        # Ensure float32 dtype
        x = x.astype(np.float32)
        kn = kn.astype(np.float32)
        
        # Predict state and get f_x, G_x using ORT
        _, f_x, G_x = self._ort_session_run(x, kn)
        # Outputs are already (state_dim,) and (state_dim, action_dim) after _ort_session_run
        
        # Compute Lie derivative terms
        lie_G = G_x.T @ self.a  # (action_dim,)
        denom = float(np.dot(lie_G, lie_G))  # equivalent to np.linalg.norm(lie_G, ord=2)**2

        # In the case that the action does not affect the barrier function, return the original action
        # This implementation is more robust than only if it equals zero.
        if denom < self.eps:
            return kn.astype(kn_dtype)

        # Compute remaining terms
        lie_F = float(np.dot(self.a, f_x - x))  # scalar
        alpha = self._compute_alpha(x)  # scalar
        rho = self._compute_rho(model_error)  # scalar

        
        # Compute correction term
        correction_term = -(lie_F + np.dot(lie_G, kn) + alpha - rho) / (denom + self.eps)  # scalar
        
        # Apply correction and return filtered action
        correction = max(correction_term, 0.0) * lie_G  # (action_dim,)

        # print out intermediate values for debugging
        self.logger.debug(f"compute_filtered_action: lie_G={lie_G}, denom={denom}, lie_F={lie_F}, alpha={alpha}, rho={rho}")
        self.logger.debug(f"compute_filtered_action: correction_term={correction_term}, correction={correction}")

        return (kn + correction).astype(kn_dtype)  # (action_dim,)


class FilterStorageBuffer:
    """
    Dual FIFO ring buffers (Recent + Critical) for accumulating filter training data.
    
    - Recent: Larger buffer storing normal recent rollouts.
    - Critical: Smaller buffer storing rollouts near the SafetyFilter boundary and/or with
      large intervention magnitude.
    
    Expected sample layout (flat float32) for Critical classification:
      (state, action_filtered, next_state, action_nominal)
    where state_dim == next_state_dim and action_dim == action_nominal_dim.
    """
    
    class _RingBuffer:
        """Simple pre-allocated FIFO ring buffer of flat float32 samples."""
        def __init__(self, max_samples: int, sample_dim: int):
            self.max_samples = int(max_samples)
            self.sample_dim = int(sample_dim)
            self.current_size = 0
            self.write_index = 0
            self.buffer = np.zeros((self.max_samples, self.sample_dim), dtype=np.float32)

        def add(self, sample: np.ndarray) -> None:
            self.buffer[self.write_index] = sample
            self.write_index = (self.write_index + 1) % self.max_samples
            if self.current_size < self.max_samples:
                self.current_size += 1

        def sample(self, n: int) -> np.ndarray:
            if n <= 0:
                return np.zeros((0, self.sample_dim), dtype=np.float32)
            if self.current_size < n:
                raise ValueError(f"Not enough samples to draw {n} (have {self.current_size})")
            idx = random.sample(range(self.current_size), n)
            return self.buffer[idx].copy()

        def clear(self) -> None:
            self.current_size = 0
            self.write_index = 0
            self.buffer.fill(0)

    def __init__(
        self,
        max_samples: int = 50000,
        min_samples_for_training: int = 1000,
        sample_dim: Optional[int] = None,
        *,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        critical_fraction: float = 0.20,
        critical_capacity_fraction: float = 0.20,
        h_critical_threshold: float = 2.0,
        intervention_l2_threshold: float = 0.10,
        compute_h_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        """
        Initialize the storage buffer with pre-allocated memory.
        
        Args:
            max_samples: Maximum number of samples to store in the Recent buffer (default: 50,000)
            min_samples_for_training: Minimum samples before training starts (default: 1,000)
            sample_dim: Dimension of each sample (flat vector). If None, inferred from first batch added.
                       If None, will be inferred from first batch added.
            state_dim: State dimension (required for Critical classification).
            action_dim: Action dimension (required for Critical classification).
            critical_fraction: Fraction of each sampled batch to draw from the Critical buffer (default: 0.20).
            critical_capacity_fraction: Critical buffer capacity as a fraction of Recent capacity (default: 0.20).
            h_critical_threshold: Route to Critical if h(x) <= threshold (default: 2.0).
            intervention_l2_threshold: Route to Critical if ||u_f - u_n||_2 >= threshold (default: 0.10).
            compute_h_fn: Callable used to compute h(x) from a state vector x.
                          Required when `state_dim` and `action_dim` are provided.
        """
        self.max_samples = int(max_samples)
        self.min_samples_for_training = min_samples_for_training
        self.sample_dim = sample_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critical_fraction = float(critical_fraction)
        self.critical_capacity_fraction = float(critical_capacity_fraction)
        self.h_critical_threshold = float(h_critical_threshold)
        self.intervention_l2_threshold = float(intervention_l2_threshold)
        self.compute_h_fn = compute_h_fn

        # Critical classification requires state/action dims and a barrier function h(x).
        if (self.state_dim is None) != (self.action_dim is None):
            raise ValueError("FilterStorageBuffer requires both state_dim and action_dim (or neither)")
        if self.state_dim is not None and self.action_dim is not None:
            if self.compute_h_fn is None:
                raise ValueError(
                    "FilterStorageBuffer requires compute_h_fn when state_dim/action_dim are provided "
                    "(needed to compute h(x) for Critical classification)"
                )

        # Buffers are allocated once sample_dim is known.
        self.recent_buffer: Optional[FilterStorageBuffer._RingBuffer] = None
        self.critical_buffer: Optional[FilterStorageBuffer._RingBuffer] = None
        self._initialized = False
        if sample_dim is not None:
            self._initialize_buffers(sample_dim)

    def _initialize_buffers(self, sample_dim: int) -> None:
        self.sample_dim = int(sample_dim)
        recent_max = int(self.max_samples)
        critical_max = max(1, int(round(recent_max * self.critical_capacity_fraction)))
        self.recent_buffer = FilterStorageBuffer._RingBuffer(recent_max, self.sample_dim)
        self.critical_buffer = FilterStorageBuffer._RingBuffer(critical_max, self.sample_dim)
        self._initialized = True

    def _classify_sample(self, sample: np.ndarray) -> bool:
        """Return True if sample should go to Critical, else Recent."""
        if self.state_dim is None or self.action_dim is None:
            return False
        if self.compute_h_fn is None:
            raise RuntimeError(
                "FilterStorageBuffer is configured for Critical classification "
                "(state_dim/action_dim set) but compute_h_fn is missing"
            )

        sdim = int(self.state_dim)
        adim = int(self.action_dim)
        if sample.shape[0] != self.sample_dim:
            raise ValueError(f"Sample dim {sample.shape[0]} doesn't match buffer dim {self.sample_dim}")

        # Expected layout: [state (sdim), action_filtered (adim), next_state (sdim), action_nominal (adim)]
        u_f_start = sdim
        u_f_end = u_f_start + adim
        u_n_start = u_f_end + sdim
        u_n_end = u_n_start + adim
        if u_n_end != self.sample_dim:
            raise ValueError(
                f"Sample layout mismatch: expected dim {u_n_end} from state_dim={sdim}, action_dim={adim}, got {self.sample_dim}"
            )

        x = sample[:sdim]
        u_f = sample[u_f_start:u_f_end]
        u_n = sample[u_n_start:u_n_end]

        h = float(self.compute_h_fn(x))
        delta = float(np.linalg.norm(u_f - u_n, ord=2))
        return (h <= self.h_critical_threshold) or (delta >= self.intervention_l2_threshold)
    
    def add_batch(self, batch: np.ndarray) -> None:
        """
        Add a batch from ring buffer to storage.
        Splits batch into individual rollouts and stores them in the ring buffer.
        
        Uses FIFO retention: when full, overwrites oldest samples.
        
        Args:
            batch: NumPy array of shape (batch_size, sample_dim)
        """
        batch_size = batch.shape[0]
        sample_dim = batch.shape[1]
        
        # Initialize buffer on first call if sample_dim wasn't provided
        if not self._initialized:
            self._initialize_buffers(sample_dim)
        
        # Verify sample dimension matches
        if sample_dim != self.sample_dim:
            raise ValueError(f"Batch sample dimension {sample_dim} doesn't match buffer dimension {self.sample_dim}")
        
        if self.recent_buffer is None or self.critical_buffer is None:
            raise RuntimeError("FilterStorageBuffer buffers are not initialized")

        # Add each rollout from the batch individually, routing to Recent/Critical
        for i in range(batch_size):
            sample = batch[i]
            if self._classify_sample(sample):
                self.critical_buffer.add(sample)
            else:
                self.recent_buffer.add(sample)
    
    def sample_batches(self, n_batches: int, batch_size: int) -> Optional[List[np.ndarray]]:
        """
        Sample random mixed batches from storage buffers for training.
        
        Args:
            n_batches: Number of batches to sample
            batch_size: Size of each batch to return
            
        Returns:
            List of batches, each of shape (batch_size, sample_dim),
            or None if not enough samples available
        """
        if not self._initialized or self.recent_buffer is None or self.critical_buffer is None:
            return None

        total = self.size()
        if total < self.min_samples_for_training:
            return None

        if total < batch_size:
            return None

        crit_target = int(round(batch_size * self.critical_fraction))
        crit_target = max(0, min(batch_size, crit_target))

        sampled_batches: List[np.ndarray] = []
        for _ in range(n_batches):
            crit_size = self.critical_buffer.current_size
            recent_size = self.recent_buffer.current_size

            n_crit = min(crit_target, crit_size)
            n_recent = min(batch_size - n_crit, recent_size)

            # If Recent couldn't fill the remainder, backfill from Critical.
            if n_crit + n_recent < batch_size:
                needed = batch_size - (n_crit + n_recent)
                n_crit = min(crit_size, n_crit + needed)

            # If still short, we can't produce a full batch.
            if n_crit + n_recent < batch_size:
                return None

            crit_samples = self.critical_buffer.sample(n_crit)
            recent_samples = self.recent_buffer.sample(n_recent)
            mixed = np.concatenate([crit_samples, recent_samples], axis=0)
            # Shuffle to avoid ordering bias (crit first).
            if mixed.shape[0] > 1:
                mixed = mixed[np.random.permutation(mixed.shape[0])]
            sampled_batches.append(mixed)

        return sampled_batches
    
    def size(self) -> int:
        """
        Return current number of samples in both buffers (Recent + Critical).
        
        Returns:
            Number of samples currently stored
        """
        if not self._initialized or self.recent_buffer is None or self.critical_buffer is None:
            return 0
        return int(self.recent_buffer.current_size + self.critical_buffer.current_size)

    def recent_size(self) -> int:
        if not self._initialized or self.recent_buffer is None:
            return 0
        return int(self.recent_buffer.current_size)

    def critical_size(self) -> int:
        if not self._initialized or self.critical_buffer is None:
            return 0
        return int(self.critical_buffer.current_size)
    
    def clear(self) -> None:
        """Clear all stored data (reset to empty state)."""
        if self.recent_buffer is not None:
            self.recent_buffer.clear()
        if self.critical_buffer is not None:
            self.critical_buffer.clear()


if __name__ == "__main__":
    # Example usage (requires actual ORT session)
    state_dim = 5
    action_dim = 3
    num_hidden = 2
    hidden_exp = 7
    dropout = 0.0
    
    # Test StatePredictor (PyTorch model for training)
    state_predictor = StatePredictor(state_dim, action_dim, num_hidden, hidden_exp, dropout)
    x = torch.randn(1, state_dim, requires_grad=True)
    u = torch.randn(1, action_dim, requires_grad=True)
    output = state_predictor(x, u)
    print(f"StatePredictor output shapes: {[o.shape for o in output]}")

    loss = output[0].sum()
    loss.backward()

    # Check if gradients exist
    print(f"x.grad is not None: {x.grad is not None}")
    print(f"Gradient norm: {x.grad.norm().item()}")