import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List
import random

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
                 ort_session: ort.InferenceSession,
                 input_names: List[str],
                 output_names: List[str]) -> None:
        """
        Initialize SafetyFilter with ONNX Runtime session.
        
        This class only supports ORT models for inference. PyTorch models are handled
        separately in custom_run.py for training, then converted to ORT format.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            ort_session: ONNXruntime inference session (required)
            input_names: Names of input tensors for ONNX model (required)
            output_names: Names of output tensors for ONNX model (required)
        """
        if ort_session is None:
            raise ValueError("ort_session must be provided")
        if input_names is None or output_names is None:
            raise ValueError("input_names and output_names must be provided")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ort_session = ort_session
        self.input_names = input_names
        self.output_names = output_names

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

    def _compute_h(self, x: np.ndarray) -> float:
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
        return self._compute_h(x) * 0.5
    
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
        # Ensure inputs are 1D arrays
        if x.ndim == 2:
            x = x[0]
        if kn.ndim == 2:
            kn = kn[0]
        
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
            return kn

        # Compute remaining terms
        lie_F = float(np.dot(self.a, f_x - x))  # scalar
        alpha = self._compute_alpha(x)  # scalar
        rho = self._compute_rho(model_error)  # scalar
        
        # Compute correction term
        correction_term = -(lie_F + np.dot(lie_G, kn) + alpha - rho) / (denom + self.eps)  # scalar
        
        # Apply correction and return filtered action
        correction = max(correction_term, 0.0) * lie_G  # (action_dim,)
        return kn + correction  # (action_dim,)


class FilterStorageBuffer:
    """
    FIFO ring buffer for accumulating filter training data with pre-allocated memory.
    
    This buffer accumulates individual rollouts (state, action, next_state) in a 
    pre-allocated ring buffer structure. When full, it uses FIFO retention (overwrites oldest).
    """
    
    def __init__(self, max_samples: int = 50000, min_samples_for_training: int = 1000, sample_dim: Optional[int] = None):
        """
        Initialize the storage buffer with pre-allocated memory.
        
        Args:
            max_samples: Maximum number of samples to store (default: 50,000)
            min_samples_for_training: Minimum samples before training starts (default: 1,000)
            sample_dim: Dimension of each sample (state_dim + action_dim + next_state_dim).
                       If None, will be inferred from first batch added.
        """
        self.max_samples = max_samples
        self.min_samples_for_training = min_samples_for_training
        self.sample_dim = sample_dim
        self.current_size = 0  # Number of samples currently in buffer
        self.write_index = 0  # Ring buffer write position
        
        # Pre-allocate buffer if sample_dim is known, otherwise allocate on first add
        if sample_dim is not None:
            self.buffer = np.zeros((max_samples, sample_dim), dtype=np.float32)
            self._initialized = True
        else:
            self.buffer = None
            self._initialized = False
    
    def add_batch(self, batch: np.ndarray) -> None:
        """
        Add a batch from ring buffer to storage.
        Splits batch into individual rollouts and stores them in the ring buffer.
        
        Uses FIFO retention: when full, overwrites oldest samples.
        
        Args:
            batch: NumPy array of shape (batch_size, state_dim + action_dim + next_state_dim)
        """
        batch_size = batch.shape[0]
        sample_dim = batch.shape[1]
        
        # Initialize buffer on first call if sample_dim wasn't provided
        if not self._initialized:
            if self.sample_dim is None:
                self.sample_dim = sample_dim
            self.buffer = np.zeros((self.max_samples, self.sample_dim), dtype=np.float32)
            self._initialized = True
        
        # Verify sample dimension matches
        if sample_dim != self.sample_dim:
            raise ValueError(f"Batch sample dimension {sample_dim} doesn't match buffer dimension {self.sample_dim}")
        
        # Add each rollout from the batch individually
        for i in range(batch_size):
            # Write to current position in ring buffer
            self.buffer[self.write_index] = batch[i]
            
            # Advance write index (ring buffer wraps around)
            self.write_index = (self.write_index + 1) % self.max_samples
            
            # Update current size (stops growing once buffer is full)
            if self.current_size < self.max_samples:
                self.current_size += 1
    
    def sample_batches(self, n_batches: int, batch_size: int) -> Optional[List[np.ndarray]]:
        """
        Sample random batches from storage buffer for training.
        
        Args:
            n_batches: Number of batches to sample
            batch_size: Size of each batch to return
            
        Returns:
            List of batches, each of shape (batch_size, sample_dim),
            or None if not enough samples available
        """
        if self.current_size < self.min_samples_for_training:
            return None
        
        if self.current_size < batch_size:
            return None
        
        if not self._initialized or self.buffer is None:
            return None
        
        sampled_indices = random.sample(range(self.current_size), n_batches * batch_size)
        samples = self.buffer[sampled_indices].copy()
        sampled_batches = []
        for i in range(n_batches):
            batch = samples[i * batch_size:(i + 1) * batch_size]
            sampled_batches.append(batch)

        return sampled_batches
    
    def size(self) -> int:
        """
        Return current number of samples in the buffer.
        
        Returns:
            Number of samples currently stored
        """
        return self.current_size
    
    def clear(self) -> None:
        """Clear all stored data (reset to empty state)."""
        self.current_size = 0
        self.write_index = 0
        if self.buffer is not None:
            self.buffer.fill(0)


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