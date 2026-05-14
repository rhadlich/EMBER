import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{device_name}'. Use one of: auto, cpu, mps, cuda.")


def _to_device(batch: Any, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    return batch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric_fns: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        *,
        val_fn: Optional[Callable] = None,
        train_method: str = "default",
        validate_each_epoch: bool = True,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric_fns = metric_fns or {}
        self.val_fn = val_fn
        self.train_method = train_method
        self.validate_each_epoch = validate_each_epoch
        self.distributed = distributed
        self.global_rank = rank
        self.world_size = world_size
        self.device = device or resolve_device("auto")

        self.epochs_run = 0
        self.loss = float("inf")
        self.avg_loss = float("inf")
        self.val_loss = torch.tensor(float("inf"), device=self.device)
        self.train_metrics: Dict[str, torch.Tensor] = {}
        self.val_metrics: Dict[str, torch.Tensor] = {}
        self.time_epoch = torch.tensor(0.0, device=self.device)
        self.history: Dict[str, list] = {"train": [], "val": []}

    @staticmethod
    def _to_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _record_train_history(self, epoch: int, steps: int, epoch_time: float) -> None:
        train_entry = {
            "epoch": int(epoch),
            "steps": int(steps),
            "time_s": float(epoch_time),
            "loss": self._to_float(self.avg_loss),
        }
        for metric_name, metric_value in self.train_metrics.items():
            train_entry[metric_name] = self._to_float(metric_value)
        for term_name, term_sum in self.extra_term_loggers.items():
            train_entry[term_name] = self._to_float(term_sum / steps)
        self.history["train"].append(train_entry)

    def _record_val_history(self, epoch: int) -> None:
        val_entry = {
            "epoch": int(epoch),
            "loss": self._to_float(self.val_loss),
        }
        for metric_name, metric_value in self.val_metrics.items():
            val_entry[metric_name] = self._to_float(metric_value)
        self.history["val"].append(val_entry)

    def _split_batch(self, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "Trainer expects each dataloader batch to be a (source, targets) pair."
            )
        return batch[0], batch[1]

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        self.loss = loss
        self.loss_logger += loss.detach()
        for metric_name, metric_fn in self.metric_fns.items():
            self.metric_loggers[metric_name] += metric_fn(output, targets).detach()

        if hasattr(self.criterion, "last_terms") and self.criterion.last_terms is not None:
            for term_name, term_value in self.criterion.last_terms.items():
                if term_name == "total":
                    continue
                term_tensor = term_value.detach() if torch.is_tensor(term_value) else torch.tensor(
                    float(term_value), device=self.device
                )
                self.extra_term_loggers[term_name] = self.extra_term_loggers.get(
                    term_name, torch.tensor(0.0, device=self.device)
                ) + term_tensor

        loss.backward()
        self.optimizer.step()

    def _run_val(self, loader):
        self.model.eval()
        step_fn = self.val_fn or self.model
        val_logger = torch.tensor(0.0, device=self.device)
        val_metric_loggers = {
            name: torch.tensor(0.0, device=self.device) for name in self.metric_fns.keys()
        }

        val_steps = 0
        with torch.no_grad():
            for batch in loader:
                source, targets = self._split_batch(batch)
                source = _to_device(source, self.device)
                targets = _to_device(targets, self.device)
                output = step_fn(source)
                loss = self.criterion(output, targets)
                val_logger += loss.detach()
                for metric_name, metric_fn in self.metric_fns.items():
                    val_metric_loggers[metric_name] += metric_fn(output, targets).detach()
                val_steps += 1

        if val_steps == 0:
            raise ValueError("Validation loader produced zero batches.")

        self.val_loss = val_logger / val_steps
        self.val_metrics = {
            metric_name: metric_value / val_steps
            for metric_name, metric_value in val_metric_loggers.items()
        }

        if self.distributed and dist.is_initialized():
            dist.all_reduce(self.val_loss)
            self.val_loss /= self.world_size
            for metric_name in list(self.val_metrics.keys()):
                dist.all_reduce(self.val_metrics[metric_name])
                self.val_metrics[metric_name] /= self.world_size

        self.model.train()

    def _run_epoch(self, epoch, *, tf_decay_exp=0.95):
        if self.distributed and hasattr(self.train_data.sampler, "set_epoch"):
            self.train_data.sampler.set_epoch(epoch)

        self.loss_logger = torch.tensor(0.0, device=self.device)
        self.metric_loggers = {
            metric_name: torch.tensor(0.0, device=self.device)
            for metric_name in self.metric_fns.keys()
        }
        self.extra_term_loggers: Dict[str, torch.Tensor] = {}

        steps = 0
        if self.train_method == "default":
            for batch in self.train_data:
                source, targets = self._split_batch(batch)
                source = _to_device(source, self.device)
                targets = _to_device(targets, self.device)
                self._run_batch(source, targets)
                steps += 1
        elif self.train_method == "teacher_forcing":
            if self.val_fn is None:
                raise ValueError("teacher_forcing requires val_fn to be set.")
            tf_ratio = tf_decay_exp**epoch
            for batch in self.train_data:
                source, targets = self._split_batch(batch)
                source = _to_device(source, self.device)
                targets = _to_device(targets, self.device)
                if torch.rand(1, device=self.device) < tf_ratio:
                    p_in = targets
                else:
                    p_in = self.val_fn(source)
                input_tuple = (source, p_in)
                self._run_batch(input_tuple, targets)
                steps += 1
        else:
            raise ValueError(f"Unsupported train_method: {self.train_method}")

        if steps == 0:
            raise ValueError("Training loader produced zero batches.")

        self.avg_loss = self.loss_logger / steps
        self.train_metrics = {
            metric_name: metric_value / steps
            for metric_name, metric_value in self.metric_loggers.items()
        }

        if self.global_rank == 0:
            train_parts = [f"[Epoch {epoch}", f"Loss: {self.avg_loss.item():.4f}"]
            for metric_name, metric_value in self.train_metrics.items():
                train_parts.append(f"{metric_name.upper()}: {metric_value.item():.6e}")
            for term_name, term_value in self.extra_term_loggers.items():
                train_parts.append(f"{term_name.upper()}: {(term_value / steps).item():.6e}")
            train_parts.append(f"Steps: {steps}]")
            print(" | ".join(train_parts))

        self.scheduler.step()
        return steps

    def train(self, max_epochs):
        time_epoch = 0.0
        self.history = {"train": [], "val": []}
        for epoch in range(self.epochs_run, max_epochs):
            tic_epoch = time.time()
            steps = self._run_epoch(epoch)
            if self.validate_each_epoch:
                self._run_val(self.val_data)
            toc_epoch = time.time()
            epoch_time = toc_epoch - tic_epoch
            time_epoch += epoch_time
            self._record_train_history(epoch, steps, epoch_time)
            if self.validate_each_epoch:
                self._record_val_history(epoch)
            if self.global_rank == 0 and self.validate_each_epoch:
                val_parts = [f"[Epoch {epoch}", f"ValLoss: {self.val_loss.item():.4f}"]
                for metric_name, metric_value in self.val_metrics.items():
                    val_parts.append(f"VAL_{metric_name.upper()}: {metric_value.item():.6e}")
                val_parts.append("]")
                print(" | ".join(val_parts))
            if self.global_rank == 0:
                print(f"Epoch time: {epoch_time}s")

        time_epoch /= max_epochs
        self.time_epoch = torch.tensor(time_epoch, device=self.device)
        if self.distributed and dist.is_initialized():
            dist.all_reduce(self.time_epoch)
            self.time_epoch /= self.world_size

        if self.global_rank == 0:
            print(f"The average time per epoch is {self.time_epoch}s")

        if not self.validate_each_epoch:
            self._run_val(self.val_data)
        if self.global_rank == 0:
            print(f"The validation loss is {self.val_loss}.")
