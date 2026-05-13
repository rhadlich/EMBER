import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden: int,
        hidden_exp: int,
        dropout: float,
    ):
        super().__init__()
        hidden_dim = int((2 ** hidden_exp) / (1 - dropout))
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLP_with_GRU_head(nn.Module):
    def __init__(
        self,
        input_dim,
        num_hidden_MLP,
        hidden_exp_MLP,
        hidden_dim_GRU,
        seq_len,
    ):
        super().__init__()
        hidden_dim = int(2 ** hidden_exp_MLP)
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_hidden_MLP):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, hidden_dim_GRU))
        self.enc = nn.Sequential(*layers)

        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim_GRU, batch_first=True)
        self.to_p = nn.Linear(hidden_dim_GRU, 1)
        self.seq_len = seq_len

    def forward(self, input_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        source, p_in = input_tuple
        h0 = torch.tanh(self.enc(source))
        h0 = h0.unsqueeze(0)

        y, _ = self.gru(p_in.unsqueeze(-1), h0)
        p_hat = self.to_p(y).squeeze(-1)
        return p_hat

    @torch.no_grad()
    def inference_no_grad(self, x_scalar: torch.Tensor):
        h0 = torch.tanh(self.enc(x_scalar))
        h = h0.unsqueeze(0)
        p_prev = torch.zeros(x_scalar.size(0), 1, device=x_scalar.device)
        outputs = []
        for _ in range(self.seq_len):
            y, h = self.gru(p_prev.unsqueeze(-1), h)
            p_prev = self.to_p(y).squeeze(-1)
            outputs.append(p_prev)
        p_hat = torch.cat(outputs, dim=-1)
        return p_hat


class MSEWithDp(nn.Module):
    def __init__(self, alpha: float = 1, beta: float = 0.01, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Reduction {reduction} is not supported, must be 'mean' or 'sum'")
        self.reduction = reduction

        self.bore = 79.0 / 1000.0
        self.stroke = 86.0 / 1000.0
        self.rod_len = 160.0 / 1000.0
        self.delta = 0.6 / 1000.0
        self.crank_radius = self.stroke / 2.0
        self.compression_ratio = 17.19
        self.vd = np.pi * ((self.bore / 2.0) ** 2) * (2.0 * self.crank_radius)
        self.vc = self.vd / (self.compression_ratio - 1.0)
        self.cad_step_deg = 0.1
        cad = torch.arange(-360.0, 360.0, self.cad_step_deg, dtype=torch.float32)
        cad_rad = torch.deg2rad(cad)
        bore = torch.tensor(self.bore, dtype=torch.float32)
        rod_len = torch.tensor(self.rod_len, dtype=torch.float32)
        crank_radius = torch.tensor(self.crank_radius, dtype=torch.float32)
        delta = torch.tensor(self.delta, dtype=torch.float32)
        vc = torch.tensor(self.vc, dtype=torch.float32)
        area = torch.pi * (bore**2) / 4.0
        volume = vc + area * (
            rod_len
            + crank_radius
            - (
                crank_radius * torch.cos(cad_rad)
                + torch.sqrt(
                    rod_len**2 - (crank_radius * torch.sin(cad_rad) + crank_radius * delta) ** 2
                )
            )
        )
        self.register_buffer("volume_trace", volume)
        self.last_terms = None

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (target - output) ** 2
        expected_len = self.volume_trace.numel()
        if output.size(1) != expected_len or target.size(1) != expected_len:
            raise ValueError(
                f"Expected pressure trace length {expected_len} to match volume trace, "
                f"got output={output.size(1)}, target={target.size(1)}."
            )

        res = 720 / output.size(1)
        start_ind = int((360 - 90) / res)
        end_ind = int((360 + 130) / res)
        dp_hat = output[:, start_ind + 2 : end_ind] - output[:, start_ind : end_ind - 2]
        dp = target[:, start_ind + 2 : end_ind] - target[:, start_ind : end_ind - 2]
        diff_dp = (dp_hat - dp) ** 2

        volume = self.volume_trace.to(device=output.device, dtype=output.dtype)
        w_target = torch.trapz(target * 100000, x=volume, dim=1)
        w_output = torch.trapz(output * 100000, x=volume, dim=1)
        w_diff = (w_target - w_output) ** 2

        if self.reduction == "mean":
            mse_term = diff.mean()
            dp_term = self.alpha * diff_dp.mean()
            work_term = self.beta * w_diff.mean()
            total = mse_term + dp_term + work_term
        else:
            mse_term = diff.sum()
            dp_term = self.alpha * diff_dp.sum()
            work_term = self.beta * w_diff.sum()
            total = mse_term + dp_term + work_term

        self.last_terms = {
            "mse": mse_term.detach(),
            "dp": dp_term.detach(),
            "work": work_term.detach(),
            "total": total.detach(),
        }
        return total
