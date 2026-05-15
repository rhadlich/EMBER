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


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_exp: int,
        dropout: float,
    ):
        super().__init__()
        hidden_dim = int((2 ** hidden_exp) / (1 - dropout))
        self.activation = nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )   

    def forward(self, x):
        return self.activation(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int,
        hidden_exp: int,
        dropout: float,
    ):
        super().__init__()

        hidden_dim = int((2 ** hidden_exp) / (1 - dropout))

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_exp, dropout) for _ in range(num_blocks)],
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class MSEWithDp(nn.Module):
    def __init__(self, alpha: float = 1, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Reduction {reduction} is not supported, must be 'mean' or 'sum'")
        self.reduction = reduction
        self.last_terms = None

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (target - output) ** 2

        res = 720 / output.size(1)
        start_ind = int((360 - 90) / res)
        end_ind = int((360 + 130) / res)
        dp_hat = output[:, start_ind + 2 : end_ind] - output[:, start_ind : end_ind - 2]
        dp = target[:, start_ind + 2 : end_ind] - target[:, start_ind : end_ind - 2]
        diff_dp = (dp_hat - dp) ** 2

        if self.reduction == "mean":
            mse_term = diff.mean()
            dp_term = self.alpha * diff_dp.mean()
            total = mse_term + dp_term
        else:
            mse_term = diff.sum()
            dp_term = self.alpha * diff_dp.sum()
            total = mse_term + dp_term

        self.last_terms = {
            "mse": mse_term.detach(),
            "dp": dp_term.detach(),
            "total": total.detach(),
        }
        return total
