from __future__ import annotations

import torch
from torch import nn


class SummaryResidualMDN(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 128, n_components: int = 5
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.n_components = int(n_components)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_dim, n_components)
        self.means = nn.Linear(hidden_dim, n_components * output_dim)
        self.log_scales = nn.Linear(hidden_dim, n_components * output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        logits = self.logits(h)
        means = self.means(h).reshape(-1, self.n_components, self.output_dim)
        log_scales = self.log_scales(h).reshape(-1, self.n_components, self.output_dim)
        log_scales = torch.clamp(log_scales, min=-6.0, max=3.0)
        return logits, means, log_scales

    def negative_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, means, log_scales = self.forward(x)
        y = y[:, None, :]
        inv_var = torch.exp(-2.0 * log_scales)
        log_prob_dim = -0.5 * ((y - means) ** 2 * inv_var) - log_scales
        log_prob_dim = log_prob_dim - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        log_prob = log_prob_dim.sum(dim=-1)
        mixture_log_prob = torch.log_softmax(logits, dim=-1) + log_prob
        return -torch.logsumexp(mixture_log_prob, dim=-1).mean()

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        logits, means, log_scales = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        draws = []
        for _ in range(n_samples):
            component = torch.multinomial(probs, num_samples=1).squeeze(-1)
            batch = torch.arange(x.shape[0], device=x.device)
            mean = means[batch, component]
            scale = torch.exp(log_scales[batch, component])
            draws.append(mean + scale * torch.randn_like(mean))
        return torch.stack(draws, dim=1)
