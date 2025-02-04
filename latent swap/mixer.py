import torch
import torch.nn as nn
from einops import rearrange


class LatentMixer(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(LatentMixer, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, ls: torch.Tensor, lt: torch.Tensor):
        lst = torch.cat([ls, lt], dim=-1)
        return self.net(lst) + lt


class Mixer(nn.Module):
    def __init__(self, num_swap=18, input_dim=1024, hidden_dim=512):
        super(Mixer, self).__init__()

        self.swap_layers = nn.ModuleList([
            LatentMixer(input_dim=input_dim, hidden_dim=hidden_dim) for _ in range(num_swap)
        ])

    def forward(self, ls: torch.Tensor, lt: torch.Tensor):
        # Ensure proper shape handling for batch processing
        assert ls.size(1) == len(self.swap_layers), "Mismatch between latent dimensions and swap layers."
        assert lt.size(1) == len(self.swap_layers), "Mismatch between latent dimensions and swap layers."

        lsw = []
        for i, layer in enumerate(self.swap_layers):
            ls_i = ls[:, i, :].unsqueeze(1)  # Extracting specific slice
            lt_i = lt[:, i, :].unsqueeze(1)
            lsw.append(layer(ls_i, lt_i))

        return torch.cat(lsw, dim=1)
