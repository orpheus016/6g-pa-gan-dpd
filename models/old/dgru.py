# ================================================================================
# PA SURROGATE MODEL: DGRU-based (Forward Learning Architecture)
# ================================================================================

import torch
from torch import nn

# Cell: DGRU PA Model Definition
class PAModelDGRU(nn.Module):
    """
    DGRU-based PA surrogate model (matches OpenDPD architecture).

    Learns to predict: u_PA (clean input) → y_PA (distorted output)
    Uses GRU to capture temporal memory effects in PA.

    Architecture:
        Input [B, seq_len, 2] → GRU [hidden=64, 2 layers] → FC [2]
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_size, 2)

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass: u_PA → y_PA

        Args:
            x: [B, seq_len, 2] or [B, 2] IQ input

        Returns:
            y: [B, 2] IQ output (last timestep)
        """
        # If input is [B, 2], expand to [B, 1, 2]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 2]

        # GRU forward pass
        out, _ = self.gru(x)  # out: [B, seq_len, hidden_size]

        # Take last timestep and project to output
        out = self.fc_out(out[:, -1, :])  # [B, 2]

        return out