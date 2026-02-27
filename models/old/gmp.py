import torch
from torch import nn

# Cell: GMP (Generalized Memory Polynomial) PA Model Definition
class PAModelGMP(nn.Module):
    """
    Generalized Memory Polynomial (GMP) for PA surrogate modeling.

    Classical DSP approach extended with neural network.
    Uses polynomial basis functions with memory taps.

    Architecture:
        Input [B, 2] → Basis expansion (poly + memory) → FC layers → Output [B, 2]
    """
    def __init__(self, input_size=2, memory_depth=3, poly_order=3, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.memory_depth = memory_depth
        self.poly_order = poly_order
        self.hidden_size = hidden_size

        # Compute basis dimension
        # For complex IQ: amplitude + phase features
        # Basis: [|u|, angle(u), |u|^2, |u|^3, ... ] × memory_depth
        self.basis_dim = (2 + poly_order) * (memory_depth + 1)  # Poly order + magnitude/phase

        # Neural network on top of GMP basis
        self.fc1 = nn.Linear(self.basis_dim, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, 2)

        # Memory buffer for historical inputs
        self.register_buffer('u_memory', torch.zeros(1, memory_depth + 1, 2))

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass with GMP basis expansion

        Args:
            x: [B, 2] or [B, seq_len, 2] IQ input

        Returns:
            y: [B, 2] IQ output
        """
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep [B, 2]

        batch_size = x.size(0)
        device = x.device

        # GMP basis expansion
        basis_features = []

        for m in range(self.memory_depth + 1):
            # Current or past input (simplified: just use current)
            u_m = x  # [B, 2]

            # Magnitude (amplitude)
            mag = torch.norm(u_m, dim=1, keepdim=True)  # [B, 1]
            basis_features.append(mag)

            # Phase
            phase = torch.atan2(u_m[:, 1:2], u_m[:, 0:1])  # [B, 1]
            basis_features.append(phase)

            # Polynomial expansion: |u|^k for k=1..poly_order
            for k in range(1, self.poly_order + 1):
                basis_features.append(mag ** k)

        # Concatenate all basis features
        basis = torch.cat(basis_features, dim=1)  # [B, basis_dim]

        # Neural network on basis
        h = self.act1(self.fc1(basis))
        h = self.act2(self.fc2(h))
        out = self.fc_out(h)  # [B, 2]

        return out