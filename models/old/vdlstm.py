import torch
from torch import nn

# ================================================================================
# ABLATION STUDY: PA MODEL ARCHITECTURES (DGRU vs VDLSTM vs GMP)
# ================================================================================

# Cell: VDLSTM PA Model Definition
class PAModelVDLSTM(nn.Module):
    """
    Vectorial Dilated LSTM (VDLSTM) for PA surrogate modeling.

    Uses dilated LSTM for larger receptive field without depth.
    Better for capturing long-range memory effects.

    Architecture:
        Input [B, seq_len, 2] → Dilated LSTM [hidden=64, 2 layers, dilation=2] → FC [2]
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dilation=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dilation = dilation

        # Stack of dilated LSTMs
        self.lstm_cells = nn.ModuleList()
        self.dilations = []

        for i in range(num_layers):
            current_dilation = dilation ** i
            self.dilations.append(current_dilation)

            if i == 0:
                in_size = input_size
            else:
                in_size = hidden_size

            self.lstm_cells.append(
                nn.LSTMCell(in_size, hidden_size)
            )

        # Output projection
        self.fc_out = nn.Linear(hidden_size, 2)

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass with dilated LSTM

        Args:
            x: [B, seq_len, 2] or [B, 2] IQ input

        Returns:
            y: [B, 2] IQ output
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 2]

        batch_size, seq_len, _ = x.size()
        device = x.device

        # Initialize hidden and cell states
        h = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]

        # Process sequence with dilated timesteps
        output = None
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, 2]

            for layer in range(self.num_layers):
                # Get dilated timestep
                dilation = self.dilations[layer]
                t_dilated = max(0, t - dilation)
                x_dilated = x[:, t_dilated, :] if layer == 0 else h[layer-1]

                # LSTM cell forward
                h[layer], c[layer] = self.lstm_cells[layer](x_dilated, (h[layer], c[layer]))

        # Project output from last layer
        output = self.fc_out(h[-1])  # [B, 2]
        return output