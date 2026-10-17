import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """Positional encoding layer using sinusoidal frequencies."""
    
    def __init__(self, d_e: int) -> None:
        """Initialise the positional encoder.

        Args:
            d_e: Embedding dimension.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encodings to embedded token representations.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError

