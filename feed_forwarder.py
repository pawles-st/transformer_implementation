import torch
import torch.nn as nn

class FeedForwarder(nn.Module):
    """Position-wise feed-forward network applied to each token representation."""

    def __init__(self, d_e: int, d_ff: int) -> None:
        """Initialise the feed-forward network.

        Args:
            d_e: Embedding dimension (input and output vector size).
            d_ff: Inner hidden dimension of the feed-forward layer.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation: Linear -> ReLU -> Linear.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError
