import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    """Single attention head computing scaled dot-product attention."""

    def __init__(self, d_e: int, d_k: int, d_v: int) -> None:
        """Initialise the attention head.

        Args:
            d_e: Embedding dimension (input vector length).
            d_k: Query and key dimension.
            d_v: Value dimension.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention for a single head.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_v).
        """
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism concatenating h attention heads and projecting back to original dimension."""

    def __init__(self, d_e: int, d_k: int, d_v: int, h: int) -> None:
        """Initialise multi-head attention.

        Args:
            d_e: Embedding dimension (input and output vector length).
            d_k: Query and key dimension per head.
            d_v: Value dimension per head.
            h: Number of attention heads.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head attention over input representations.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError
