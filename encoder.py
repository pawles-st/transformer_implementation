import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    """Single encoder layer composed of Multi-Head Attention and Feed-Forward Network."""

    def __init__(self, d_e: int, d_k: int, d_v: int, d_ff: int, h: int) -> None:
        """Initialise the encoder layer.

        Args:
            d_e: Embedding dimension.
            d_k: Query/key vector dimension.
            d_v: Value vector dimension.
            d_ff: Inner hidden dimension of feed-forward network.
            h: Number of attention heads.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Compute multi-head attention and project: (seq_len, h * d_v) -> (seq_len, d_e).
        """Pass input through the encoder layer.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError

class Encoder(nn.Module):  # Stack of N identical EncoderLayers.
    """Encoder module in a transformer architecture composed of EncoderLayers."""

    def __init__(self, num_layers: int, d_e: int, d_k: int, d_v: int, d_ff: int, h: int) -> None:
        """Initialise the encoder.

        Args:
            num_layers: Number of identical encoder layers.
            d_e: Embedding dimension.
            d_k: Query/key vector dimension.
            d_v: Value vector dimension.
            d_ff: Inner hidden dimension of feed-forward network.
            h: Number of attention heads.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through each of the encoder layers.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError
