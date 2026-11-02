import torch
import torch.nn as nn
from attention import AttentionHead
from attention import MultiHeadAttention

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
        self.multi_head = MultiHeadAttention(d_e, d_k, d_v, h)
        self.layer_norm1 = nn.LayerNorm(d_e)
        self.layer_norm2 = nn.LayerNorm(d_e)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.FFN = nn.Sequential(
            nn.Linear(d_e, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_e))
        # z tego co wiem, nn.Linear używa inicjalizacji wag He (He et al. 2015) w wersji unioform
        # można też spróbować normal
        # na przykład:
        # nn.init.kaiming_normal_(self.FFN[0].weight, nonlinearity='relu')


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Compute multi-head attention and project: (seq_len, h * d_v) -> (seq_len, d_e).
        """Pass input through the encoder layer.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """

        # dropout + add + norm
        m_h = self.dropout1(self.multi_head(x))
        m_h = self.layer_norm1(m_h + x)

        y = self.FFN(m_h)
        # dropout + add + Norm
        y = self.dropout2(y)
        y = self.layer_norm2(y + m_h)

        return y

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
        self.encoder_layers = nn.Sequential(EncoderLayer(d_e, d_k, d_v, d_ff, h))
        for _ in range(num_layers-1):
            self.encoder_layers.append(EncoderLayer(d_e, d_k, d_v, d_ff, h))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through each of the encoder layers.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        return self.encoder_layers(x)


