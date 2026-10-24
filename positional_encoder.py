import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Add sinusoidal positional encodings to token embeddings.

    Positional encodings represent the positions of tokens in a
    sequence. They are calculated using sine and cosine functions
    of different frequencies, as described in the original
    Transformer architecture.

    The positional encoding is added element-wise to the input
    embeddings. It is calculated for the sequence length provided
    to the forward method.

    Doctest:
        >>> encoder = PositionalEncoder(d_e=4)
        >>> x = torch.zeros((3, 4))
        >>> output = encoder(x)
        >>> output.shape
        torch.Size([3, 4])
        >>> torch.allclose(output[0], torch.tensor([0., 1., 0., 1.]))
        True
    """
    
    def __init__(self, d_e: int) -> None:
        """
        Initialise the positional encoder.

        Stores the embedding dimension, which determines the
        number of components in each positional encoding vector.

        Args:
            d_e: Dimension of the input embedding vectors.
        """
        super().__init__()
        self.d_e = d_e
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encodings to token embeddings.

        For each position, calculates a positional encoding using
        sine functions at even dimensions and cosine functions
        at odd dimensions. The positional encodings are then
        added element-wise to the input tensor.

        Args:
            x: A floating-point tensor of shape (seq_len, d_e),
                containing the token embeddings.

        Returns:
            A tensor of shape (seq_len, d_e) containing the
            input embeddings with positional information added.
        """
        seq_len = x.size(0)
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_e, 2, dtype=x.dtype, device=x.device)*(-math.log(10000.0) / self.d_e))

        pe = torch.zeros(x.size(0), self.d_e, dtype=x.dtype, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe     
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
