import math
import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
    Convert token IDs into learned, scaled embedding vectors.

    Each token ID is mapped to a vector of dimension 'd_e' using
    a trainable embedding matrix. The resulting vectors are scaled
    by sqrt(d_e), following the original Transformer architecture.

    This module does not encode token positions. Positional
    information is added separately by the positional encoder.
    """

    def __init__(self, d_e: int, vocab_size: int) -> None:
        """
        Initialise the embedding layer.

        Creates an embedding matrix with 'vocab_size' rows and
        'd_e' columns. Each row represents a token, and its values
        are trainable parameters initialised by PyTorch.

        Args:
            d_e: Number of dimensions in each embedding vector.
            vocab_size: Number of tokens in the vocabulary.
        """
        super().__init__()
        self.d_e = d_e
        self.embedding = nn.Embedding(vocab_size, d_e)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map token IDs to scaled embedding vectors.

        Each input ID selects a row from the embedding matrix.
        The selected vectors are multiplied by sqrt(d_e).

        Args:
            token_ids: A torch.long tensor containing token IDs.
                It can have any shape, such as seq_len.

        Returns:
            A floating-point tensor with shape
            (*token_ids.shape, d_e).
        """
        return self.embedding(token_ids) * math.sqrt(self.d_e)

