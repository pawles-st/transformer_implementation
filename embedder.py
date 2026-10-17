import torch
import torch.nn as nn

class Embedder(nn.Module):
    """Text embedder transforming input into multidimensional real vectors."""

    def __init__(self, d_e: int, vocab_size: int) -> None:
        """Initialise the embedder.

        Args:
            d_e: Embedding dimension.
            vocab_size: Number of tokens in vocabulary.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Transform a tensor of tokens into their numerical embeddings.
        
        Args:
            token_ids: LongTensor of shape (seq_len,).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        raise NotImplementedError

