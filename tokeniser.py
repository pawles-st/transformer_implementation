import torch

class Tokeniser:
    """Tokeniser for converting raw text into integer token sequences."""

    def __init__(self) -> None:
        """Initialise the tokeniser."""
        raise NotImplementedError

    def encode(self, text: str) -> torch.Tensor:
        """Encode a string into a 1D tensor of token IDs.

        Args:
            text: Input string to tokenise.

        Returns:
            LongTensor of shape (seq_len,).
        """
        raise NotImplementedError

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a 1D tensor of token IDs into a string.

        Args:
            token_ids: LongTensor of token IDs of shape (seq_len,).

        Returns:
            Reconstructed text string.
        """
        raise NotImplementedError

