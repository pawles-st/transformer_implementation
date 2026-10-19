import torch

class Tokeniser:
    """Tokeniser for converting raw text into integer token sequences.

        doctest:
            >>> tokeniser = Tokeniser()
            >>> corpus = ["Hello, world!"]
            >>> tokeniser.build_vocab(corpus)
            >>> encoded = tokeniser.encode("Hello, world!")
            >>> encoded
            tensor([0, 1, 2, 3])
            >>> decoded = tokeniser.decode(encoded)
            >>> decoded
            'Hello , world !'
        """

    def __init__(self) -> None:
        """Initialise the tokeniser."""
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
            
    def _tokenize(self, text: str) -> list[str]:
        characters = [".", ",", "!", "?", "(", ")"]

        for char in characters:
            text = text.replace(char, f" {char} ")
        
        return text.split()
    
    def build_vocab(self, texts: list[str]) -> None:
        for text in texts:
            tokens = self._tokenize(text)

            for token in tokens:
                if token not in self.token_to_id:
                    token_id = len(self.token_to_id)

                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token 

    def encode(self, text: str) -> torch.Tensor:
        """Encode a string into a 1D tensor of token IDs.

        Args:
            text: Input string to tokenise.

        Returns:
            LongTensor of shape (seq_len,).
        """
        
        tokens = self._tokenize(text)
        token_ids = [self.token_to_id[token] for token in tokens]
        
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a 1D tensor of token IDs into a string.

        Args:
            token_ids: Tensor containing token IDs.

        Returns:
            Decoded string.
        """
        tokens = [self.id_to_token[token_id.item()] for token_id in token_ids]

        return " ".join(tokens)
        
        
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

