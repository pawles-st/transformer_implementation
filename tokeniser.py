import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

class Tokeniser:
    """
    Tokeniser for converting raw text into integer token sequences.
    
    Doctest:
        >>> tokeniser = Tokeniser("Hello world!")
        >>> tokeniser.decode(tokeniser.encode("Hello world!"))
        'hello world !'
    """

    def __init__(self, text: str) -> None:
        """
        Initialise the tokeniser and build its vocabulary.

        The input text is tokenised using 'basic_english'.

        'Counter' counts how many times each token occurs in the
        tokenised text. This frequency information is used by 'Vocab'
        to build the vocabulary.

        'Vocab' assigns an integer ID to each token and provides
        mappings between tokens and their IDs. The special tokens
        '<pad>', '<unk>', and '<EOS>' are placed at the beginning
        of the vocabulary in the specified order.

        Args:
            text: Text used to build the vocabulary.
        """
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = Vocab(
            Counter(self.tokenizer(text)),
            specials=["<pad>", "<unk>", "<EOS>"],
            specials_first=True,
        )
        self.vocab_size: int = len(self.vocab)
            
            
    def encode(self, text: str) -> torch.Tensor:
        """
        Convert a string into a 1D tensor of token IDs.

        The tokenizer splits the text into tokens, and the vocabulary
        maps them to integer IDs. Tokens not found in the vocabulary
        are mapped to the '<unk>' token.

        Args:
            text: Input text to tokenize.

        Returns:
            A 1D torch.Tensor of dtype torch.long with shape (seq_len,).
        """
        words: list[str] = self.tokenizer(text)  
        ids: list[int] = [self.vocab[word] for word in words]
        long_tensor = torch.tensor(ids, dtype=torch.long)
        return long_tensor

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Convert a 1D tensor of token IDs into a string.

        The vocabulary maps each token ID back to its token. The tokens
        are joined with single spaces, so the original text formatting
        is not preserved exactly.

        Args:
            token_ids: A 1D torch.Tensor containing token IDs.

        Returns:
            A string containing the decoded tokens separated by spaces.
        """
        words: list[str] = [self.vocab.itos[i] for i in token_ids.tolist()]
        decoded_string: list[str] = " ".join(words)
        return decoded_string
        
        
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

