import torch
from transformers import AutoTokenizer


class Tokeniser:
    """
    Tokeniser based on the pretrained GPT-2 tokenizer.

    Uses the tokenizer released by OpenAI and distributed through
    the Hugging Face model repository. Text is converted into token
    IDs using byte-level Byte Pair Encoding (BPE).

    The tokenizer is pretrained and uses an existing vocabulary;
    this class does not train a tokenizer or build a vocabulary.

    Authors of the original GPT-2 work:
        Alec Radford, Jeff Wu, Rewon Child, David Luan,
        Dario Amodei, and Ilya Sutskever.

    References:
        GPT-2 paper:
        https://openai.com/research/better-language-models/

        Original GitHub repository:
        https://github.com/openai/gpt-2

        Hugging Face tokenizer repository:
        https://huggingface.co/openai-community/gpt2

        Transformers library:
        https://github.com/huggingface/transformers

    License:
        The GPT-2 model repository declares the MIT license.
        The Transformers library is licensed under Apache-2.0.
    """

    def __init__(self) -> None:
        """
        Load the pretrained GPT-2 tokenizer and configure padding.

        Adds a dedicated padding token if one is not already defined,
        then stores the IDs of the padding and end-of-sequence tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Vocabulary size, including added special tokens.
        """
        return len(self.tokenizer)

    def encode(self, text: str) -> torch.Tensor:
        """
        Convert text into a one-dimensional tensor of token IDs.

        The pretrained tokenizer splits the text into tokens and maps
        them to integer IDs. The end-of-sequence token is appended
        to mark the end of the input sequence.

        Args:
            text: Text to tokenize.

        Returns:
            torch.Tensor: One-dimensional tensor of type torch.long
            containing token IDs, including the EOS token.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(self.eos_id)

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Convert a one-dimensional tensor of token IDs into text.

        Special tokens, such as EOS and padding, are omitted
        from the decoded text.

        Args:
            token_ids (torch.Tensor): One-dimensional tensor
                containing token IDs.

        Returns:
            str: Text reconstructed from the token IDs.
        """
        return self.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=True
        )

