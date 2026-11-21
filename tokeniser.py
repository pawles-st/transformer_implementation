from pathlib import Path
from typing import Iterable
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

class GPT2Tokeniser:
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

    def encode(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text or a batch of texts into token IDs and an attention mask.

        The pretrained tokenizer splits the text into tokens and maps
        them to integer IDs. The end-of-sequence token is appended
        to mark the end of the input sequence.

        If a list of strings is provided, shorter sequences are padded
        with `pad_id` to match the longest sequence in the batch.

        Args:
            text: A single string or a list of strings to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - tokens: LongTensor of shape (batch_size, max_seq_len)
                  containing token IDs.
                - mask: BoolTensor of shape (batch_size, max_seq_len) where
                  True indicates a valid token and False indicates a pad token.
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        sequences: list[torch.Tensor] = []
        for t in texts:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            token_ids.append(self.eos_id)
            sequences.append(torch.tensor(token_ids, dtype=torch.long))

        if is_single:
            tokens = sequences[0].unsqueeze(0)
        else:
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=self.pad_id
            )

        mask = tokens != self.pad_id
        return tokens, mask

    def decode(self, token_ids: torch.Tensor) -> str | list[str]:
        """
        Convert token IDs into text.

        Special tokens, such as EOS and padding, are omitted
        from the decoded text.

        Args:
            token_ids (torch.Tensor): 1D tensor (seq_len,) or 2D tensor
                (batch_size, seq_len) containing token IDs.

        Returns:
            str | list[str]: Reconstructed text or list of texts.
        """
        if token_ids.dim() == 2:
            return [
                self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                for ids in token_ids
            ]

        return self.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=True
        )


class BPETokeniser:
    """
    Compact Byte Pair Encoding (BPE) tokeniser.

    Uses byte-level BPE to provide subword tokenization with a compact,
    task-fitted vocabulary (default: 3,000 tokens) and zero out-of-vocabulary
    words.
    """

    DEFAULT_SAVE_PATH = Path(__file__).parent / "data" / "bpe_tokenizer.json"

    def __init__(
        self,
        tokenizer_file: str | Path | None = None,
        vocab_size: int = 3000,
    ) -> None:
        """
        Initialize or load a compact BPE tokenizer, training and caching it if not found.
        """
        target_path = Path(tokenizer_file) if tokenizer_file else (
            self.DEFAULT_SAVE_PATH if vocab_size == 3000 and self.DEFAULT_SAVE_PATH.exists()
            else Path(__file__).parent / "data" / f"bpe_tokenizer_{vocab_size}.json"
        )

        if not target_path.exists():
            self._train_from_sst2(target_path, vocab_size)

        self.tokenizer = HFTokenizer.from_file(str(target_path))
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    @classmethod
    def _train_from_sst2(cls, target_path: Path, vocab_size: int) -> None:
        import csv
        sst2_file = Path(__file__).parent / "data" / "SST-2" / "train.tsv"
        if not sst2_file.exists():
            raise FileNotFoundError(f"Cannot auto-train BPE tokenizer: '{sst2_file}' not found.")
        with open(sst2_file, "r", encoding="utf-8") as f:
            texts = [row["sentence"] for row in csv.DictReader(f, delimiter="\t")]
        cls._train_bpe(texts, vocab_size, min_frequency=2, save_path=target_path)

    @staticmethod
    def _train_bpe(
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int,
        save_path: Path | None = None,
    ) -> HFTokenizer:
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<eos>"],
        )
        tok = HFTokenizer.from_str(bpe.to_str())
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tok.save(str(save_path))
        return tok

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 3000,
        min_frequency: int = 2,
        save_path: str | Path | None = None,
    ) -> "BPETokeniser":
        """Train a new BPE tokenizer on texts and return an instance."""
        save = Path(save_path) if save_path else cls.DEFAULT_SAVE_PATH
        cls._train_bpe(texts, vocab_size, min_frequency, save)
        return cls(tokenizer_file=save)

    @property
    def vocab_size(self) -> int:
        """Returns total vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert text or batch of texts into token IDs and attention mask."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        sequences: list[torch.Tensor] = []
        for t in texts:
            token_ids = self.tokenizer.encode(t).ids
            token_ids.append(self.eos_id)
            sequences.append(torch.tensor(token_ids, dtype=torch.long))

        if is_single:
            tokens = sequences[0].unsqueeze(0)
        else:
            tokens = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=self.pad_id
            )

        mask = tokens != self.pad_id
        return tokens, mask

    def decode(self, token_ids: torch.Tensor) -> str | list[str]:
        """Convert token IDs into text, omitting special tokens."""
        if token_ids.dim() == 2:
            return [self._decode_single(ids) for ids in token_ids]
        return self._decode_single(token_ids)

    def _decode_single(self, token_ids: torch.Tensor) -> str:
        cleaned = [
            token_id for token_id in token_ids.tolist()
            if token_id not in (self.pad_id, self.eos_id)
        ]
        return self.tokenizer.decode(cleaned)


