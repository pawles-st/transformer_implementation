import torch
import torch.nn as nn
import torch.nn.functional as F

from category import Category
from embedder import Embedder
from tokeniser import BPETokeniser, GPT2Tokeniser
from lstm import LSTMM


class LSTMClassifier(nn.Module):
    """Text classifier based on a bidirectional LSTM."""

    def __init__(
        self,
        num_layers: int = 2,
        d_e: int = 64,
        hidden_dim: int = 64,
        bidirectional: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        tokeniser: BPETokeniser | GPT2Tokeniser | None = None,
        tokenizer_type: str = "bpe",
        vocab_size: int = 3000,
        pooling: str = "mean_max",
        dropout: float = 0.1,
    ) -> None:
        """Initialise the LSTM classifier.

        Args:
            num_layers: Number of recurrent layers.
            d_e: Embedding dimension.
            hidden_dim: Hidden dimension per LSTM direction.
            bidirectional: Whether to use bidirectional LSTM.
            lr: Learning rate for internal optimizer.
            weight_decay: Weight decay penalty for AdamW.
            tokeniser: Tokeniser instance (defaults to BPETokeniser).
            tokenizer_type: Tokenizer type ("bpe" or "gpt2").
            vocab_size: Vocabulary size when using default BPETokeniser.
            pooling: Pooling mode ("mean_max", "max", or "mean").
            dropout: Dropout probability.
        """
        super().__init__()

        self.categories = list(Category)
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.pooling = pooling
        self.bidirectional = bidirectional

        # Core pipeline components
        if tokeniser is not None:
            self.tokeniser = tokeniser
        elif tokenizer_type == "gpt2":
            self.tokeniser = GPT2Tokeniser()
        else:
            self.tokeniser = BPETokeniser(vocab_size=vocab_size)

        self.embedder = Embedder(d_e, self.tokeniser.vocab_size, padding_idx=self.tokeniser.pad_id)
        self.lstm_model = LSTMM(
            embedding_dim=d_e,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            embedding_dropout=dropout,
            bidirectional=bidirectional,
            pooling=pooling,
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x: str | list[str]) -> torch.Tensor:
        """Pass raw text or batch of texts through the LSTM classifier."""
        tokens, mask = self.tokeniser.encode(x)
        embeddings = self.embedder(tokens)
        return self.lstm_model(embeddings, mask=mask)

    def fit(self, x: str | list[str], y: Category | list[Category] | torch.Tensor) -> tuple[float, int]:
        """Fit model on a single labelled example or a batch of examples."""
        self.train()
        self.optimizer.zero_grad()

        logits = self.forward(x)

        if isinstance(y, torch.Tensor):
            target = y.to(device=logits.device, dtype=torch.long)
        elif isinstance(y, Category):
            target = torch.tensor([self.category_to_idx[y]], dtype=torch.long, device=logits.device)
        else:
            target = torch.tensor([self.category_to_idx[cat] for cat in y], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, target)
        loss.backward()
        self.optimizer.step()

        pred_indices = torch.argmax(logits.detach(), dim=1)
        num_correct = int((pred_indices == target).sum().item())
        return loss.item(), num_correct

    def predict(self, x: str | list[str]) -> Category | list[Category]:
        """Predict the category label of input text or batch of texts."""
        self.eval()
        is_single = isinstance(x, str)
        with torch.no_grad():
            logits = self.forward(x)
            pred_indices = torch.argmax(logits, dim=1).tolist()
            predictions = [self.categories[idx] for idx in pred_indices]
            return predictions[0] if is_single else predictions
