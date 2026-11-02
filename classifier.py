import torch
import torch.nn as nn
import torch.nn.functional as F

from tokeniser import Tokeniser
from embedder import Embedder
from positional_encoder import PositionalEncoder
from encoder import Encoder
from category import Category

# Main module of the project. The class tree is:
#
# Classifier
# |-- Tokeniser
# |-- Embedder
# |-- Positional Encoder
# |-- Encoder
# |   |-- Encoder Layer
# |       |-- MultiHeadAttention
# |           |-- AttentionHead
# |       |-- FeedForwarder
class Classifier(nn.Module):
    """Text classifier based on a transformer encoder."""

    def __init__(
        self,
        num_layers: int,
        d_e: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        h: int,
        lr: float = 1e-4,
    ) -> None:
        """Initialise the classifier.

        Args:
            num_layers: Number of encoder layers.
            d_e: Embedding dimension.
            d_k: Query/key vector dimension.
            d_v: Value vector dimension.
            d_ff: Inner hidden dimension of feed-forward network.
            h: Number of attention heads.
            lr: Learning rate for the internal optimizer.
        """
        super().__init__()

        # Category mapping
        self.categories = list(Category)
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        num_classes = len(self.categories)

        # Core pipeline components
        self.tokeniser = Tokeniser()
        self.embedder = Embedder(d_e, self.tokeniser.vocab_size, padding_idx=self.tokeniser.pad_id)
        self.positional_encoder = PositionalEncoder(d_e)
        self.encoder = Encoder(num_layers, d_e, d_k, d_v, d_ff, h)

        # Classification head: reduces pooled vector (d_e) to class logits (num_classes)
        self.head = nn.Linear(d_e, num_classes)

        # Optimizer for single-example fit() calls
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: str | list[str]) -> torch.Tensor:
        """Pass raw text input through the entire transformer classifier.

        Args:
            x: Raw input text or a batch of text strings.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # 1. Tokenize text -> (batch_size, seq_len) and attention mask
        tokens, mask = self.tokeniser.encode(x)

        # 2. Embed tokens -> (batch_size, seq_len, d_e)
        embeddings = self.embedder(tokens)

        # 3. Add positional encodings -> (batch_size, seq_len, d_e)
        pos_embeddings = self.positional_encoder(embeddings)

        # 4. Pass through transformer encoder stack -> (batch_size, seq_len, d_e)
        encoded = self.encoder(pos_embeddings, mask=mask)

        # 5. Masked global mean pooling over sequence length (dim=1) -> (batch_size, d_e)
        mask_expanded = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        sum_embeddings = (encoded * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = sum_embeddings / token_counts

        # 6. Linear head -> (batch_size, num_classes)
        logits = self.head(pooled)

        return logits

    def fit(self, x: str | list[str], y: Category | list[Category] | torch.Tensor) -> float:
        """Fit the model on a single labelled example or a batch of examples.

        Args:
            x: Raw input text or a list of texts.
            y: Target category label or list of labels or tensor of target class indices.

        Returns:
            The scalar cross-entropy loss value.
        """
        self.train() # set to training mode
        self.optimizer.zero_grad() # zero out the accumulated gradient from previous step

        logits = self.forward(x)

        # Convert y to target tensor of shape (batch_size,)
        if isinstance(y, Category):
            labels = [self.category_to_idx[y]]
        elif isinstance(y, torch.Tensor):
            labels = y
        else: # list[Category]
            labels = [self.category_to_idx[cat] for cat in y]

        if not isinstance(labels, torch.Tensor):
            target = torch.tensor(labels, dtype=torch.long, device=logits.device)
        else:
            target = labels.to(device=logits.device, dtype=torch.long)

        # Compute loss, backpropagate and optimize
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: str | list[str]) -> Category | list[Category]:
        """Predict the category label of input text or a batch of texts.

        Args:
            x: Raw input text or a list of texts.

        Returns:
            The predicted category label (if x is str) or list of category labels (if x is list).
        """
        self.eval() # set to inference mode
        is_single = isinstance(x, str)
        with torch.no_grad():
            # Predict the category with argmax
            logits = self.forward(x)
            pred_indices = torch.argmax(logits, dim=1).tolist()
            predictions = [self.categories[idx] for idx in pred_indices]
            return predictions[0] if is_single else predictions
