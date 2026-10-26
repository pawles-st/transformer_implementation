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
        num_classes = len(self.categories) if len(self.categories) > 0 else 2

        # Core pipeline components
        self.tokeniser = Tokeniser()
        self.embedder = Embedder(d_e, self.tokeniser.vocab_size)
        self.positional_encoder = PositionalEncoder(d_e)
        self.encoder = Encoder(num_layers, d_e, d_k, d_v, d_ff, h)

        # Classification head: reduces pooled vector (d_e) to class logits (num_classes)
        self.head = nn.Linear(d_e, num_classes)

        # Optimizer for single-example fit() calls
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: str) -> torch.Tensor:
        """Pass raw text input through the entire transformer classifier.

        Args:
            x: Raw input text.

        Returns:
            Logits tensor of shape (1, num_classes).
        """
        # 1. Tokenize text -> (seq_len,)
        tokens = self.tokeniser.encode(x)

        # 2. Embed tokens -> (seq_len, d_e)
        embeddings = self.embedder(tokens)

        # 3. Add positional encodings -> (seq_len, d_e)
        pos_embeddings = self.positional_encoder(embeddings)

        # 4. Pass through transformer encoder stack -> (seq_len, d_e)
        encoded = self.encoder(pos_embeddings)

        # 5. Global mean pooling over sequence length -> (d_e,)
        pooled = encoded.mean(dim=0)

        # 6. Linear head -> (num_classes,)
        logits = self.head(pooled)

        # Return with batch dimension for standard PyTorch loss functions: (1, num_classes)
        return logits.unsqueeze(0)

    def fit(self, x: str, y: Category) -> None:
        """Fit the model on a single labelled example.

        Args:
            x: Raw input text.
            y: Target category label.
        """
        self.train() # set to training mode
        self.optimizer.zero_grad() # zero out the accumulated gradient from previous step

        logits = self.forward(x)

        # Find target class as a tensor with dim (1,)
        label_idx = self.category_to_idx[y]
        target = torch.tensor([label_idx], dtype=torch.long)

        # Compute loss, backpropagate and optimize
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self.optimizer.step()

    def predict(self, x: str) -> Category:
        """Predict the category label of an input text.

        Args:
            x: Raw input text.

        Returns:
            The predicted category label.
        """
        self.eval() # set to inference mode
        with torch.no_grad():
            # Predict the category with argmax
            logits = self.forward(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            return self.categories[pred_idx]
