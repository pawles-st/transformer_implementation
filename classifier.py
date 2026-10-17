from enum import Enum

class Category(Enum):  # Name subject to change.
    pass

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
class Classifier:
    """Text classifier based on a transformer encoder."""

    def __init__(self, d_e: int, d_k: int, d_v: int, d_ff: int, h: int) -> None:
        """Initialise the classifier.

        Args:
            d_e: Embedding dimension.
            d_k: Query/key vector dimension.
            d_v: Value vector dimension.
            d_ff: Inner hidden dimension of feed-forward network.
            h: Number of attention heads.
        """
        raise NotImplementedError

    def fit(self, x: str, y: Category) -> None:
        """Fit the model on a single labelled example.

        Args:
            x: Raw input text.
            y: Target category label.
        """
        raise NotImplementedError

    def predict(self, x: str) -> Category:
        """Predict the category label of an input text.

        Args:
            x: Raw input text.

        Returns:
            The predicted category label.
        """
        raise NotImplementedError

