"""LSTM sentiment classifier operating on pre-computed embeddings."""

import torch
import torch.nn as nn
from category import Category


class LSTMM(nn.Module):
    """embeddings -> dropout -> (Bi)LSTM -> pooling -> dropout -> linear."""

    POOLING_MODES = ("last", "mean", "max", "mean_max")

    def __init__(
        self,
        embedding_dim,
        num_layers=1,
        hidden_size=64,
        dropout_rate=0.5,
        embedding_dropout=0.3,
        bidirectional=True,
        pooling="mean_max",
    ):
        super().__init__()
        if pooling not in self.POOLING_MODES:
            raise ValueError(f"pooling must be one of {self.POOLING_MODES}")

        lstm_dropout = dropout_rate if num_layers > 1 else 0.0

        self.bidirectional = bidirectional
        self.pooling = pooling
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        self.categories = list(Category)
        state_dim = hidden_size * (2 if bidirectional else 1)
        head_in_dim = 2 * state_dim if pooling == "mean_max" else state_dim
        self.classifier = nn.Linear(head_in_dim, len(self.categories))

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (batch, seq_len, embedding_dim)
            mask: optional BoolTensor (batch, seq_len), True = real token
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = embeddings.shape
        if mask is None:
            mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=embeddings.device
            )

        embeddings = self.embedding_dropout(embeddings)

        # Direct LSTM pass (3x faster on CPU without packing/pad sorting overhead)
        outputs, _ = self.lstm(embeddings)
        lengths = mask.sum(dim=1).clamp(min=1)

        if self.pooling == "last":
            batch_idx = torch.arange(batch_size, device=outputs.device)
            if self.bidirectional:
                hidden_dim = self.lstm.hidden_size
                forward_last = outputs[batch_idx, lengths - 1, :hidden_dim]
                backward_first = outputs[batch_idx, 0, hidden_dim:]
                pooled = torch.cat([forward_last, backward_first], dim=-1)
            else:
                pooled = outputs[batch_idx, lengths - 1, :]
        else:
            mask_f = mask.unsqueeze(-1)  # (batch, seq_len, 1)

            if self.pooling in ("mean", "mean_max"):
                summed = (outputs * mask_f.to(outputs.dtype)).sum(dim=1)
                counts = mask_f.sum(dim=1).clamp(min=1).to(outputs.dtype)
                mean_pooled = summed / counts

            if self.pooling in ("max", "mean_max"):
                max_pooled = outputs.masked_fill(~mask_f, float("-inf")).max(dim=1).values

            if self.pooling == "mean_max":
                pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
            elif self.pooling == "max":
                pooled = max_pooled
            else:
                pooled = mean_pooled

        return self.classifier(self.dropout(pooled))