import torch
import torch.nn as nn
from enum import Enum

from tokeniser import Tokeniser
from embedder import Embedder
from category import Category

class LSTMM(nn.Module):

    def __init__(self, embedding_dim, num_layers = 1, hidden_size = 64):
        super().__init__()

        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)

        self.categories = list(Category)
        self.classifier = nn.Linear(hidden_size, len(self.categories))

    def forward(self, embeddings):

        _, (h_n, _) = self.lstm(embeddings)

        reprezentacja = h_n[-1]

        return self.classifier(reprezentacja)