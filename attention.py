import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    """Single attention head computing scaled dot-product attention."""

    def __init__(self, d_e: int, d_k: int, d_v: int) -> None:
        """Initialise the attention head.

        Args:
            d_e: Embedding dimension.
            d_k: Query/key vector dimension.
            d_v: Value vector dimension.
        """
        super().__init__()
        self.d_k = d_k
        self.Wk = nn.Parameter(torch.randn(d_e, d_k))
        self.Wq = nn.Parameter(torch.randn(d_e, d_k))
        self.Wv = nn.Parameter(torch.randn(d_e, d_v))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (seq_len, d_e).
        Returns:
            FloatTensor of shape (seq_len, d_v).
        """
        Q = x@self.Wq        
        K = x@self.Wk        
        V = x@self.Wv        

        return torch.softmax(Q@torch.transpose(K,0,1)*(self.d_k**(-0.5)), dim=1)@V


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism concatenating h attention heads and projecting back to original dimension."""

    def __init__(self, d_e: int, d_k: int, d_v: int, h: int) -> None:
        """Initialise multi-head attention.

        Args:
            d_e: Embedding dimension (input and output vector length).
            d_k: Query and key dimension per head.
            d_v: Value dimension per head.
            h: Number of attention heads.
        """
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_e, d_k, d_v) for _ in range(h)])
        self.WO = nn.Parameter(torch.rand(d_v*h, d_e))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Compute multi-head attention and project: (seq_len, h * d_v) -> (seq_len, d_e).
        """Compute multi-head attention over input representations.

        Args:
            x: FloatTensor of shape (seq_len, d_e).

        Returns:
            FloatTensor of shape (seq_len, d_e).
        """
        attention = [head(x) for head in self.heads]
        multi_head = torch.cat(attention, dim=1) 

        # połączenie rezydualne
        # tutaj jeszcze layerNorm (później)
        multi_head = multi_head@self.WO
        return multi_head
