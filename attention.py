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
        self.Wk = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_e, d_k)))
        self.Wq = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_e, d_k)))
        self.Wv = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_e, d_v)))
        # w internecie przeczytałem, że inicjalizacja xavier (Glorot, Bengio 2010) była w "Attention is all you need"
        # w samym artykule nie ma nic o inicjalizacji wag w macierzach uwagi Wk, Wq, Wv :(
        # w środku można też dać nn.init.normal_(..., mean=0, std=0.02)
        # można sprawdzić co lepiej działa 

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (seq_len, d_e).
            mask: tensor of dim (batch size, 1, seq_len) with zeros and -inf indicating padding
        Returns:
            FloatTensor of shape (batch_size, seq_len, d_v).
        """
        Q = x@self.Wq        
        K = x@self.Wk        
        V = x@self.Wv        

        return torch.softmax(Q@torch.transpose(K,1,2)*(self.d_k**(-0.5)+mask), dim=2)@V


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: # Compute multi-head attention and project: (seq_len, h * d_v) -> (seq_len, d_e).
        """Compute multi-head attention over input representations.

        Args:
            x: FloatTensor of shape (batch_size, seq_len, d_e).
            mask: tensor of dim (batch size, 1, seq_len) with zeros and -inf indicating padding (-inf=<pad>)
        Returns:
            FloatTensor of shape (batch_size, seq_len, d_e).
        """
        attention = [head(x, mask) for head in self.heads]
        multi_head = torch.cat(attention, dim=2) 

        multi_head = multi_head@self.WO
        return multi_head
