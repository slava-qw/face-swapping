import numpy as np
import torch
from einops import rearrange
from torch import nn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, q, k, v, mask=None):
        # resulted shape will be: [batch, heads, tokens, tokens]
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor
        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_0(out)


class FaceTransferAttnModule(nn.Module):
    def __init__(self, num_heads=4, embed_dim=512):
        super(FaceTransferAttnModule, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = num_heads

        self.mhca_s = MultiHeadCrossAttention(heads=num_heads, dim=embed_dim)
        self.mhca_t = MultiHeadCrossAttention(heads=num_heads, dim=embed_dim)

    def forward(self, W_s, W_t):
        assert W_s.dim() == 3 and W_t.dim() == 3
        qkv_s, qkv_t = self.mhca_s.to_qvk(W_s), self.mhca_t.to_qvk(W_t)  # [batch, tokens, dim*3*heads] each
        q_s, k_s, v_s = tuple(rearrange(qkv_s, 'b t (d k h) -> k b h t d ', k=3, h=self.num_heads))
        q_t, k_t, v_t = tuple(rearrange(qkv_t, 'b t (d k h) -> k b h t d ', k=3, h=self.num_heads))

        A_s = self.mhca_s(q_t, k_s, v_s)
        A_t = self.mhca_t(q_s, k_t, v_t)

        A_s = torch.exp(A_s)
        A_t = torch.exp(A_t)
        A_s = A_s / torch.sum(A_s, dim=-1, keepdim=True)
        A_t = A_t / torch.sum(A_t, dim=-1, keepdim=True)

        swapped_lats = A_s * W_s + A_t * W_t

        return swapped_lats
