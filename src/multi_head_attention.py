import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention Module as described in the paper "Attention is All You Need".
    It takes in model size and number of heads.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculate the attention weights and return the weighted sum of values.
        """
        # Compute dot product of Q and K transposed for each head (scaled by sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Reverses the operation performed by split_heads.
        """
        x = x.transpose(1, 2).contiguous()
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Perform linear operations and split into num_heads
        Q = self.split_heads(self.W_q(Q), batch_size)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)

        # Apply scaled dot product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine the attention output heads into a single matrix
        attn_output = self.combine_heads(attn_output)

        # Final linear layer
        output = self.W_o(attn_output)
        return output
