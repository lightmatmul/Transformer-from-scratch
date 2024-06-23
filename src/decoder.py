import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    """
    Represents one layer of the Transformer decoder stack.
    Each layer has three sub-layers: a multi-head self-attention mechanism, a multi-head cross-attention mechanism, and a position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        """
        Perform forward pass of the decoder layer.
        Parameters:
            tgt (Tensor): Input tensor to the decoder layer.
            memory (Tensor): Output tensor from the encoder.
            tgt_mask (Tensor): Mask to be applied on the target input.
            memory_mask (Tensor): Mask to be applied for the encoder output.
        Returns:
            Tensor: Output tensor of the decoder layer.
        """
        # Self attention
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        # Cross attention
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        # Feed forward
        output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(output))
        return tgt
