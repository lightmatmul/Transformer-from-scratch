import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    """
    Represents one layer of the Transformer encoder stack.
    Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Perform forward pass of the encoder layer.
        Parameters:
            src (Tensor): Input tensor to the encoder layer.
            src_mask (Tensor): Mask to be applied on the input tensor.
        Returns:
            Tensor: Output tensor of the encoder layer.
        """
        # Apply self attention
        attn_output = self.self_attn(src, src, src, src_mask)
        # Add & norm
        src = self.norm1(src + self.dropout(attn_output))
        # Apply feed forward network
        output = self.feed_forward(src)
        # Second add & norm
        src = self.norm2(src + self.dropout(output))
        return src
