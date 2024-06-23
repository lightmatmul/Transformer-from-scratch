import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor to introduce information about the position of tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that they can be summed.
    Uses sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding layer.
        Parameters:
            d_model (int): The dimension of the embeddings.
            max_len (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create a long enough 'pe' matrix that can be sliced according to actual sequence lengths.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register pe as a buffer that is not a model parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Applies the positional encoding to the input embeddings.
        Arguments:
            x (Tensor): The input embeddings (batch_size, seq_len, d_model).
        Returns:
            Tensor: The embeddings with positional encoding added, with dropout applied.
        """
        # Add positional encoding to each embedding and apply dropout.
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
