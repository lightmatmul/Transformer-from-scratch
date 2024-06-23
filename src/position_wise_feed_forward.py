import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network described in "Attention is All You Need".
    This consists of two dense layers with a ReLU activation in between.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Parameters:
            d_model (int): The size of the input and output dimensions.
            d_ff (int): The size of the hidden layer dimensions.
            dropout (float): Dropout rate.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        The forward method for PositionWiseFeedForward.
        Applies two linear transformations with a ReLU activation in between,
        with dropout applied after the first linear and the ReLU activation.
        """
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
