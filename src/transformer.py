import torch
import torch.nn as nn
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    The Transformer model follows the architecture described in "Attention is All You Need".
    It includes an encoder and a decoder, each composed of a stack of layers.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        """
        Initialize the Transformer model.
        Parameters:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): The dimensionality of the input/output tokens.
            num_heads (int): The number of heads in the multi-head attention models.
            num_layers (int): The number of encoder and decoder layers.
            d_ff (int): The dimensionality of the feed-forward layer.
            max_seq_length (int): The maximum length of the input sequences.
            dropout (float): The dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore the padding tokens.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Creates a mask for the target sequence to prevent the decoder from attending to future positions.
        """
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        no_peak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask & no_peak_mask
        return tgt_mask

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        return output
