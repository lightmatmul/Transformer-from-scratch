import torch
import torch.nn as nn
import torch.optim as optim
from src.transformer import Transformer

# Configuration
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Initialize the Transformer
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# Setup for training
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training process
transformer.train()
for epoch in range(100):  # Train for 100 epochs
    optimizer.zero_grad()
    # Forward pass: shift tgt data for 'teacher forcing' during training
    output = transformer(src_data, tgt_data[:, :-1])
    # Compute the loss between the output and the shifted target sequence
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
