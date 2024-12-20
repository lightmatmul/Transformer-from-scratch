# Transformer from Scratch

This repository contains a PyTorch implementation of the Transformer model as described in the paper "Attention is All You Need" by Vaswani et al. The implementation includes all necessary components such as multi-head attention, positional encoding, and feed-forward networks, with a sample usage.py to test on a generated random set.
  
## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [Multi-Head Attention](#multi-head-attention)
  - [Positional Encoding](#positional-encoding)
  - [Feed-Forward Networks](#feed-forward-networks)
  - [Encoder and Decoder Layers](#encoder-and-decoder-layers)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)


## Overview

The Transformer model is a deep learning model that revolutionized the way sequential data is processed. It eschews recurrence in favor of attention mechanisms, providing significant advantages in parallelization and performance on large-scale applications. This implementation is structured to reflect the modular and clear design of the original Transformer architecture.


## Components

### Multi-Head Attention

This component is pivotal in allowing the Transformer to efficiently process different positions of a single sequence. It splits the attention mechanism into multiple heads, allowing the model to attend to different parts of the sequence simultaneously, enhancing the model's ability to capture various linguistic nuances.


### Positional Encoding

Since the Transformer does not inherently process sequential data as sequential, positional encodings are added to give the model some information about the order of words in sentences. This implementation uses sine and cosine functions to embed positional information, which is then added to the input embeddings.


### Feed-Forward Networks

Each layer in both the encoder and decoder contains a position-wise feed-forward network, which applies a fully connected feed-forward network to each position separately. This part of the model allows non-linear transformations of the data, vital for complex pattern recognition in the sequence data.


### Encoder and Decoder Layers

The encoder maps an input sequence of symbol representations to a sequence of continuous representations. The decoder then takes this sequence and generates an output sequence. Each layer in both the encoder and decoder contains a series of sub-layers including multi-head attention and feed-forward networks, processed with residual connections followed by layer normalization.


## Model Architecture

The model architecture follows the design proposed by Vaswani et al., consisting of an encoder and decoder. The encoder is made up of multiple layers of two sub-layers: multi-head self-attention and position-wise feed-forward networks. The decoder also includes these, plus a third sub-layer that performs multi-head attention over the encoder's output


## Installation

To get started with this project, clone this repository and install the required dependencies.

```bash
git clone https://github.com/lightmatmul/transformer-from-scratch.git
cd transformer-from-scratch
pip install torch
```

## Usage

To use the Transformer model, you can import it in your Python script or Jupyter notebook from the src directory. Here is a basic example of how to initialize and use the model:

```python
from src.transformer import Transformer

# Initialize the Transformer with configuration parameters
transformer = Transformer(src_vocab_size=5000, tgt_vocab_size=5000, d_model=512,
                          num_heads=8, num_layers=6, d_ff=2048, max_seq_length=100,
                          dropout=0.1)

# Example of model usage
output = transformer(src_data, tgt_data)
```
