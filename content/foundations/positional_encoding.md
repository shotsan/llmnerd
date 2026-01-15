---
title: Positional Encoding
toc: true
---

Methods for incorporating position information into token embeddings.

## Overview

Since attention is permutation-invariant, positional encodings provide sequence order.

## Key Methods

### Sinusoidal Positional Encoding

Used in original Transformer:
- Absolute positions
- Fixed functions based on sine/cosine

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### Rotary Position Embedding (RoPE)

- Relative position information
- Used in LLaMA, GPT-NeoX
- Better generalization to longer sequences

### ALiBi (Attention with Linear Biases)

- Simple and effective
- No learned positional embeddings
- Linear bias based on distance
