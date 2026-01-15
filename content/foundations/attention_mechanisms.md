---
title: Attention Mechanisms
toc: true
---

The core mechanism enabling LLMs to process sequential data.

## Overview

Attention allows models to weigh the importance of different tokens when processing each token.

## Key Concepts

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention

- Multiple attention heads in parallel
- Each head learns different relationships
- Outputs concatenated and projected

### Variations

#### Causal/Autoregressive Attention
- Masks future tokens
- Used in GPT-style models

#### Bidirectional Attention
- Full context access
- Used in BERT-style models
