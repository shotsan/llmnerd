---
title: Embeddings
toc: true
---

Vector representations of tokens in high-dimensional space.

## Overview

Embeddings convert discrete tokens into dense vectors that capture semantic meaning.

## Key Concepts

### Token Embeddings

- Learned lookup table (embedding matrix)
- Each token maps to a vector
- Typical dimensions: 2048, 4096, 8192

### Layer Normalization

- Stabilizes training
- Applied before/after attention/MLP blocks

### Dropout

- Regularization technique
- Prevents overfitting

### Residual Connections

- Enables deep networks
- Helps with gradient flow
