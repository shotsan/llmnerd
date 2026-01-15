---
title: Model Architecture
toc: true
---

The overall structure of transformer-based language models.

## Overview

LLMs are built from stacked transformer blocks with consistent patterns.

## Core Components

### Transformer Block

1. **Attention Layer**
   - Multi-head self-attention
   - Output projection

2. **Feed-Forward Network (FFN)**
   - Two linear transformations
   - GELU activation
   - Expanded hidden dimension (typically 4x embedding size)

3. **Layer Normalization**
   - Pre-norm or post-norm

4. **Residual Connections**
   - Around attention and FFN

### Stacking

- Number of layers (N): 12, 24, 32, 96, etc.
- Each block identical (weight-sharing in some architectures)
