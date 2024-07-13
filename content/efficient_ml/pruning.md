---
title: Pruning
type: about
---

arg min L(x;Wp)
subject to 
|| Wp||O <N

Goal is keep the number of non-zero weights within a pre-determined number

## Unstructure pruning vs coarse-grained
1. Unstructured is hard to accelerate due to irregular pattern
2. However, offers more flexiblity
3. Coarse grained less flexible but easy to accelerate
4. Weights of convolutional layer have 4 dimensions: Input channels,
5. output channels, kernel size height, kernel size width

### some of common pruning patterns
1. Irregular or fine grained  Fine grained pruning
2. Pattern-based pruning  follow a particular pattern
3. Vector-level pruning - prune a row or column
4. Kernel-level pruning - Complete kernel
5. Channel - Prune a channel or two
6. Pattern-based pruning: N:M sparsity, for N contiguous elements prune M
7. Pattern based pruning is supported by Nvidia Ampere GPU architectures~ can deliver upt 2x speeds

### Neural Network Pruning
Goal is to prune parameters that are less important
Magnitude pruning considers weights to large absolute value and removes other weights
Importance L1 of 
