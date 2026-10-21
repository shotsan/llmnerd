---
title: Foundations
toc: true
type: docs
---

The mathematical and engineering principles that form the foundation of Large Language Models. Rather than treating architectures as black boxes, this section explores the representational friction and computational bottlenecks that catalyzed each breakthrough.

---

## Conceptual Pillars

{{< cards >}}
  {{< card link="tokenization" title="Tokenization" subtitle="The discrete symbol-to-index boundary: Information-theoretic subword compression, Byte-Pair Encoding from scratch, UTF-8 byte fallbacks, and multilingual fertility." >}}
  {{< card link="embeddings" title="Embeddings" subtitle="From discrete indices to continuous semantic manifolds: Vector arithmetic, cosine geometry, weight tying, and RMSNorm stabilization." >}}
  {{< card link="positional_encoding" title="Positional Encoding" subtitle="Breaking permutation symmetry in self-attention: Harmonic sinusoidal waves, complex 2D Rotary Position Embeddings (RoPE), ALiBi distance biases, and long-context YaRN scaling." >}}
  {{< card link="attention_mechanisms" title="Attention Mechanisms" subtitle="Parallel associative memory retrieval: Scaled dot-product matrix arithmetic, causal masking, KV-cache memory bandwidth walls, MQA/GQA, and FlashAttention SRAM tiling." >}}
  {{< card link="model_architecture" title="Model Architecture" subtitle="The evolutionary arc: From 2017 Transformer encoder-decoders to BERT, GPT, T5, sparse Mixture of Experts (MoE), DeepSeek Multi-Head Latent Attention (MLA), and text diffusion." >}}
{{< /cards >}}
