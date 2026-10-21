---
title: Foundations
toc: true
type: docs
---

The mathematical and engineering principles that form the foundation of Large Language Models. Rather than treating architectures as black boxes, this section explores the representational friction and computational bottlenecks that catalyzed each breakthrough.

---

## Conceptual Pillars

{{< hextra/feature-grid >}}
{{< hextra/feature-card
  title="Tokenization"
  subtitle="The discrete symbol-to-index boundary: Information-theoretic subword compression, Byte-Pair Encoding from scratch, UTF-8 byte fallbacks, and multilingual fertility."
  link="tokenization"
>}}

{{< hextra/feature-card
  title="Embeddings"
  subtitle="From discrete indices to continuous semantic manifolds: Vector arithmetic, cosine geometry, weight tying, and RMSNorm stabilization."
  link="embeddings"
>}}

{{< hextra/feature-card
  title="Positional Encoding"
  subtitle="Breaking permutation symmetry in self-attention: Harmonic sinusoidal waves, complex 2D Rotary Position Embeddings (RoPE), ALiBi distance biases, and long-context YaRN scaling."
  link="positional_encoding"
>}}

{{< hextra/feature-card
  title="Attention Mechanisms"
  subtitle="Parallel associative memory retrieval: Scaled dot-product matrix arithmetic, causal masking, KV-cache memory bandwidth walls, MQA/GQA, and FlashAttention SRAM tiling."
  link="attention_mechanisms"
>}}

{{< hextra/feature-card
  title="Model Architecture"
  subtitle="The evolutionary arc: From 2017 Transformer encoder-decoders to BERT, GPT, T5, sparse Mixture of Experts (MoE), DeepSeek Multi-Head Latent Attention (MLA), and text diffusion."
  link="model_architecture"
>}}
{{< /hextra/feature-grid >}}
