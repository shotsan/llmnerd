---
title: Foundations
toc: false
type: docs
---

<div class="hx-mt-4 hx-mb-4">
{{< hextra/hero-headline >}}
  Language Model Foundations
{{< /hextra/hero-headline >}}
</div>

<div class="hx-mb-10">
{{< hextra/hero-subtitle >}}
  The mathematical formulations and systems engineering mechanics that power modern Large Language Models — tracing the end-to-end computational pipeline from discrete byte streams to emergent generation.
{{< /hextra/hero-subtitle >}}
</div>

{{< hextra/feature-grid cols="2" >}}
  {{< hextra/feature-card
    link="/foundations/tokenization"
    title="1. Tokenization"
    subtitle="The discrete symbol-to-index boundary: Information-theoretic subword compression, Byte-Pair Encoding (BPE), byte-level fallbacks, and cross-lingual fertility."
    style="background: radial-gradient(ellipse at 50% 80%,rgba(56,189,248,0.35),hsla(0,0%,100%,0));"
  >}}

  {{< hextra/feature-card
    link="/foundations/embeddings"
    title="2. Embeddings"
    subtitle="From discrete indices to continuous manifolds: Parameter matrices, cosine geometry, dual projection weight tying, and RMSNorm stabilization."
    style="background: radial-gradient(ellipse at 50% 80%,rgba(168,85,247,0.35),hsla(0,0%,100%,0));"
  >}}

  {{< hextra/feature-card
    link="/foundations/positional_encoding"
    title="3. Positional Encoding"
    subtitle="Breaking permutation symmetry in self-attention: Harmonic sinusoidal waves, complex 2D Rotary Position Embeddings (RoPE), ALiBi distance biases, and YaRN context scaling."
    style="background: radial-gradient(ellipse at 50% 80%,rgba(244,63,94,0.35),hsla(0,0%,100%,0));"
  >}}

  {{< hextra/feature-card
    link="/foundations/attention_mechanisms"
    title="4. Attention Mechanisms"
    subtitle="Parallel associative memory retrieval: Scaled dot-product matrix arithmetic, causal masking, KV-cache memory bandwidth walls, MQA/GQA, and FlashAttention SRAM tiling."
    style="background: radial-gradient(ellipse at 50% 80%,rgba(245,158,11,0.35),hsla(0,0%,100%,0));"
  >}}

  {{< hextra/feature-card
    link="/foundations/model_architecture"
    title="5. Model Architecture"
    subtitle="The evolutionary arc: From 2017 Transformer encoder-decoders to BERT, GPT, T5, sparse Mixture of Experts (MoE), DeepSeek Multi-Head Latent Attention (MLA), and text diffusion."
    class="md:hx-col-span-2"
    style="background: radial-gradient(ellipse at 50% 80%,rgba(16,185,129,0.35),hsla(0,0%,100%,0));"
  >}}
{{< /hextra/feature-grid >}}
