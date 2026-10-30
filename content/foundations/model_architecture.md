---
title: Model Architecture
toc: true
type: docs
math: true
weight: 5
---

The architectural evolution of large language models is driven by three core engineering tensions: **representational capacity**, **parallel training throughput**, and **inference generation latency**.

Rather than remaining a static blueprint, the Transformer evolved through distinct paradigms—each triggered by a specific hardware bottleneck, memory wall, or scaling law discovery—culminating in modern frontier decoders that combine pre-normalized residual streams, gated activations, sparse expert routing, and low-rank latent attention.

---

## 1. Original Encoder-Decoder

Prior to 2017, sequence-to-sequence translation relied on recurrent networks (LSTMs, GRUs) that processed tokens step-by-step. The original Transformer (Vaswani et al. 2017) broke this sequential bottleneck by introducing a dual-stack **Encoder-Decoder** architecture built purely on self-attention and cross-attention:

{{< visual/arch-enc-dec >}}

### Cross-Attention Conditioning
* **The Encoder**: Consumes the full source sentence using unmasked bidirectional attention, computing contextual representations $\mathbf{H}_{\mathrm{enc}} \in \mathbb{R}^{S \times d}$.
* **The Decoder**: Generates the target sequence autoregressively. Each decoder layer performs two distinct attention passes:
  1. **Causal Self-Attention**: Allows target tokens to attend only to earlier generated tokens ($t' \le t$).
  2. **Cross-Attention**: Target tokens emit **Queries** ($\mathbf{Q}_{\mathrm{dec}}$), but retrieve **Keys** ($\mathbf{K}_{\mathrm{enc}}$) and **Values** ($\mathbf{V}_{\mathrm{enc}}$) directly from the encoder representations:
     $$
     \operatorname{CrossAttention}(\mathbf{Q}_{\mathrm{dec}}, \mathbf{K}_{\mathrm{enc}}, \mathbf{V}_{\mathrm{enc}}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}_{\mathrm{dec}} \mathbf{K}_{\mathrm{enc}}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\mathrm{enc}}
     $$

---

## 2. Encoders vs. Decoders

Between 2018 and 2020, researchers decoupled the two halves of the Transformer for fundamentally different regimes:

### Bidirectional Encoders (BERT)
By discarding the decoder and training solely on **Masked Language Modeling (MLM)** (randomly masking 15% of tokens and predicting them from surrounding bidirectional context), BERT achieved breakthroughs in text classification, semantic search, and passage ranking.
* **The Generation Limit**: Because attention is fully bidirectional, an encoder cannot generate open-ended text efficiently—predicting a 100-token paragraph would require 100 separate forward passes from scratch.

### Autoregressive Decoders (GPT)
By discarding the encoder and training purely on **causal next-token prediction** ($p(x_t \mid x_{<t})$), the GPT series demonstrated that next-token prediction at scale absorbs world knowledge, reasoning, and zero-shot task execution without task-specific fine-tuning. Empirical scaling laws established causal decoders as the most compute-efficient architecture for pre-training.

---

## 3. The Modern Decoder Block

Modern frontier foundation models (LLaMA 3, Mistral, Gemma, Qwen) have converged on a standardized two-sublayer autoregressive block engineered for training stability at depth and inference efficiency:

{{< visual/arch-decoder-block >}}

### Pre-LN & Residual Connections
Think of a deep Transformer not as a rigid stack of layers, but as a high-speed **information highway** (the residual stream):
* Each attention layer and MLP block reads from the highway, computes an additive refinement, and deposits it back onto the highway:
  $$
  \mathbf{x}_{l+1} = \mathbf{x}_l + f(\mathbf{x}_l)
  $$
* Under **Pre-RMSNorm**, activations are normalized *before* entering the sublayer, preserving an unobstructed identity path where gradients backpropagate unimpeded from layer $L$ to layer $1$. This allows 100+ layer networks to train stably without exploding or vanishing gradients.

### SwiGLU Feed-Forward Networks
Traditional Transformers used standard MLPs with ReLU or GELU activations. Modern models replace this with **SwiGLU** (Shazeer 2020), a Gated Linear Unit with Swish activation ($\operatorname{Swish}(x) = x \cdot \sigma(x)$):

$$
\operatorname{SwiGLU}(\mathbf{x}) = \left( \operatorname{Swish}(\mathbf{x} \mathbf{W}_{\mathrm{gate}}) \odot (\mathbf{x} \mathbf{W}_{\mathrm{up}}) \right) \mathbf{W}_{\mathrm{down}}
$$

where $\odot$ represents element-wise multiplication. By using a multiplicative gating branch, SwiGLU allows the network to suppress or amplify specific feature coordinates conditionally, delivering higher parameter efficiency.

---

## 4. Sparse Mixture of Experts (MoE)

In standard dense models, every parameter is activated for every token. Scaling capacity to hundreds of billions of parameters linearly multiplies floating-point operations (FLOPs) and compute costs.

**Mixture of Experts (MoE)** decouples total parameter capacity from per-token compute:
* The monolithic Feed-Forward Network is replaced with $E$ parallel expert networks $\{E_1, E_2, \dots, E_E\}$.
* A learned **Gating Router** calculates a sparse probability distribution over the experts:
  $$
  \mathbf{g}(\mathbf{x}) = \operatorname{Softmax}\!\left(\operatorname{TopK}(\mathbf{x} \mathbf{W}_g, k)\right)
  $$
* For each token, only the top-$k$ experts are activated (e.g. $k=2$ out of $E=8$ in Mixtral 8x7B):
  $$
  \mathbf{y} = \sum_{i \in \operatorname{TopK}} g_i(\mathbf{x}) E_i(\mathbf{x})
  $$

{{< visual/architecture-interactive >}}

This delivers the representational capacity of a large model with the inference speed and FLOP budget of a much smaller active network.

---

## 5. Multi-Head Latent Attention (MLA)

Standard Grouped-Query Attention (GQA) reduces the KV-cache by sharing key-value heads. DeepSeek introduced **Multi-Head Latent Attention (MLA)**, taking cache compression to its theoretical limit using low-rank joint projection:

{{< visual/arch-mla >}}

1. **Down-Projection to Compressed Latent**:
   Instead of caching full high-dimensional Keys and Values for every token, MLA projects the hidden state down into a compact latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll n_h d_h$).
2. **The Cache Advantage**:
   **Only $\mathbf{c}_t^{KV}$ is saved to the KV-cache during generation**, slashing the memory footprint by over **85%** compared to standard Multi-Head Attention.
3. **Decompression On-The-Fly**:
   During attention computation, the compact vector is expanded into multi-head keys and values via learned up-projection matrices, combining massive memory savings with full multi-head attention expressivity.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1706.03762" title="Attention Is All You Need" subtitle="Vaswani et al. (NeurIPS 2017) • The foundation: introduced the Transformer encoder-decoder." >}}
  {{< card link="https://arxiv.org/abs/1810.04805" title="BERT: Pre-training of Deep Bidirectional Transformers" subtitle="Devlin et al. (NAACL 2019) • Established bidirectional masked language modeling for representation learning." >}}
  {{< card link="https://arxiv.org/abs/2002.05202" title="GLU Variants Improve Transformer" subtitle="Shazeer (2020) • Introduced the SwiGLU activation function used universally in modern LLMs." >}}
  {{< card link="https://arxiv.org/abs/2401.04088" title="Mixtral of Experts" subtitle="Jiang et al. (Mistral AI 2024) • Open-weights sparse 8x7B MoE matching top dense models at a fraction of active compute." >}}
  {{< card link="https://arxiv.org/abs/2405.04434" title="DeepSeek-V2 & V3 Technical Reports" subtitle="DeepSeek-AI (2024) • Formulated Multi-Head Latent Attention (MLA) and fine-grained DeepSeekMoE architectures." >}}
{{< /cards >}}
