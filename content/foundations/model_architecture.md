---
title: Model Architecture
toc: true
type: docs
math: true
weight: 5
---

The architectural evolution of large language models is the story of an empirical search across three engineering tensions: **representational capacity**, **parallel training throughput**, and **inference generation latency**.

Rather than remaining a static blueprint, the Transformer evolved through distinct developmental epochs. Each major architectural transition was triggered by a specific hardware bottleneck, memory wall, or scaling law discovery—culminating in modern frontier models that synthesize pre-normalized residual streams, gated activations, sparse expert routing, and low-rank latent attention.

---

## The Evolutionary Arc of Transformer Architectures

How the fundamental Transformer evolved from a translation dual-stack into modern frontier decoders:

```mermaid
flowchart LR
    subgraph Era1["2017: The Foundation"]
        Trans["<b>Original Transformer</b><br/>Dual-Stack Encoder-Decoder"]
    end

    subgraph Era2["2018–2020: Specialization"]
        direction TB
        BERT["<b>Encoders (BERT)</b><br/>Bidirectional Masked LM"]
        GPT["<b>Decoders (GPT-1/2/3)</b><br/>Autoregressive Causal LM"]
    end

    subgraph Era3["2022–2024: Modern Standards"]
        direction TB
        Llama["<b>Modern Decoder</b><br/>Pre-RMSNorm • RoPE • SwiGLU • GQA"]
        MoE["<b>Sparse MoE</b><br/>Mixtral 8x7B (Top-2 Routing)"]
    end

    subgraph Era4["Frontier Breakthroughs"]
        direction TB
        MLA["<b>DeepSeek MLA & MoE</b><br/>Latent KV Compression"]
        Diff["<b>Diffusion LM</b><br/>Parallel Iterative Denoising"]
    end

    Trans --> BERT
    Trans --> GPT
    GPT --> Llama
    Llama --> MoE
    Llama --> MLA
    Llama --> Diff
```

---

## 1. The Original Encoder-Decoder (Vaswani et al. 2017)

Prior to 2017, sequence-to-sequence translation relied on recurrent networks (LSTMs, GRUs) that processed tokens step-by-step. The original Transformer broke this bottleneck by introducing a dual-stack **Encoder-Decoder** architecture built purely on self-attention and cross-attention:

```
Source Sequence (e.g., French)        Target Sequence (e.g., English)
           │                                       │
     ┌─────▼─────┐                           ┌─────▼─────┐
     │  Encoder  │                           │  Decoder  │
     │(Bidirect.)│                           │ (Causal)  │
     └─────┬─────┘                           └─────┬─────┘
           │                                       │
           └────────────► [Cross-Attention] ◄──────┘
                                   │
                           ┌───────▼───────┐
                           │   Next-Word   │
                           │  Prediction   │
                           └───────────────┘
```

### Asymmetric Cross-Attention Conditioning
* **The Encoder**: Consumes the full source sentence using unmasked bidirectional attention, computing contextual representations $H_{\text{enc}} \in \mathbb{R}^{S \times d}$.
* **The Decoder**: Generates the target sentence autoregressively. Each decoder layer performs two distinct attention passes:
  1. **Causal Self-Attention**: Allows target tokens to attend only to earlier generated tokens ($t' \le t$).
  2. **Cross-Attention**: Target tokens emit **Queries** ($Q_{\text{dec}}$), but retrieve **Keys** ($K_{\text{enc}}$) and **Values** ($V_{\text{enc}}$) directly from the encoder representations:
     $$
     \text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}
     $$

---

## 2. Specialization: Encoders (BERT) vs. Decoders (GPT)

Between 2018 and 2020, researchers discovered that the two halves of the Transformer could be decoupled for fundamentally different tasks:

### Bidirectional Masked Encoders (BERT, RoBERTa)
By discarding the decoder and training solely on **Masked Language Modeling (MLM)** (randomly masking 15% of tokens and training the model to predict them from surrounding bidirectional context), BERT achieved breakthroughs in text classification, semantic search, and passage ranking.
* **The Generation Limit**: Because attention is fully bidirectional, an encoder cannot generate open-ended text efficiently. Predicting a 100-token paragraph would require 100 separate, full forward passes from scratch.

### Autoregressive Causal Decoders (GPT Series)
By discarding the encoder and training purely on **causal next-token prediction** ($p(x_t \mid x_{<t})$), the GPT series demonstrated that next-token prediction at massive scale absorbs world knowledge, logical reasoning, and zero-shot task execution without task-specific fine-tuning. Empirical scaling laws established causal decoders as the most compute-efficient architecture for pre-training.

---

## 3. The Modern Decoder Block: Residual Highways & Pre-LN

Modern frontier foundation models (LLaMA 3, Mistral, Gemma, Qwen) have converged on a standardized autoregressive block engineered for training stability at depth and inference efficiency:

```mermaid
flowchart LR
    subgraph Block["Modern Decoder Block (Pre-RMSNorm)"]
        direction TB
        In["<b>Hidden State (x_l)</b>"] --> Norm1["<b>RMSNorm</b>"]
        Norm1 --> Attn["<b>GQA with RoPE</b>"]
        In --> Add1["<b>Residual Sum (⊕)</b>"]
        Attn --> Add1

        Add1 --> Norm2["<b>RMSNorm</b>"]
        Norm2 --> SwiGLU["<b>SwiGLU MLP</b>"]
        Add1 --> Add2["<b>Residual Sum (⊕)</b>"]
        SwiGLU --> Add2
        Add2 --> Out["<b>Hidden State (x_{l+1})</b>"]
    end
```

### The Residual Highway Mental Model
Think of a deep Transformer not as a rigid stack of layers, but as a high-speed **information highway** (the residual stream).
* Each attention layer and MLP block simply reads from the highway, computes an additive refinement, and deposits it back onto the highway:
  $$
  x_{l+1} = x_l + f(x_l)
  $$
* Under **Pre-RMSNorm**, activations are normalized *before* entering the sublayer, preserving an unobstructed identity path where gradients backpropagate unimpeded from layer $L$ to layer $1$. This allows 100+ layer networks to train stably without exploding or vanishing gradients.

### SwiGLU Gated Multi-Layer Perceptrons
Traditional transformers used standard MLPs with ReLU or GELU activations. Modern models replace this with **SwiGLU** (Shazeer 2020), a Gated Linear Unit with Swish activation ($\text{Swish}(x) = x \cdot \sigma(x)$):

$$
\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}
$$

where $\odot$ represents element-wise multiplication. By using a multiplicative gating branch, SwiGLU allows the network to suppress or amplify specific feature coordinates conditionally, delivering higher parameter efficiency.

---

## PyTorch Reference Implementation: Modern Decoder Block

Here is how a modern Pre-RMSNorm Transformer block is constructed in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root Mean Square normalization (no mean-centering)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x * W_gate) * (x * W_up) -> W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class ModernDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, mlp_hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Pre-Norm Attention + Residual Highway
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        
        # 2. Pre-Norm SwiGLU + Residual Highway
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 4. Sparse Mixture of Experts (MoE)

In standard dense models, every parameter is activated for every token. Scaling capacity to hundreds of billions of parameters linearly multiplies floating-point operations (FLOPs) and compute costs.

**Mixture of Experts (MoE)** decouples total parameter capacity from per-token compute:
* The monolithic Feed-Forward Network is replaced with $E$ parallel expert networks $\{E_1, E_2, \dots, E_n\}$.
* A learned **Gating Router** calculates a probability distribution over the experts:
  $$
  H(x) = \text{Softmax}(\text{TopK}(x \cdot W_g, k))
  $$
* For each token, only the top-$k$ experts are activated (e.g. $k=2$ out of $E=8$ in Mixtral 8x7B):
  $$
  y = \sum_{i \in \text{TopK}} g(x)_i E_i(x)
  $$

```
Input Token x ────► [ Gating Router ] ───► Select Top-2 Experts
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
   [ Expert #2 ]                     [ Expert #7 ]
         │                                 │
         └───────────────► ⨁ ◄─────────────┘
                           │
                     Output Tensor y
```

This delivers the representational capacity of a large model with the inference speed and FLOP budget of a much smaller active network.

---

## 5. Frontier Breakthroughs: Multi-Head Latent Attention (MLA)

Standard Grouped-Query Attention (GQA) reduces KV-cache by sharing key-value heads. DeepSeek introduced **Multi-Head Latent Attention (MLA)**, taking cache compression to its theoretical limit using low-rank joint projection:

```mermaid
flowchart LR
    H["<b>Hidden State (h_t)</b>"] --> Down["<b>Down-Projection</b><br/>W^(DKV)"]
    Down --> Latent["<b>Compressed Latent c_t^(KV)</b><br/>★ ONLY Vector Stored in KV-Cache"]
    Latent --> UpK["<b>Up-Project Keys</b><br/>W^(UK)"]
    Latent --> UpV["<b>Up-Project Values</b><br/>W^(UV)"]
    UpK --> Attn["<b>Attention Core</b>"]
    UpV --> Attn
    RoPE["<b>Decoupled RoPE Key</b>"] --> Attn
```

1. **Down-Projection to Compressed Latent**:
   Instead of caching full high-dimensional Keys and Values for every token, MLA projects the hidden state down into a compact latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll n_h d_h$).
2. **The Cache Advantage**:
   **Only $c_t^{KV}$ is saved to the KV-cache during generation**, slashing the memory footprint by over **85%** compared to standard Multi-Head Attention.
3. **Decompression On-The-Fly**:
   During attention computation, the compact vector is expanded into multi-head keys and values via learned up-projection matrices, combining massive memory savings with state-of-the-art attention expressivity.

---

## Seminal Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1706.03762" title="Attention Is All You Need" subtitle="Vaswani et al. (NeurIPS 2017) • The foundation: introduced the Transformer encoder-decoder." >}}
  {{< card link="https://arxiv.org/abs/1810.04805" title="BERT: Pre-training of Deep Bidirectional Transformers" subtitle="Devlin et al. (NAACL 2019) • Established bidirectional masked language modeling for representation learning." >}}
  {{< card link="https://arxiv.org/abs/2002.05202" title="GLU Variants Improve Transformer" subtitle="Shazeer (2020) • Introduced the SwiGLU activation function used universally in modern LLMs." >}}
  {{< card link="https://arxiv.org/abs/2401.04088" title="Mixtral of Experts" subtitle="Jiang et al. (Mistral AI 2024) • Open-weights sparse 8x7B MoE matching top dense models at a fraction of active compute." >}}
  {{< card link="https://arxiv.org/abs/2405.04434" title="DeepSeek-V2 & V3 Technical Reports" subtitle="DeepSeek-AI (2024) • Formulated Multi-Head Latent Attention (MLA) and fine-grained DeepSeekMoE architectures." >}}
{{< /cards >}}
