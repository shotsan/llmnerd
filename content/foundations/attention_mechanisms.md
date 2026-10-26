---
title: Attention Mechanisms
toc: true
type: docs
math: true
weight: 4
---

Before the emergence of attention, sequence modeling was dominated by Recurrent Neural Networks (RNNs, LSTMs, and GRUs). In an RNN, computing the hidden state $h_t$ at time step $t$ strictly required the prior state $h_{t-1}$:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)
$$

This unbroken recurrence chain created two fundamental engineering roadblocks:

1. **The Parallelization Barrier**: An RNN cannot process token $t+1$ until token $t$ finishes executing. Training could not scale efficiently across distributed GPU clusters along the sequence dimension.
2. **Information Bottlenecks**: Compressing an entire sequence of hundreds or thousands of tokens into a single, fixed-size vector $h_t$ caused early context to be gradually overwritten and forgotten.

**Attention** eliminated sequential recurrence entirely. Instead of passing state through time, attention treats every token in a sequence as an addressable database entry. Contextualization is reframed as a parallelized, content-based associative retrieval:

* **Query ($\mathbf{Q}$)**: *"What information am I searching for?"*
* **Key ($\mathbf{K}$)**: *"What topics and context do I represent?"*
* **Value ($\mathbf{V}$)**: *"What raw content do I provide if selected?"*

By computing compatibility scores between Queries and Keys, each token dynamically routes information from across the entire sequence in a single parallel operation.

---

## Scaled Dot-Product Attention

The core operation of the Transformer architecture is **Scaled Dot-Product Attention** (Vaswani et al. 2017). It maps a set of Query vectors to a weighted sum of Value vectors based on compatibility scores computed against Key vectors:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}
$$

### Mathematical Definitions & Tensor Shapes

To understand how attention operates across parallel hardware, each symbol in the governing equation is defined below with its exact geometric meaning and tensor dimensions:

| Symbol | Shape | Description & Mathematical Role |
| :--- | :--- | :--- |
| $\mathbf{X}$ | $\mathbb{R}^{T \times d_{\text{model}}}$ | **Input Sequence Matrix**: $T$ tokens, each represented as a $d_{\text{model}}$-dimensional continuous embedding vector. |
| $\mathbf{W}_Q, \mathbf{W}_K$ | $\mathbb{R}^{d_{\text{model}} \times d_k}$ | **Query & Key Projections**: Learned linear weights mapping input embeddings into a shared $d_k$-dimensional matching subspace. |
| $\mathbf{W}_V$ | $\mathbb{R}^{d_{\text{model}} \times d_v}$ | **Value Projection**: Learned linear weights mapping input embeddings into the $d_v$-dimensional informational subspace (usually $d_v = d_k$). |
| $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ | $\mathbb{R}^{T \times d_k}$ | **Query Matrix**: Row $\mathbf{q}_i \in \mathbb{R}^{d_k}$ represents the retrieval request of token $i$ (*"what information do I need?"*). |
| $\mathbf{K} = \mathbf{X}\mathbf{W}_K$ | $\mathbb{R}^{T \times d_k}$ | **Key Matrix**: Row $\mathbf{k}_j \in \mathbb{R}^{d_k}$ represents the indexing descriptor of token $j$ (*"what context do I offer?"*). |
| $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ | $\mathbb{R}^{T \times d_v}$ | **Value Matrix**: Row $\mathbf{v}_j \in \mathbb{R}^{d_v}$ represents the raw content extracted from token $j$ (*"what payload do I deliver?"*). |
| $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ | $\mathbb{R}^{T \times T}$ | **Raw Compatibility Matrix**: Unnormalized scalar dot products. Entry $S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$ measures directional alignment between query $i$ and key $j$. |
| $\sqrt{d_k}$ | Scalar $\in \mathbb{R}^+$ | **Softmax Temperature**: Normalizing divisor. Because $\operatorname{Var}(\mathbf{q}_i \cdot \mathbf{k}_j) = d_k$, dividing by $\sqrt{d_k}$ restores variance to $1.0$, preventing softmax saturation. |
| $\mathbf{M}$ | $\mathbb{R}^{T \times T}$ | **Autoregressive Causal Mask**: An upper-triangular mask ($M_{ij} = 0$ for $j \le i$, and $-\infty$ for $j > i$) preventing position $i$ from attending to future tokens $j > i$. |
| $\mathbf{A}$ | $\mathbb{R}^{T \times T}$ | **Attention Weight Matrix**: $\operatorname{softmax}((\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}) + \mathbf{M})$. Row $\mathbf{A}[i, :]$ forms a valid probability distribution over visible tokens ($\sum_{j=1}^T A_{ij} = 1$). |
| $\mathbf{O} = \mathbf{A}\mathbf{V}$ | $\mathbb{R}^{T \times d_v}$ | **Contextual Output Matrix**: Each token's final representation $\mathbf{o}_i = \sum_{j=1}^i A_{ij}\mathbf{v}_j$ is a dynamic convex combination of value vectors. |

{{< visual/attention-diagram >}}

---

## Forward Pass Walkthrough

To trace the arithmetic mechanics, let us calculate the forward pass for an autoregressive sequence of 3 tokens ($T=3$) with a head dimension of $d_k = 2$:
* $t_1$: `"The"`
* $t_2$: `"cat"`
* $t_3$: `"sat"`

### 1. Projecting Queries, Keys, and Values
Each token embedding is projected into Query, Key, and Value coordinate matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{3 \times 2}$:

$$
\mathbf{Q} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad
\mathbf{K} = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}, \quad
\mathbf{V} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 4.0 \\ 1.0 & 2.0 \end{bmatrix}
$$

### 2. Computing Attention Scores
We compute the pairwise dot product $\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{3 \times 3}$, where entry $S_{ij}$ represents the compatibility between query $i$ and key $j$:

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^T = \begin{bmatrix}
1.0 & 1.0 & 0.0 \\
2.0 & 0.0 & 4.0 \\
2.0 & 1.0 & 2.0
\end{bmatrix}
$$

### 3. Temperature Scaling
Dividing by $\sqrt{d_k} = \sqrt{2} \approx 1.414$ normalizes score variance to $1.0$, preventing extreme values that cause softmax gradients to saturate:

$$
\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{2}} \approx \begin{bmatrix}
0.71 & 0.71 & 0.00 \\
1.41 & 0.00 & 2.83 \\
1.41 & 0.71 & 1.41
\end{bmatrix}
$$

> **The Mathematical Intuition for Scaling**:  
> If components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance, their dot product has mean $0$ and variance $d_k$.  
> When $d_k$ is large (e.g., $d_k = 128$), dot products grow to large magnitudes ($>15$), pushing $\operatorname{softmax}$ into flat saturation regions where gradients vanish ($\approx 0$). Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$.

### 4. Causal Masking
In an autoregressive decoder, token $t$ is strictly forbidden from looking into future tokens ($t' > t$). We enforce this by setting the upper triangle of the score matrix to $-\infty$ ($e^{-\infty} = 0$):

$$
\mathbf{S}_{\text{masked}} = \begin{bmatrix}
0.71 & -\infty & -\infty \\
1.41 & 0.00 & -\infty \\
1.41 & 0.71 & 1.41
\end{bmatrix}
$$

### 5. Softmax Normalization
Applying $\operatorname{softmax}$ row-wise normalizes scores into a valid probability distribution ($\sum_j A_{ij} = 1$):

$$
\mathbf{A} = \operatorname{softmax}(\mathbf{S}_{\text{masked}}) = \begin{bmatrix}
1.000 & 0.000 & 0.000 \\
0.804 & 0.196 & 0.000 \\
0.401 & 0.198 & 0.401
\end{bmatrix}
$$

* **Row 1 (`"The"`)**: Attends solely to itself ($100\%$).
* **Row 2 (`"cat"`)**: Distributes attention between `"The"` ($80.4\%$) and `"cat"` ($19.6\%$).
* **Row 3 (`"sat"`)**: Synthesizes context across `"The"` ($40.1\%$), `"cat"` ($19.8\%$), and `"sat"` ($40.1\%$).

### 6. Value Aggregation
Multiplying attention weights by the Value matrix $\mathbf{V}$ produces the contextual output $\mathbf{O} \in \mathbb{R}^{3 \times 2}$:

$$
\mathbf{O} = \mathbf{A}\mathbf{V} = \begin{bmatrix}
1.000 & 0.000 & 0.000 \\
0.804 & 0.196 & 0.000 \\
0.401 & 0.198 & 0.401
\end{bmatrix}
\begin{bmatrix}
2.0 & 0.0 \\
0.0 & 4.0 \\
1.0 & 2.0
\end{bmatrix}
= \begin{bmatrix}
2.000 & 0.000 \\
1.608 & 0.784 \\
1.203 & 1.594
\end{bmatrix}
$$

Token 3 (`"sat"`) has synthesized information from `"The"` (40.1%), `"cat"` (19.8%), and itself (40.1%), producing a rich, contextual vector representation.

---

## Interactive Attention Simulator

Test the mechanics interactively below. Drag the temperature slider to see how variance scaling influences the sharpness of the probability distribution, or toggle causal masking to compare decoder vs. encoder attention:

{{< visual/attention-interactive >}}

---

## PyTorch Implementation

Here is how scaled dot-product attention translates into clean, idiomatic PyTorch:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor of shape   [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor of shape [batch_size, num_heads, seq_len_v, head_dim]
        mask: Optional boolean or additive mask tensor
    Returns:
        output: Contextualized output tensor [batch_size, num_heads, seq_len_q, head_dim]
        attn_weights: Softmax attention map  [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    d_k = q.size(-1)
    
    # 1. Pairwise compatibility scoring (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Apply causal mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
    # 3. Probability normalization
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. Contextual Value aggregation
    output = torch.matmul(attn_weights, v)
    return output, attn_weights
```

---

## GPU Memory & Hardware Efficiency

### The KV-Cache Bottleneck

During prefill (processing the initial prompt), tokens are known upfront and can be processed in parallel, saturating the GPU's tensor computing cores.

However, during **autoregressive token generation**, the model produces one token at a time. Recomputing past token keys and values from scratch on every step would scale compute quadratically ($\mathcal{O}(T^2)$). Instead, modern LLMs maintain a **KV-Cache** in GPU High-Bandwidth Memory (HBM):

$$
\text{KV-Cache Footprint} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L_{\text{seq}} \times \text{bytes per param}
$$

For a 70B parameter model (80 layers, 64 heads, $d_{\text{head}} = 128$) running at FP16 precision:
* At context length $L = 4{,}096$: $\approx 5.2\text{ GB}$ per user.
* At context length $L = 128{,}000$: **$\approx 163\text{ GB}$ per user**.

When generating each new token, the GPU must stream this entire $163\text{ GB}$ cache from off-chip HBM into on-chip SRAM just to compute dot products against a single new Query vector. The model is not limited by compute (TFLOPs), but by **memory bandwidth (TB/s)**.

### MHA vs. MQA vs. GQA

To break the memory bandwidth bottleneck, modern LLMs use grouped Key-Value sharing architectures:

```
Multi-Head Attention (MHA - Vaswani et al. 2017):
  Queries:  [Q₁] [Q₂] [Q₃] [Q₄] [Q₅] [Q₆] [Q₇] [Q₈]
  Keys:     [K₁] [K₂] [K₃] [K₄] [K₅] [K₆] [K₇] [K₈]  (100% KV Cache)
  Values:   [V₁] [V₂] [V₃] [V₄] [V₅] [V₆] [V₇] [V₈]

Multi-Query Attention (MQA - Shazeer 2019):
  Queries:  [Q₁] [Q₂] [Q₃] [Q₄] [Q₅] [Q₆] [Q₇] [Q₈]
  Keys:     [               K₁               ]  (12.5% KV Cache;
  Values:   [               V₁               ]   slight quality loss)

Grouped-Query Attention (GQA - Ainslie et al. 2023; Llama 3, Mistral):
  Queries:  [Q₁  Q₂]  [Q₃  Q₄]  [Q₅  Q₆]  [Q₇  Q₈]
  Keys:     [  K₁  ]  [  K₂  ]  [  K₃  ]  [  K₄  ]  (25-50% KV Cache;
  Values:   [  V₁  ]  [  V₂  ]  [  V₃  ]  [  V₄  ]   zero quality degradation)
```

By grouping Query heads over shared Key-Value heads, **Grouped-Query Attention (GQA)** cuts memory bandwidth consumption by $4\times$ to $8\times$, drastically accelerating inference throughput.

### FlashAttention & SRAM Tiling

Standard naive attention implementations materialize the intermediate $T \times T$ attention matrix in GPU global memory (HBM). For long sequences ($T = 64\text{k}$), reading and writing this quadratic matrix to HBM introduces severe I/O latency.

**FlashAttention** avoids materializing the full $T \times T$ matrix entirely:
1. **Tiling**: Loads small blocks of $Q, K, V$ into high-speed on-chip SRAM ($192\text{ KB}$ per streaming multiprocessor).
2. **Online Softmax**: Computes softmax normalizers incrementally across tiles using running maximums and sums.
3. **IO-Awareness**: Eliminates quadratic HBM roundtrips, reducing memory operations from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$ and accelerating attention by $2\times$ to $4\times$.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1706.03762" title="Attention Is All You Need" subtitle="Vaswani et al. (NeurIPS 2017) • The seminal paper introducing Scaled Dot-Product and Multi-Head Attention." >}}
  {{< card link="https://arxiv.org/abs/1911.02150" title="Fast Transformer Decoding: One Write-Head is All You Need" subtitle="Shazeer (2019) • Introduced Multi-Query Attention (MQA) to relieve inference memory-bandwidth bottlenecks." >}}
  {{< card link="https://arxiv.org/abs/2305.13245" title="GQA: Training Generalized Multi-Query Transformer Models" subtitle="Ainslie et al. (EMNLP 2023) • Introduced Grouped-Query Attention, now the universal standard in modern LLMs." >}}
  {{< card link="https://arxiv.org/abs/2205.14135" title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" subtitle="Dao et al. (NeurIPS 2022) • Pioneered exact tiled SRAM attention, revolutionizing long-context training throughput." >}}
{{< /cards >}}
