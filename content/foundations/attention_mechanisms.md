---
title: Attention Mechanisms
toc: true
type: docs
math: true
weight: 4
---

When reading a complex sentence like *The trophy didn't fit into the brown suitcase because **it** was too big*, understanding what the pronoun **it** refers to requires linking it directly back to *trophy*.

Before the Transformer, sequence models processed text strictly one word at a time from left to right. Every past word had to be compressed into a single, fixed-size hidden vector passed step-by-step along the chain:

1. **The Parallelization Barrier**: Because step $t$ could not begin until step $t - 1$ finished, GPUs could not process tokens in parallel across a sequence during training.
2. **The Information Bottleneck**: Forcing an entire document of thousands of words through a single fixed-width vector caused early details to fade or get overwritten long before later tokens needed them.

**Attention** eliminates sequential passing entirely. Instead of funneling history through a single state vector, attention treats every token in the context window as an addressable entry in a parallel key-value database. Every token simultaneously broadcasts three vectors:

* **Query ($\mathbf{Q}$)**: Encodes what specific context or grammatical partner the token is looking for.
* **Key ($\mathbf{K}$)**: Advertises the syntactic and semantic features the token contains.
* **Value ($\mathbf{V}$)**: Holds the actual information payload contributed once selected.

By taking dot products between Queries and Keys across all token pairs simultaneously, any token can directly retrieve information from any earlier token in a single parallel matrix operation—regardless of how far apart they sit in the text.

---

## Scaled Dot-Product Attention

The fundamental building block of the Transformer is **Scaled Dot-Product Attention** (Vaswani et al. 2017). Given packed matrices of Queries $\mathbf{Q}$, Keys $\mathbf{K}$, and Values $\mathbf{V}$, it computes contextualized representations as a weighted sum of values:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}
$$

### Mathematical Definitions & Tensor Shapes

Each symbol in the governing equation plays a distinct geometric and computational role:

{{< visual/attention-math-defs >}}

{{< visual/attention-diagram >}}

---

## Forward Pass Walkthrough

To trace the arithmetic mechanics, let us calculate the forward pass for an autoregressive sequence of 3 tokens ($T=3$) with a head dimension of $d_k = 2$:
* $t_1$: `The`
* $t_2$: `cat`
* $t_3$: `sat`

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

* **Row 1 (`The`)**: Attends solely to itself ($100\%$).
* **Row 2 (`cat`)**: Distributes attention between `The` ($80.4\%$) and `cat` ($19.6\%$).
* **Row 3 (`sat`)**: Synthesizes context across `The` ($40.1\%$), `cat` ($19.8\%$), and `sat` ($40.1\%$).

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

Token 3 (`sat`) has synthesized information from `The` (40.1%), `cat` (19.8%), and itself (40.1%), producing a rich, contextual vector representation.

---

## Interactive Attention Simulator

Test the mechanics interactively below. Drag the temperature slider to see how variance scaling influences the sharpness of the probability distribution, or toggle causal masking to compare decoder vs. encoder attention:

{{< visual/attention-interactive >}}

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

### MHA vs. GQA vs. MQA

To break this memory-bandwidth wall without sacrificing the expressivity of multiple Query perspectives, modern Transformers share Key and Value heads across groups of Queries:

{{< visual/mha-gqa-mqa >}}

By grouping Query heads over shared Key-Value heads, **Grouped-Query Attention (GQA)** cuts KV-cache capacity and memory-bandwidth pressure by $4\times$ to $8\times$ while matching standard Multi-Head Attention perplexity.

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
