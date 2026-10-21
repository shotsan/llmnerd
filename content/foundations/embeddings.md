---
title: Embeddings
toc: true
type: docs
math: true
---

After tokenization decomposes raw text into discrete integers, neural networks face a geometric challenge: how can arbitrary discrete indices be represented in a space where mathematical operations like addition, dot products, and gradient propagation carry semantic meaning?

The simplest representation is a **one-hot vector** $e_i \in \{0, 1\}^V$, where an index is represented by a vector of zeros with a single $1$ at index $i$. While conceptually straightforward, one-hot vectors present fatal bottlenecks for deep sequence models:

1. **Orthogonality and Semantic Blindness**: The dot product between any two distinct one-hot vectors $e_i$ and $e_j$ is identically zero:
   $$
   e_i \cdot e_j = 0 \quad (\forall i \neq j)
   $$
   In one-hot space, the distance between `"cat"` and `"kitten"` is identical to the distance between `"cat"` and `"refrigerator"`. The model cannot share statistical strength across related concepts.
2. **Curse of Dimensionality**: With modern vocabularies of $V = 128{,}000$ tokens (as in Llama 3), a one-hot vector requires 128,000 dimensions per token, almost all of which are empty zeros.

**Dense embeddings** resolve this by projecting discrete tokens into a continuous low-dimensional manifold $\mathbb{R}^d$ (typically $d \in [2048, 8192]$). In this continuous space, tokens become coordinate vectors whose geometric distances (Euclidean norm) and angular orientations (cosine similarity) reflect linguistic, syntactic, and functional relationships.

---

## The Embedding Lookup & Semantic Geometry

The embedding layer performs a direct row-slice lookup from a parameter matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$, mapping discrete integer IDs to continuous coordinates:

```
Discrete Token ID          Weight Matrix W_E ∈ ℝ^(V × d)            Dense Vector in ℝ^d
     id: 1 ("cat") ───►  ┌──────────────────────────────┐
                         │ row 0: [ 0.05, -0.12,  0.08] │
     Lookup row 1  ───►  │ row 1: [ 0.80,  0.10, -0.50] │ ──► e_cat = [0.80, 0.10, -0.50]
                         │ row 2: [ 0.15,  0.90,  0.20] │
                         │ row 3: [ 0.78,  0.12, -0.48] │
                         │ row 4: [ 0.10, -0.05,  0.85] │
                         └──────────────────────────────┘
```

---

## Step-by-Step Toy Walkthrough: Embedding Arithmetic

To understand how embedding parameters are indexed and how semantic proximity is quantified, let us construct a miniature embedding space with vocabulary size $V=5$ and hidden dimension $d=3$.

### Step 1: The Embedding Matrix $\mathbf{W}_E$
Let the vocabulary be:
$$
\mathcal{V} = \{ 0: \text{"the"}, \quad 1: \text{"cat"}, \quad 2: \text{"sat"}, \quad 3: \text{"dog"}, \quad 4: \text{"mat"} \}
$$

The model maintains a learned weight matrix $\mathbf{W}_E \in \mathbb{R}^{5 \times 3}$:

$$
\mathbf{W}_E = \begin{bmatrix}
0.05 & -0.12 & 0.08 \\
0.80 &  0.10 & -0.50 \\
0.15 &  0.90 &  0.20 \\
0.78 &  0.12 & -0.48 \\
0.10 & -0.05 &  0.85
\end{bmatrix}
\begin{matrix}
\leftarrow \text{row 0 ("the")} \\
\leftarrow \text{row 1 ("cat")} \\
\leftarrow \text{row 2 ("sat")} \\
\leftarrow \text{row 3 ("dog")} \\
\leftarrow \text{row 4 ("mat")}
\end{matrix}
$$

### Step 2: Index Lookup Operation
When the token sequence `["cat", "dog"]` (indices `[1, 3]`) enters the model, the lookup operation retrieves the corresponding rows:

$$
\mathbf{e}_{\text{cat}} = \mathbf{W}_E[1] = \begin{bmatrix} 0.80, & 0.10, & -0.50 \end{bmatrix}
$$
$$
\mathbf{e}_{\text{dog}} = \mathbf{W}_E[3] = \begin{bmatrix} 0.78, & 0.12, & -0.48 \end{bmatrix}
$$

Mathematically, this lookup is equivalent to multiplying a one-hot vector $e_1^T \in \mathbb{R}^{1 \times V}$ by the weight matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$:
$$
e_1^T \mathbf{W}_E = \mathbf{e}_{\text{cat}}
$$
In production frameworks (PyTorch, JAX), this is implemented as an $\mathcal{O}(1)$ direct memory pointer lookup rather than an expensive matrix multiplication.

### Step 3: Measuring Semantic Alignment
To quantify how closely related `"cat"` and `"dog"` are in this learned space, we calculate their **dot product** and **cosine similarity**:

1. **Dot Product**:
   $$
   \mathbf{e}_{\text{cat}} \cdot \mathbf{e}_{\text{dog}} = (0.80)(0.78) + (0.10)(0.12) + (-0.50)(-0.48)
   $$
   $$
   = 0.624 + 0.012 + 0.240 = \mathbf{0.876}
   $$

2. **Vector Magnitudes (Euclidean Norms)**:
   $$
   \|\mathbf{e}_{\text{cat}}\| = \sqrt{0.80^2 + 0.10^2 + (-0.50)^2} = \sqrt{0.64 + 0.01 + 0.25} = \sqrt{0.90} \approx 0.9487
   $$
   $$
   \|\mathbf{e}_{\text{dog}}\| = \sqrt{0.78^2 + 0.12^2 + (-0.48)^2} = \sqrt{0.6084 + 0.0144 + 0.2304} = \sqrt{0.8532} \approx 0.9237
   $$

3. **Cosine Similarity**:
   $$
   \cos(\theta) = \frac{\mathbf{e}_{\text{cat}} \cdot \mathbf{e}_{\text{dog}}}{\|\mathbf{e}_{\text{cat}}\| \|\mathbf{e}_{\text{dog}}\|} = \frac{0.876}{0.9487 \times 0.9237} = \frac{0.876}{0.8763} \approx \mathbf{0.9996}
   $$

The cosine similarity approaches $1.0$, proving that the vectors point in virtually identical directions, encoding their shared categorical identity as domestic animals. In contrast, evaluating `"cat"` against `"the"` yields $\cos(\theta) \approx -0.11$, demonstrating functional orthogonality.

---

## A Bit Deeper: Systems & Mechanics

### Weight Tying (Tied Output Projections)
In an autoregressive language model, an embedding lookup occurs at the very start (converting token IDs to hidden states), and an inverse projection occurs at the very end: the final hidden state $h_L \in \mathbb{R}^d$ is projected onto the vocabulary via a language model head $\mathbf{W}_{\text{head}} \in \mathbb{R}^{d \times V}$ to compute unnormalized log-probabilities (logits):

$$
\text{logits} = h_L \mathbf{W}_{\text{head}}
$$

**Weight Tying** (Press & Wolf 2016) enforces that the output projection matrix is literally the transpose of the input embedding matrix:
$$
\mathbf{W}_{\text{head}} = \mathbf{W}_E^T
$$

#### Engineering Significance:
1. **Parameter Savings**: In a model with hidden size $d=4096$ and vocabulary $V=128{,}000$, a single embedding matrix contains $4096 \times 128{,}000 \approx \mathbf{524\text{ million}}$ parameters (over 1 GB of VRAM in float16). Tying weights eliminates an entire 524M-parameter matrix.
2. **Representational Duality**: It forces the geometric space where tokens are read into the model to be identical to the space where tokens are predicted, serving as a powerful regularizer against overfitting on rare tokens.

### Embedding Scaling by $\sqrt{d_{\text{model}}}$
In the original Transformer architecture (Vaswani et al. 2017), the raw lookup vectors are multiplied by $\sqrt{d_{\text{model}}}$ before adding positional encodings:

$$
\mathbf{x}_i = \sqrt{d_{\text{model}}} \cdot \mathbf{W}_E[t_i] + \mathbf{PE}_i
$$

#### Why Scale?
When embedding weights are initialized using standard Gaussian initialization (mean 0, variance $1/d_{\text{model}}$), the expected norm of an embedding vector is $\mathcal{O}(1)$. However, positional encoding values (sinusoids) also have magnitudes on the order of $\mathcal{O}(1)$. Adding them without scaling would cause the positional signal to overpower the semantic information. Scaling by $\sqrt{d_{\text{model}}}$ elevates the semantic signal variance to $\mathcal{O}(\sqrt{d_{\text{model}}})$, ensuring the model retains semantic dominance while using positional values as coordinate modulations.

### Normalization Dynamics: LayerNorm vs. RMSNorm
Embeddings entering deep transformer stacks must be normalized to prevent exploding or vanishing activations across dozens of layers.

```
Standard LayerNorm (Vaswani 2017):
  μ = (1/d) Σ x_i
  σ² = (1/d) Σ (x_i - μ)²
  y = [(x - μ) / √(σ² + ε)] ⊙ γ + β   (Requires 2 passes over memory)

RMSNorm (Zhang & Sennrich 2019, Llama 2/3, Mistral):
  RMS(x) = √[(1/d) Σ x_i² + ε]
  y = [x / RMS(x)] ⊙ γ               (Strips mean-centering; 10-50% faster)
```

By removing the requirement to compute the mean $\mu$, RMSNorm reduces GPU memory read/write passes (SRAM-to-HBM bandwidth) without degrading training loss or perplexity.

---

## Seminal Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1310.4546" title="Distributed Representations of Words and Phrases and their Compositionality" subtitle="Mikolov et al. (NeurIPS 2013) • Demonstrated linear vector analogies (King - Man + Woman ≈ Queen) in continuous spaces." >}}
  {{< card link="https://arxiv.org/abs/1608.05859" title="Using the Output Embedding to Improve Language Models" subtitle="Press & Wolf (EACL 2017) • Formulated Weight Tying, binding input and output embedding matrices." >}}
  {{< card link="https://arxiv.org/abs/1910.07467" title="Root Mean Square Layer Normalization" subtitle="Zhang & Sennrich (NeurIPS 2019) • Established RMSNorm as a computationally efficient replacement for LayerNorm." >}}
{{< /cards >}}
