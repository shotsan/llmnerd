---
title: Embeddings
toc: true
type: docs
math: true
weight: 2
---

Once tokenization converts raw text strings into discrete integer token IDs, neural networks face a fundamental geometric question: how can arbitrary discrete numbers like `1923` and `4812` be represented in a space where mathematical operations like vector addition, dot products, and gradient updates carry semantic meaning?

The most basic representation is a **one-hot vector** $\mathbf{e}_i \in \{0, 1\}^V$, where an index is represented by an array of zeros with a single $1$ at position $i$. While straightforward, one-hot encodings suffer from two critical flaws:

1. **Orthogonality and Semantic Blindness**: The dot product between any two distinct one-hot vectors is identically zero:
   $$
   \mathbf{e}_i \cdot \mathbf{e}_j = 0 \quad (\forall i \neq j)
   $$
   In one-hot space, `cat` is just as distant from `kitten` as it is from `refrigerator`. The model cannot share statistical learning across semantically related words.
2. **The Curse of Dimensionality**: In modern LLMs with vocabulary size $V = 128{,}000$ (e.g. LLaMA 3), a single one-hot vector requires 128,000 sparse dimensions per token, wasting memory and computation.

**Dense embeddings** resolve this by projecting discrete tokens into a continuous coordinate space $\mathbb{R}^d$ (typically $d \in [2048, 8192]$). In this continuous space, tokens become dense vectors whose geometric distances (Euclidean norm) and angular orientations (cosine similarity) reflect linguistic meaning and functional relationships.

---

## From Discrete Tokens to Continuous Vectors

The embedding layer operates as an efficient table lookup into a learned parameter matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$:

{{< visual/embedding-blueprint >}}

---

## Vector Geometry & Memory Layout

To see how embedding matrices are indexed and how semantic similarity is quantified geometrically, consider a toy embedding space with vocabulary size $V = 5$ and hidden dimension $d = 3$:

$$
\mathcal{V} = \{ 0: \text{the}, \quad 1: \text{cat}, \quad 2: \text{sat}, \quad 3: \text{dog}, \quad 4: \text{mat} \}
$$

### 1. Embedding Matrix Structure

The model maintains a learned weight matrix $\mathbf{W}_E \in \mathbb{R}^{5 \times 3}$, where each row stores the 3D coordinate vector for one vocabulary token:

| Row Index ($i$) | Token | Dim 1 ($x_1$) | Dim 2 ($x_2$) | Dim 3 ($x_3$) | Row Vector $\mathbf{W}_E[i] \in \mathbb{R}^3$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Row 0** | `the` | $0.05$ | $-0.12$ | $0.08$ | $\begin{bmatrix} 0.05, & -0.12, & 0.08 \end{bmatrix}$ |
| **Row 1** | `cat` | $\mathbf{0.80}$ | $\mathbf{0.10}$ | $\mathbf{-0.50}$ | $\begin{bmatrix} 0.80, & 0.10, & -0.50 \end{bmatrix}$ |
| **Row 2** | `sat` | $0.15$ | $0.90$ | $0.20$ | $\begin{bmatrix} 0.15, & 0.90, & 0.20 \end{bmatrix}$ |
| **Row 3** | `dog` | $\mathbf{0.78}$ | $\mathbf{0.12}$ | $\mathbf{-0.48}$ | $\begin{bmatrix} 0.78, & 0.12, & -0.48 \end{bmatrix}$ |
| **Row 4** | `mat` | $0.10$ | $-0.05$ | $0.85$ | $\begin{bmatrix} 0.10, & -0.05, & 0.85 \end{bmatrix}$ |

### 2. Memory Lookup & Gather

When the sequence `[cat, dog]` (token IDs `[1, 3]`) enters the model, the embedding layer retrieves rows 1 and 3 directly from memory:

$$
\mathbf{e}_{\text{cat}} = \mathbf{W}_E[1] = \begin{bmatrix} 0.80, & 0.10, & -0.50 \end{bmatrix}, \qquad
\mathbf{e}_{\text{dog}} = \mathbf{W}_E[3] = \begin{bmatrix} 0.78, & 0.12, & -0.48 \end{bmatrix}
$$

Mathematically, this lookup is equivalent to multiplying a one-hot row vector $\mathbf{e}_1^T \in \mathbb{R}^{1 \times V}$ by the weight matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$:

$$
\mathbf{e}_1^T \mathbf{W}_E = \mathbf{e}_{\text{cat}}
$$

In GPU kernels, this is executed as an $\mathcal{O}(1)$ pointer-offset memory gather, bypassing sparse matrix multiplication entirely.

### 3. Measuring Semantic Similarity

How close are `cat` and `dog` in this continuous coordinate space?

1. **Dot Product (Projection Alignment)**:
   $$
   \mathbf{e}_{\text{cat}} \cdot \mathbf{e}_{\text{dog}} = (0.80)(0.78) + (0.10)(0.12) + (-0.50)(-0.48) = \mathbf{0.876}
   $$

2. **Vector Magnitudes (Euclidean Norms)**:
   $$
   \|\mathbf{e}_{\text{cat}}\| = \sqrt{0.80^2 + 0.10^2 + (-0.50)^2} = \sqrt{0.64 + 0.01 + 0.25} = \sqrt{0.90} \approx 0.9487
   $$
   $$
   \|\mathbf{e}_{\text{dog}}\| = \sqrt{0.78^2 + 0.12^2 + (-0.48)^2} = \sqrt{0.6084 + 0.0144 + 0.2304} = \sqrt{0.8532} \approx 0.9237
   $$

3. **Cosine Similarity (Angular Affinity)**:
   $$
   \cos(\theta) = \frac{\mathbf{e}_{\text{cat}} \cdot \mathbf{e}_{\text{dog}}}{\|\mathbf{e}_{\text{cat}}\| \, \|\mathbf{e}_{\text{dog}}\|} = \frac{0.876}{0.9487 \times 0.9237} = \frac{0.876}{0.8763} \approx \mathbf{0.9996}
   $$

A cosine similarity of $\approx 0.9996$ indicates that both coordinate vectors point in nearly identical directions, reflecting their shared linguistic status as animals. In contrast, comparing `cat` against the grammatical function word `the` yields $\cos(\theta) \approx -0.11$, demonstrating functional orthogonality.

---

## Interactive Embedding Space Simulator

{{< visual/embedding-interactive >}}

---

## Memory Optimization & Normalization

In production LLMs, the embedding table and residual stream rely on three critical architectural optimizations to minimize VRAM footprint and stabilize deep gradient flow:

{{< visual/embedding-systems >}}

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1310.4546" title="Distributed Representations of Words and Phrases and their Compositionality" subtitle="Mikolov et al. (NeurIPS 2013) • Demonstrated linear vector analogies (King - Man + Woman ≈ Queen) in continuous spaces." >}}
  {{< card link="https://arxiv.org/abs/1608.05859" title="Using the Output Embedding to Improve Language Models" subtitle="Press & Wolf (EACL 2017) • Formulated Weight Tying, binding input and output embedding matrices." >}}
  {{< card link="https://arxiv.org/abs/1910.07467" title="Root Mean Square Layer Normalization" subtitle="Zhang & Sennrich (NeurIPS 2019) • Established RMSNorm as a computationally efficient replacement for LayerNorm." >}}
{{< /cards >}}
