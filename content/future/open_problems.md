---
title: Open Frontier Problems & Interpretability
toc: true
type: docs
math: true
weight: 5
---

Despite scaling across orders of magnitude in parameters, data, and test-time compute, frontier language models exhibit deep theoretical and mechanistic mysteries that remain unsolved. Why do individual MLP neurons fire on completely unrelated concepts? Why can a model trained on *"A is B"* fail to answer *"Who is B?"* with $0\%$ accuracy? And what are the formal mathematical limits of what a constant-depth Transformer circuit can compute?

Addressing these foundational questions requires uniting **high-dimensional geometry (the Superposition Hypothesis)**, **Sparse Autoencoder (SAE) dictionary learning**, **parametric memory asymmetry analysis**, and **Formal Circuit Complexity ($\mathsf{TC}^0$ vs. $\mathsf{NC}^1$)**.

{{< visual/open-problems-blueprint >}}

---

## 1. The Superposition Hypothesis & Polysemanticity

When researchers inspect individual neurons inside a Transformer's MLP layer ($d_{\text{ff}} = 4d$), they rarely find clean "monosemantic" detectors. Instead, a single neuron fires strongly on **Academic Citations**, **English Dialogues**, **HTTP Requests**, and **Korean Text** simultaneously—a phenomenon known as **polysemanticity**.

Elhage et al. (2022) (*Toy Models of Superposition*) proved that polysemanticity is a deliberate geometric compression strategy governed by two mathematical properties:

1. **The Johnson-Lindenstrauss Lemma (Exponential Almost-Orthogonal Packing):** In a $d$-dimensional Euclidean space $\mathbb{R}^d$, you can fit at most $d$ strictly orthogonal unit vectors ($\mathbf{w}_i^\top \mathbf{w}_j = 0$). However, if you tolerate a tiny interference angle $|\mathbf{w}_i^\top \mathbf{w}_j| \le \epsilon$, the number of **almost-orthogonal** unit vectors $M$ that fit in $\mathbb{R}^d$ grows **exponentially** with dimension $d$:
   $$M \le \exp\!\left(\frac{\epsilon^2 d}{4}\right)$$
2. **Sparsity Enables Non-Linear Interference Cancellation:** Consider the toy autoencoder $\hat{\mathbf{x}} = \operatorname{ReLU}(\mathbf{W}^\top \mathbf{W}\mathbf{x} + \mathbf{b})$ where $\mathbf{W} \in \mathbb{R}^{d \times M}$ ($M > d$). When input features $x_i$ are **dense** ($P(x_i > 0) \approx 1$), off-diagonal dot products $\sum_{j \neq i} (\mathbf{w}_i^\top \mathbf{w}_j) x_j$ corrupt the reconstruction, forcing the network to represent only the top $d$ features orthogonally. When features are **sparse** ($P(x_i > 0) = 1 - S \ll 1$), two features rarely co-occur in the same token, allowing the negative bias $\mathbf{b} < \mathbf{0}$ and $\operatorname{ReLU}$ threshold to filter out the $\epsilon$ cross-talk completely and pack $M \gg d$ features into superposition!

---

## 2. Mechanistic Interpretability via Sparse Autoencoders (SAEs)

If the residual stream $\mathbf{h} \in \mathbb{R}^d$ is a compressed linear combination of $M \gg d$ superposed features $\mathbf{h} \approx \sum_{k=1}^{M} f_k \mathbf{d}_k$, how can we recover the true underlying monosemantic features?

Bricken et al. (2023) and Templeton et al. (2024) (*Scaling Monosemanticity* on Claude 3 Sonnet) train an overcomplete **Sparse Autoencoder (SAE)** with dictionary size $M \in [16d, 64d]$ (up to $M = 34\text{ million}$ features) on frozen residual stream activations $\mathbf{h} \in \mathbb{R}^d$:

$$\mathbf{f} = \operatorname{ReLU}\!\left(\mathbf{W}_{\text{enc}}(\mathbf{h} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}}\right) \in \mathbb{R}_{\ge 0}^M, \qquad \hat{\mathbf{h}} = \mathbf{W}_{\text{dec}}\mathbf{f} + \mathbf{b}_{\text{dec}} = \sum_{k=1}^{M} f_k \mathbf{d}_k + \mathbf{b}_{\text{dec}}$$

where the columns $\mathbf{d}_k$ of $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d \times M}$ are constrained to unit norm $\|\mathbf{d}_k\|_2 = 1$, and the SAE is trained by minimizing reconstruction MSE plus an $L_1$ sparsity penalty (or direct **Top-$K$** activation selection, Gao et al. 2024):

$$\mathcal{L}_{\text{SAE}}(\mathbf{h}) = \|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 + \lambda \|\mathbf{f}\|_1$$

### Causal Feature Steering (Activation Clamping)
Unlike passive probing classifiers, SAE dictionary directions $\mathbf{d}_k \in \mathbb{R}^d$ are **causally active**: adding $\alpha \cdot \mathbf{d}_k$ directly to the residual stream during a forward pass ($\mathbf{h}' \leftarrow \mathbf{h} + \alpha \mathbf{d}_k$) deterministically amplifies or suppresses high-level concepts—such as the famous *Golden Gate Bridge* feature, sycophantic praise, code security vulnerabilities, or deceptive alignment reasoning.

{{< visual/open-problems-interactive >}}

---

## 3. Worked Numerical Walkthrough: 2D Antipodal Superposition & ReLU Filtering

Let us prove mathematically how a $1$-dimensional hidden layer ($d = 1$) can losslessly represent $M = 2$ sparse features $\mathbf{x} = (x_1, x_2)^\top \in \{0, 1\}^2$ that never fire simultaneously ($x_1 x_2 = 0$).

1. **Antipodal Weight Packing:** Set the $1 \times 2$ projection matrix to $\mathbf{W} = [+1, \, -1]$, so feature $1$ points to $+1$ and feature $2$ points to $-1$ ($180^\circ$ antipodal packing). The Gram matrix is:
   $$\mathbf{W}^\top \mathbf{W} = \begin{bmatrix} +1 & -1 \\ -1 & +1 \end{bmatrix}$$
2. **Case 1 ($\mathbf{x} = (1, 0)^\top$, Feature 1 Active):**
   * Compressed scalar state: $h = \mathbf{W}\mathbf{x} = (+1)(1) + (-1)(0) = +1$.
   * Linear readout before ReLU: $\mathbf{W}^\top h = [+1, \, -1]^\top$.
   * Apply elementwise $\operatorname{ReLU}(z) = \max(0, z)$:
     $$\hat{\mathbf{x}} = \operatorname{ReLU}\!\left(\begin{bmatrix} +1 \\ -1 \end{bmatrix}\right) = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \mathbf{x}$$
3. **Case 2 ($\mathbf{x} = (0, 1)^\top$, Feature 2 Active):**
   * Compressed scalar state: $h = \mathbf{W}\mathbf{x} = (+1)(0) + (-1)(1) = -1$.
   * Apply elementwise $\operatorname{ReLU}$ to $\mathbf{W}^\top h = [-1, \, +1]^\top$:
     $$\hat{\mathbf{x}} = \operatorname{ReLU}\!\left(\begin{bmatrix} -1 \\ +1 \end{bmatrix}\right) = \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \mathbf{x}$$

Because $\operatorname{ReLU}$ zeroes out the $-1$ antipodal cross-talk, a **1D scalar channel reconstructs two 1-bit features with zero error** ($\mathcal{L} = 0$) whenever sparsity guarantees at most one feature is active!

---

## 4. Open Theoretical Limits: The Reversal Curse & Circuit Complexity (TC⁰)

Even when a foundation model memorizes facts perfectly and avoids hallucination on forward queries, two fundamental structural limits govern autoregressive Transformers:

### 1. The Reversal Curse (Asymmetric Key-Value Indexing)
Berglund et al. (2024) discovered that if a language model is fine-tuned on the sentence *"Uriah Hawthorne is the composer of Abyssal Melodies"* until $\log P(\text{"Abyssal Melodies"} \mid \text{"Uriah Hawthorne..."}) \approx 0$, and then asked *"Who is the composer of Abyssal Melodies?"*, its accuracy is **no better than random chance ($0\%$)**.

**Mathematical Cause:** As established in our analysis of MLP associative memory ($\mathbf{m} = \mathbf{W}_{\text{proj}}\sigma(\mathbf{W}_{\text{fc}}\mathbf{h})$) and causal self-attention, autoregressive gradient updates are **strictly directional**: training on $A \to B$ updates the weight columns indexed by the hidden representation of prefix $A$ to emit vector $B$, but never presents $B$ as a causal prefix key to retrieve value $A$. The associative weight matrix $\mathbf{W}_{\text{proj}}$ is a **one-way hash table**, not a symmetric relational knowledge graph.

### 2. Formal Expressivity Bounds: TC⁰ vs. NC¹
What problems can a fixed-depth $L$-layer Transformer **never** solve in a single forward pass as input length $T \to \infty$? Merrill and Sabharwal (2023) and Liu et al. (2023) proved that any constant-depth $L$, polynomial-width, log-precision Transformer without Chain-of-Thought belongs to the uniform threshold circuit class **$\mathsf{TC}^0$**:

$$\mathsf{Transformer}\!\left(L = \mathcal{O}(1),\, \text{precision } \mathcal{O}(\log T)\right) \subseteq \mathsf{ uniform-}\mathsf{TC}^0$$

Unless the foundational complexity conjecture $\mathsf{TC}^0 \subsetneq \mathsf{NC}^1$ is false, a constant-depth Transformer **cannot** solve any $\mathsf{NC}^1$-complete problem—including **tracking non-solvable group permutations ($S_5$, such as tracking 5 shuffled cups over $T$ swaps)**, **evaluating arbitrary boolean formulas**, or **simulating finite-state automata**—unless it either:
1. Emits $\mathcal{O}(T)$ serial **Chain-of-Thought tokens** (expanding depth to $\mathcal{O}(T)$), or
2. Incorporates a true non-commutative **recurrent state transition** (such as full-matrix DeltaNet / Mamba with non-negative eigenvalues).

| Frontier Open Problem | Mathematical Root Cause | Failure Manifestation at Scale | Active Research Direction |
| :--- | :--- | :--- | :--- |
| **Polysemantic Superposition** | Packing $M \gg d$ sparse concepts into $\mathbb{R}^d$ via $\epsilon$-orthogonal vectors | Opaque internal circuits; entangled safety & deception features | Overcomplete Top-$K$ Sparse Autoencoders (SAEs) & Transcoders |
| **The Reversal Curse** | Causal loss $\log P(B \mid A)$ updates directional MLP key $A \to$ value $B$ only | $0\%$ zero-shot recall on inverse query $B \to A$ without bidirectional data | Bidirectional knowledge augmentation & entity-swapped synthetic pre-training |
| **$\mathsf{TC}^0$ State-Tracking Limit** | Parallel softmax attention has constant circuit depth $\mathcal{O}(L)$ | Fails on $S_5$ permutation tracking & parity as $T$ exceeds layer count $L$ | Hybrid SSM-Transformer recurrence & test-time CoT depth scaling |
| **Length Generalization** | Out-of-distribution attention entropy growth $\log T$ & unseen RoPE phases | Sudden perplexity collapse beyond training window $T_{\text{train}}$ | Temperature-scaled Yarn attention ($\tau \propto \ln T$) & differential attention |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2209.10652" title="Toy Models of Superposition" subtitle="Elhage et al. (Anthropic 2022) • Proved how neural networks use Johnson-Lindenstrauss almost-orthogonal packing and ReLU thresholds to store M ≫ d sparse features." >}}
  {{< card link="https://transformer-circuits.pub/2024/scaling-monosemanticity/" title="Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" subtitle="Templeton et al. (Anthropic 2024) • Scaled Sparse Autoencoders to 34M dictionary features and demonstrated causal behavioral steering." >}}
  {{< card link="https://arxiv.org/abs/2309.12288" title="The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'" subtitle="Berglund et al. (ICLR 2024) • Demonstrated fundamental directional asymmetry in autoregressive parametric weight updates." >}}
  {{< card link="https://arxiv.org/abs/2305.16960" title="The Parallelism Tradeoff: Limitations of Log-Precision Transformers" subtitle="Merrill & Sabharwal (TACL 2023) • Proved that constant-depth Transformers lie in uniform TC⁰ and cannot solve NC¹-complete state tracking without CoT." >}}
{{< /cards >}}
