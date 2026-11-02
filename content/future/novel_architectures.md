---
title: Sub-Quadratic & Hybrid Architectures
toc: true
type: docs
math: true
weight: 1
---

The self-attention operator $\operatorname{Softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V}$ is the computational engine of modern foundation models, yet it carries a fundamental structural flaw: because every token attends to every preceding token explicitly, prefill compute scales quadratically as $\mathcal{O}(T^2 d)$ and the autoregressive KV-cache grows linearly as $\mathcal{O}(T d)$ without bound.

Sub-quadratic sequence architectures—led by **Structured State Space Models (S4)**, **Selective State Spaces (Mamba-1 & Mamba-2)**, **Gated Linear Attention**, and **Hybrid SSM-Transformer topologies (Jamba, Zamba)**—replace the expanding key-value history with a fixed-dimensional recurrent state $\mathbf{h}_t \in \mathbb{R}^{d \times N}$ that compresses arbitrary sequence lengths into $\mathcal{O}(T)$ training time and $\mathcal{O}(1)$ inference memory.

{{< visual/novel-arch-blueprint >}}

---

## 1. Continuous-to-Discrete State Space Foundations

A classical Linear Time-Invariant (LTI) State Space Model maps a 1D continuous scalar input signal $x(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ via an $N$-dimensional latent state vector $\mathbf{h}(t) \in \mathbb{R}^N$:

$$\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), \qquad y(t) = \mathbf{C}\mathbf{h}(t)$$

where $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix initialized via **HiPPO** (High-Order Polynomial Projection Operators) to compress continuous history into orthogonal Legendre polynomials, and $\mathbf{B} \in \mathbb{R}^{N \times 1}, \mathbf{C} \in \mathbb{R}^{1 \times N}$ are projection vectors.

### Zero-Order Hold (ZOH) Discretization

To apply this continuous differential equation to a discrete token sequence $(x_1, x_2, \dots, x_T)$, we hold the input value $x_t$ constant over a continuous step size $\Delta > 0$. Integrating the linear ODE over $[t, t + \Delta]$ yields the exact **Zero-Order Hold (ZOH)** discrete parameters $(\overline{\mathbf{A}}, \overline{\mathbf{B}})$:

$$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \qquad \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1}\left(\exp(\Delta \mathbf{A}) - \mathbf{I}\right) \cdot \Delta \mathbf{B} \approx \Delta \mathbf{B}$$

where the first-order Euler approximation $\overline{\mathbf{B}} \approx \Delta \mathbf{B}$ holds when $\Delta$ is small. The discrete system can then be evaluated in two dual modes:

1. **Linear Recurrence ($\mathcal{O}(1)$ Autoregressive Decode):**
   $$\mathbf{h}_t = \overline{\mathbf{A}}\mathbf{h}_{t-1} + \overline{\mathbf{B}}x_t, \qquad y_t = \mathbf{C}\mathbf{h}_t$$
2. **Global Convolution ($\mathcal{O}(T \log T)$ Parallel Training for LTI S4):** Because $\overline{\mathbf{A}}, \overline{\mathbf{B}}, \mathbf{C}$ are constant across time $t$, unrolling the recurrence yields a 1D causal convolution $\mathbf{y} = \overline{\mathbf{K}} * \mathbf{x}$ with kernel $\overline{\mathbf{K}} = (\mathbf{C}\overline{\mathbf{B}}, \mathbf{C}\overline{\mathbf{A}}\overline{\mathbf{B}}, \dots, \mathbf{C}\overline{\mathbf{A}}^{T-1}\overline{\mathbf{B}})$.

---

## 2. Selective State Spaces (Mamba) & Associative Scans

While LTI models (S4, H3) achieve $\mathcal{O}(T)$ scaling, their time-invariant matrices $(\overline{\mathbf{A}}, \overline{\mathbf{B}}, \mathbf{C})$ cannot perform **content-based reasoning** (such as selective copying or induction heads): the transition weights are identical regardless of whether token $x_t$ is a critical entity name or an irrelevant punctuation mark.

Gu and Dao (2023) solved this limitation in **Mamba** by making the step size $\Delta_t$ and projection matrices $\mathbf{B}_t, \mathbf{C}_t$ explicit functions of the input token $\mathbf{x}_t \in \mathbb{R}^d$:

$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t, \qquad \mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t, \qquad \Delta_t = \operatorname{softplus}\!\left(\mathbf{W}_\Delta \mathbf{x}_t + \mathbf{b}_\Delta\right)$$

### The Information-Theoretic Meaning of Δₜ

When we restrict $\mathbf{A} = -\operatorname{diag}(a_1, \dots, a_N)$ with $a_n > 0$, the selective ZOH recurrence becomes an adaptive continuous-time **gating mechanism** analogous to an LSTM forget/input gate:

* **$\Delta_t \to \infty$ (Salient Token Focus):** $\overline{\mathbf{A}}_t = \exp(-\Delta_t a_n) \to 0$ and $\overline{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$ grows large. The model flushes its old recurrent state $\mathbf{h}_{t-1}$ and writes the current token $\mathbf{x}_t$ into memory.
* **$\Delta_t \to 0$ (Noise / Filler Bypass):** $\overline{\mathbf{A}}_t = \exp(-\Delta_t a_n) \to 1$ and $\overline{\mathbf{B}}_t \to 0$. The model ignores the current token $x_t$ completely and persists its prior memory state $\mathbf{h}_{t-1}$ unchanged across arbitrary distractors.

Because $\overline{\mathbf{A}}_t$ now varies with $t$, the FFT convolution trick no longer applies. Instead, Mamba fuses the discretization and recurrence inside GPU SRAM using a work-efficient **Blelloch Parallel Associative Scan**. Because the first-order affine composition $(\overline{\mathbf{A}}_t, \overline{\mathbf{B}}_t x_t) \bullet (\overline{\mathbf{A}}_{t-1}, \overline{\mathbf{B}}_{t-1} x_{t-1}) = (\overline{\mathbf{A}}_t \overline{\mathbf{A}}_{t-1}, \overline{\mathbf{A}}_t \overline{\mathbf{B}}_{t-1} x_{t-1} + \overline{\mathbf{B}}_t x_t)$ is strictly associative, all $T$ recurrent states are computed in $\mathcal{O}(\log T)$ parallel tree depth without materializing intermediate state tensors in HBM.

---

## 3. Worked Numerical Walkthrough: Selective ZOH Step & Associative Scan

Let us trace a $1$-dimensional selective state space coordinate with continuous decay parameter $A = -1.0$ over a $3$-token sequence $x = (x_1, x_2, x_3) = (2.0, \, 0.5, \, 4.0)$ starting from $h_0 = 0$.

1. **Token 1 ($x_1 = 2.0$, Salient Entity):** Suppose the selection network outputs $\Delta_1 = 1.20$, $B_1 = 1.0$, $C_1 = 1.0$.
   $$\overline{A}_1 = \exp(-1.20) \approx 0.301, \qquad \overline{B}_1 x_1 = (1.20)(1.0)(2.0) = 2.40$$
   $$h_1 = 0.301(0) + 2.40 = \mathbf{2.40}$$

2. **Token 2 ($x_2 = 0.5$, Irrelevant Filler):** The selection network suppresses the step size to $\Delta_2 = 0.05$, $B_2 = 1.0$.
   $$\overline{A}_2 = \exp(-0.05) \approx 0.951, \qquad \overline{B}_2 x_2 = (0.05)(1.0)(0.5) = 0.025$$
   $$h_2 = 0.951(2.40) + 0.025 = 2.282 + 0.025 = \mathbf{2.307}$$
   Notice that $95.1\%$ of Token 1's state survives across the distractor token.

3. **Token 3 ($x_3 = 4.0$, New Salient Fact):** The selection network spikes $\Delta_3 = 2.00$, $B_3 = 1.0$.
   $$\overline{A}_3 = \exp(-2.00) \approx 0.135, \qquad \overline{B}_3 x_3 = (2.00)(1.0)(4.0) = 8.00$$
   $$h_3 = 0.135(2.307) + 8.00 = 0.311 + 8.00 = \mathbf{8.311}$$

---

## 4. Structured State Space Duality (Mamba-2 SSD) & Hybrid Architectures

In **Mamba-2**, Dao and Gu (2024) proved **Structured State Space Duality (SSD)**: when the diagonal matrix $\mathbf{A}_t = a_t \mathbf{I}$ is simplified to a scalar identity multiplier $a_t \in [0, 1]$ per head, unrolling the selective SSM recurrence $\mathbf{y} = \operatorname{SSM}(\mathbf{x})$ is mathematically identical to multiplying the input by a **$1$-semiseparable causal mask $\mathbf{L}$** in linear attention:

$$\mathbf{y} = \left(\mathbf{L} \odot \left(\mathbf{C}\mathbf{B}^\top\right)\right)\mathbf{x}, \qquad \mathbf{L}_{ij} = \begin{cases} a_i a_{i-1} \cdots a_{j+1} & \text{if } i \ge j \\ 0 & \text{if } i < j \end{cases}$$

By partitioning the sequence into blocks of size $Q = 64$, SSD computes intra-block transitions using dense **Tensor Core matrix multiplications** ($\mathbf{C}\mathbf{B}^\top$, exactly like FlashAttention) while passing the recurrent state between blocks in $\mathcal{O}(T)$ time—achieving $2\text{--}8\times$ higher training throughput than Mamba-1's elementwise scan.

{{< visual/novel-arch-interactive >}}

### Why Hybrid Architectures (Jamba / Zamba) Win in Practice

Despite their $\mathcal{O}(1)$ memory efficiency, pure fixed-state models face a fundamental information-theoretic bottleneck on **exact in-context copying and multi-needle associative recall**: a fixed state $\mathbf{h}_t \in \mathbb{R}^{d \times N}$ of $2\text{ MB}$ cannot losslessly store $500\text{K}$ arbitrary random key-value pairs.

Frontier hybrid architectures (AI21 **Jamba**, Zyphra **Zamba**, NVIDIA **Nemotron-H**) resolve this by interleaving **1 full GQA Self-Attention layer for every 7 State Space (Mamba-2) layers** ($1:8$ ratio):

| Architectural Family | Prefill Compute | State / KV Memory at $T=256\text{K}$ | Exact Multi-Needle Recall | Representative Models |
| :--- | :--- | :--- | :--- | :--- |
| **Dense GQA Transformer** | $\mathcal{O}(T^2 d)$ | $67.1\text{ GB}$ (100% KV Cache) | **Exact ($99.8\%$)** | LLaMA-3, Qwen-2.5 |
| **Pure Selective SSM** | $\mathcal{O}(T d N)$ | **$0.13\text{ GB}$ ($\mathcal{O}(1)$ Fixed)** | Degrades on dense random keys | Mamba-1, Mamba-2 |
| **Gated Linear / RNN** | $\mathcal{O}(T d^2)$ | **$0.26\text{ GB}$ ($\mathcal{O}(1)$ Matrix)** | Moderate ($82\text{--}90\%$) | RWKV-6, RetNet, GLA |
| **Hybrid SSM-Attention (1:8)** | $\mathcal{O}\!\left(\frac{1}{8}T^2 d + \frac{7}{8}T d N\right)$ | **$8.5\text{ GB}$ ($8\times$ smaller)** | **Exact ($99.2\%$)** | Jamba-1.5, Zamba-2, Falcon-Mamba |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2111.00396" title="Efficiently Modeling Long Sequences with Structured State Spaces (S4)" subtitle="Gu et al. (ICLR 2022) • Introduced HiPPO orthogonal state initialization and Cauchy kernel diagonalization for continuous-time SSMs." >}}
  {{< card link="https://arxiv.org/abs/2312.00752" title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces" subtitle="Gu & Dao (COLM 2024) • Replaced LTI convolutions with input-dependent ZOH selection gates and SRAM-fused parallel associative scans." >}}
  {{< card link="https://arxiv.org/abs/2405.21060" title="Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)" subtitle="Dao & Gu (ICML 2024) • Proved exact duality between selective SSMs and semiseparable masked linear attention for Tensor Core acceleration." >}}
  {{< card link="https://arxiv.org/abs/2403.19887" title="Jamba: A Hybrid Transformer-Mamba Language Model" subtitle="Lieber et al. (2024) • Demonstrated that interleaving 1 Attention layer per 7 Mamba layers achieves 8× KV cache reduction with full recall." >}}
{{< /cards >}}
