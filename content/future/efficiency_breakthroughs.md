---
title: Next-Gen Systems & Extreme Compression
toc: true
type: docs
math: true
weight: 4
---

Standard post-training optimizations—such as 4-bit weight quantization (AWQ/GPTQ) and Grouped-Query Attention (GQA)—treat the pre-trained Transformer architecture as fixed and compress weights or KV states after the fact. The next generation of efficiency breakthroughs redesigns the core mathematical operators **from scratch during pre-training** so that low-rank KV compression, ternary multiplier-free arithmetic, dynamic layer skipping, and multi-token decoding are native properties of the network.

{{< visual/efficiency-breakthroughs-blueprint >}}

---

## 1. Multi-Head Latent Attention (MLA — DeepSeek-V2 / V3)

In Grouped-Query Attention (GQA), reducing the KV-cache footprint requires forcing $G$ query heads to share a single key and value head ($n_{kv} = n_h / G$), which degrades representation capacity when $n_{kv}$ becomes small. **Multi-Head Latent Attention (MLA)** (DeepSeek-AI, 2024) achieves even smaller KV caches than GQA while preserving **distinct keys and values for every single attention head**.

### Low-Rank Joint KV Compression & Matrix Absorption
Given hidden state $\mathbf{h}_t \in \mathbb{R}^d$, MLA projects $\mathbf{h}_t$ down into a shared low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ (where $d_c = 512 \ll d_h n_h = 16{,}384$), and reconstructs full multi-head keys and values via up-projection matrices $\mathbf{W}^{UK}, \mathbf{W}^{UV} \in \mathbb{R}^{(d_h n_h) \times d_c}$:

$$\mathbf{c}_t^{KV} = \mathbf{W}^{DKV} \mathbf{h}_t, \qquad \mathbf{k}_t^C = \mathbf{W}^{UK} \mathbf{c}_t^{KV}, \qquad \mathbf{v}_t^C = \mathbf{W}^{UV} \mathbf{c}_t^{KV}$$

If the model had to materialize $\mathbf{k}_t^C$ and $\mathbf{v}_t^C$ in HBM during inference, the memory bandwidth savings would vanish. However, because matrix multiplication is associative, **$\mathbf{W}^{UK}$ and $\mathbf{W}^{UV}$ are absorbed directly into the query and output projection matrices**:

$$\mathbf{q}_t^\top \mathbf{k}_j^C = \left(\mathbf{W}^Q \mathbf{h}_t\right)^\top \left(\mathbf{W}^{UK} \mathbf{c}_j^{KV}\right) = \left(\left(\mathbf{W}^{UK}\right)^\top \mathbf{W}^Q \mathbf{h}_t\right)^\top \mathbf{c}_j^{KV}$$

During decoding, we only store $\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$ in the KV cache! Why do we still need a small decoupled key $\mathbf{k}_t^R \in \mathbb{R}^{d_R}$ ($d_R = 64$)? Because **Rotary Position Embedding (RoPE)** injects a position-dependent rotation matrix $\mathbf{R}_{t-j}$ between $\mathbf{W}^Q$ and $\mathbf{W}^{UK}$, preventing $\mathbf{W}^{UK}$ from being statically merged into $\mathbf{W}^Q$ across varying relative distances $t - j$. MLA solves this by concatenating a single shared $64$-dimensional RoPE key $\mathbf{k}_t^R = \operatorname{RoPE}(\mathbf{W}^{KR}\mathbf{h}_t)$ alongside the $512$-dimensional position-independent latent $\mathbf{c}_t^{KV}$, caching just $d_c + d_R = 576$ elements per token.

{{< visual/efficiency-breakthroughs-interactive >}}

---

## 2. 1.58-Bit Native Ternary Pre-Training (BitNet b1.58)

Post-training quantization (PTQ) degrades rapidly below 3 bits per weight because rounding a pre-trained Gaussian weight matrix destroys outlier pathways. **BitNet b1.58** (Ma et al. 2024) trains the Transformer from initialization using **Quantization-Aware Training (QAT)** where every linear projection weight $\widetilde{W}_{ij}$ is constrained to the ternary set $\{-1, 0, +1\}$ ($\log_2(3) \approx 1.58\text{ bits}$):

$$\widetilde{\mathbf{W}} = \operatorname{RoundClip}\!\left(\frac{\mathbf{W}}{\gamma + \epsilon},\, -1,\, +1\right), \qquad \gamma = \frac{1}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} |W_{ij}|$$

where $\operatorname{RoundClip}(x, a, b) = \max(a, \min(b, \operatorname{round}(x)))$ is an `absmean` quantizer differentiated via the Straight-Through Estimator, paired with per-token `absmax` 8-bit integer activation quantization:

$$\tilde{\mathbf{x}} = \operatorname{Clip}\!\left(\mathbf{x} \cdot \frac{Q_b}{\|\mathbf{x}\|_\infty},\, -Q_b + \epsilon,\, Q_b - \epsilon\right), \qquad Q_b = 2^{b-1} = 128$$

### Multiplier-Free Matrix Multiplication
In standard `FP16` hardware ($45\text{nm}\text{--}5\text{nm}$ CMOS), a floating-point multiplication consumes **$4\times\text{--}30\times$ more silicon area and dynamic energy** than an integer addition. Because $\widetilde{W}_{ij} \in \{-1, 0, +1\}$, the matrix-vector product $\mathbf{y} = \widetilde{\mathbf{W}}\tilde{\mathbf{x}}$ eliminates floating-point multipliers completely, reducing to conditional `INT8` accumulation:

$$y_i = \gamma \cdot \frac{\|\mathbf{x}\|_\infty}{128} \left(\sum_{j : \widetilde{W}_{ij} = +1} \tilde{x}_j - \sum_{j : \widetilde{W}_{ij} = -1} \tilde{x}_j\right)$$

---

## 3. Worked Numerical Walkthrough: BitNet Absmean Ternary Quantization & MLA Cache

### Part A: BitNet b1.58 Ternary Matrix Quantization
Consider a $2 \times 3$ continuous weight matrix $\mathbf{W}$ during a BitNet forward pass:

$$\mathbf{W} = \begin{bmatrix} +0.84 & -0.12 & -0.66 \\ +0.09 & +0.75 & -0.54 \end{bmatrix}$$

1. **Compute Absmean Scale $\gamma$:**
   $$\gamma = \frac{0.84 + 0.12 + 0.66 + 0.09 + 0.75 + 0.54}{6} = \frac{3.00}{6} = \mathbf{0.50}$$
2. **Normalize by $\gamma = 0.50$:**
   $$\frac{\mathbf{W}}{\gamma} = \begin{bmatrix} +1.68 & -0.24 & -1.32 \\ +0.18 & +1.50 & -1.08 \end{bmatrix}$$
3. **Apply $\operatorname{RoundClip}(\cdot, -1, +1)$:**
   $$\widetilde{\mathbf{W}} = \begin{bmatrix} +1 & 0 & -1 \\ 0 & +1 & -1 \end{bmatrix}$$
Given an input vector $\tilde{\mathbf{x}} = [10, \, 25, \, 40]^\top$, computing $\widetilde{\mathbf{W}}\tilde{\mathbf{x}}$ requires **zero multiplications**: Row 1 computes $10 - 40 = -30$, and Row 2 computes $25 - 40 = -15$.

### Part B: MLA vs. MHA vs. GQA Per-Token Cache Ratio
For a model with $n_h = 128$ heads of dimension $d_h = 128$:
* **Standard MHA:** $2 \times 128 \times 128 = 32{,}768$ elements/token/layer.
* **GQA ($n_{kv} = 8$):** $2 \times 8 \times 128 = 2{,}048$ elements/token/layer.
* **MLA ($d_c = 512, d_R = 64$):** $512 + 64 = 576$ elements/token/layer—achieving a **$56.9\times$ reduction vs. MHA** and a **$3.56\times$ reduction vs. GQA-8**.

---

## 4. Dynamic Compute Routing: Mixture-of-Depths & Multi-Token Prediction

### 1. Mixture-of-Depths (MoD — Token-Level Layer Skipping)
Just as Mixture-of-Experts (MoE) routes tokens across different FFN experts, **Mixture-of-Depths (MoD)** (Raposo et al. 2024) observes that predicting a simple syntactic token (`"the"`, `")"`) does not require passing through all $L$ Attention and MLP blocks. At layer $l$, a lightweight scalar router $r_i = \mathbf{w}_l^\top \mathbf{x}_l^i$ scores all $T$ tokens in the sequence and selects only the **top-$k$** tokens ($k = C \cdot T$, e.g., capacity $C = 12.5\%$ or $50\%$ with threshold $P_\beta$) to enter the Self-Attention and MLP block $\tilde{f}_l$, while the remaining $T - k$ tokens bypass the layer completely via the residual stream:

$$\mathbf{x}_l^{i+1} = \begin{cases} r_i \tilde{f}_l(\mathbf{X}_l^{\text{top-}k})_i + \mathbf{x}_l^i & \text{if } r_i > P_\beta(\mathbf{R}_l) \\ \mathbf{x}_l^i & \text{if } r_i \le P_\beta(\mathbf{R}_l) \end{cases}$$

Because only $k = C \cdot T$ tokens participate in self-attention at layer $l$, attention FLOPs drop quadratically to $C^2 \cdot \mathcal{O}(T^2)$.

### 2. Multi-Token Prediction (MTP & Medusa Tree Decoding)
Instead of relying on a separate external draft model for speculative decoding, **Multi-Token Prediction (MTP)** (Gloeckle et al. 2024; DeepSeek-V3) and **Medusa** (Cai et al. 2024) attach $K$ lightweight sequential or parallel prediction heads directly onto the top of the target model's trunk $\mathbf{h}_t^{(L)}$:

$$\mathcal{L}_{\text{MTP}} = -\sum_{t=1}^{T}\sum_{k=1}^{K} \lambda_k \log P_\theta^{(k)}\!\left(x_{t+k} \mid x_{\le t}\right)$$

During training, predicting $K$ tokens ahead forces the hidden representation $\mathbf{h}_t^{(L)}$ to plan multi-token phrasal structure rather than myopic next-token statistics; during inference, the auxiliary MTP heads act as a **self-drafting speculative decoder**, verifying $2\text{--}4$ tokens per forward pass with zero external draft model management.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2405.04434" title="DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)" subtitle="DeepSeek-AI (2024) • Introduced Multi-Head Latent Attention (MLA) with low-rank joint KV compression and matrix absorption." >}}
  {{< card link="https://arxiv.org/abs/2402.17764" title="The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)" subtitle="Ma et al. (2024) • Demonstrated that native ternary {-1, 0, +1} pre-training matches FP16 perplexity at 3B+ scale with adder-only matmuls." >}}
  {{< card link="https://arxiv.org/abs/2404.02258" title="Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models" subtitle="Raposo et al. (2024) • Applied top-k token routing across residual layers to slash attention and FFN FLOPs by up to 50%." >}}
  {{< card link="https://arxiv.org/abs/2404.19737" title="Better & Faster Large Language Models via Multi-token Prediction" subtitle="Gloeckle et al. (ICML 2024) • Showed that training auxiliary future-token heads improves both algorithmic reasoning and self-speculative decoding." >}}
{{< /cards >}}
