---
title: Inference & Training Efficiency
toc: true
type: docs
math: true
weight: 3
---

Modern GPU accelerators are asymmetric machines: their matrix compute throughput (FLOP/s) has scaled $10\times$ faster than their off-chip High-Bandwidth Memory (HBM) speed. An NVIDIA H100 SXM GPU delivers **$989\text{ TFLOP/s}$** of FP16 tensor compute, but only **$3.35\text{ TB/s}$** of HBM3 memory bandwidth.

To saturate the GPU's matrix units without stalling on memory transfers, a kernel must perform at least **295 floating-point operations for every byte fetched from HBM** (the hardware **Ridge Point**):

$$
I_{\text{ridge}} = \frac{\text{Peak Compute (FLOP/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295.2\text{ FLOPs/Byte}
$$

This single ratio explains why LLM inference is divided into two physically opposite regimes:

1. **Prefill (Compute-Bound)**: Processing an input prompt of $T = 2{,}048$ tokens in parallel performs $2 N T$ FLOPs while reading the $2N$ byte weight matrix once, yielding $I_{\text{prefill}} \approx T = 2{,}048 \gg 295\text{ FLOPs/Byte}$.
2. **Autoregressive Decode (Memory-Bandwidth Bound)**: Generating a single token ($T=1$) for batch size $B=1$ performs $2N$ FLOPs while streaming all $2N$ bytes of model weights from HBM, yielding **$I_{\text{decode}} \approx 1\text{ FLOP/Byte}$**—leaving **99.6% of the GPU's compute cores idle**.

{{< visual/efficiency-blueprint >}}

---

## Weight Quantization Walkthrough (AWQ & GPTQ)

Because small-batch decoding is bottlenecked strictly by streaming weight bytes from HBM, compressing each parameter from 16 bits (`FP16`/`BF16`) to 4 bits (`INT4`) reduces memory traffic by nearly $4\times$ and directly multiplies token generation speed.

### 1. Group-Wise Affine Quantization
Given a weight block $\mathbf{w} \in \mathbb{R}^{128}$, 4-bit asymmetric quantization maps floating-point weights to integers $\{0, 1, \dots, 15\}$:

$$
\Delta = \frac{\max(\mathbf{w}) - \min(\mathbf{w})}{2^b - 1}, \qquad z = \operatorname{round}\!\left(\frac{-\min(\mathbf{w})}{\Delta}\right), \qquad \mathbf{w}_q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{\mathbf{w}}{\Delta}\right) + z,\, 0,\, 15\right)
$$

Consider a 4-element weight slice $\mathbf{w} = [-0.60, \, -0.15, \, +0.30, \, +1.20]$ quantized to $b = 4$ bits ($2^4 - 1 = 15$ intervals):
* **Scale Factor $\Delta$**: $\frac{1.20 - (-0.60)}{15} = \frac{1.80}{15} = \mathbf{0.12}$
* **Zero-Point $z$**: $\operatorname{round}(0.60 / 0.12) = \mathbf{5}$
* **Quantized Integers $\mathbf{w}_q$**: $[0, \, 4, \, 8, \, 15]$
* **Dequantized Reconstruction $\hat{\mathbf{w}} = \Delta(\mathbf{w}_q - z)$**: $[-0.60, \, -0.12, \, +0.36, \, +1.20]$

### 2. Activation-Aware Weight Quantization (AWQ)
In Transformer hidden states, $0.1\%\text{--}1\%$ of channels exhibit massive activation outliers ($100\times$ larger than average). Quantizing the corresponding weight columns naively causes severe perplexity degradation.

**AWQ** (Lin et al. 2024) protects salient weight channels without mixed-precision hardware overhead by multiplying salient weight column $\mathbf{w}_j$ by a per-channel scale $s_j > 1$ before quantization and dividing the input activation channel $\mathbf{x}_j$ by $s_j$:

$$
\mathbf{y} = (\mathbf{w}_j \cdot s_j) \left(\frac{\mathbf{x}_j}{s_j}\right) \implies \operatorname{Quant}(\mathbf{w}_j \cdot s_j) \cdot \frac{\mathbf{x}_j}{s_j}
$$

---

## Speculative Decoding Math & Expected Speedup

How can we accelerate autoregressive decoding without modifying model weights? **Speculative Decoding** (Leviathan et al. 2023; Chen et al. 2023) uses a fast draft model $q(x_t \mid x_{<t})$ to speculate $K$ tokens ahead, then uses a single parallel forward pass of the target model $p(x_t \mid x_{<t})$ to verify all $K$ tokens simultaneously.

To guarantee that the output is **mathematically identical** to sampling from the target model $p(x)$, each drafted token $x$ is accepted with probability:

$$
\alpha(x) = \min\!\left(1, \, \frac{p(x)}{q(x)}\right)
$$

If token $x_k$ is rejected, we immediately sample a replacement token from the residual distribution without running another forward pass:

$$
x'_k \sim \operatorname{norm}\!\left(\max\left(0, \, p(x) - q(x)\right)\right)
$$

If the average acceptance rate across drafted tokens is $\alpha \in [0, 1]$, the expected number of tokens generated per target forward pass is:

$$
\mathbb{E}[\text{Tokens / Pass}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

For $K = 4$ and $\alpha = 0.80$, a single target verification pass emits $\frac{1 - 0.80^5}{1 - 0.80} = \mathbf{3.36\text{ tokens}}$, delivering a $2.5\times$ wall-clock speedup with zero accuracy loss.

---

## Interactive Roofline & Quantization Simulator

Explore the H100 roofline model below for a 70B parameter model. Notice how weight quantization shifts the operating point along the memory-bandwidth slope at small batch sizes:

{{< visual/efficiency-interactive >}}

---

## Distributed Training Memory: The 16Ψ Rule & ZeRO

Training a model with $\Psi$ parameters in mixed precision (`BF16` weights/gradients + `FP32` AdamW master states) requires **16 bytes per parameter** before activations:
* **Parameters (`BF16`)**: $2\Psi\text{ bytes}$
* **Gradients (`BF16`)**: $2\Psi\text{ bytes}$
* **AdamW Master Weights, Momentum $m_t$, and Variance $v_t$ (`FP32`)**: $4\Psi + 4\Psi + 4\Psi = 12\Psi\text{ bytes}$

For a 70B model ($\Psi = 7 \times 10^{10}$), static states alone require $16 \times 70\text{B} = \mathbf{1{,}120\text{ GB}}$—exceeding a single 80 GB GPU by $14\times$. **ZeRO / FSDP** (Rajbhandari et al. 2020) eliminates replication across $N_{\text{GPU}}$ data-parallel ranks:

| Sharding Stage | Partitioned States Across $N_{\text{GPU}}$ | Per-GPU Memory Formula | 70B Footprint ($N_{\text{GPU}} = 64$) | Communication Volume / Step |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline DDP** | None (Full replication) | $16\Psi$ | $1{,}120.0\text{ GB}$ (OOM) | $2\Psi$ (`AllReduce` gradients) |
| **ZeRO-1** | Optimizer states ($12\Psi$) | $4\Psi + \frac{12\Psi}{N_{\text{GPU}}}$ | $293.1\text{ GB}$ (OOM) | $2\Psi$ (`ReduceScatter` + `AllGather`) |
| **ZeRO-2** | Optimizer + Gradients ($14\Psi$) | $2\Psi + \frac{14\Psi}{N_{\text{GPU}}}$ | $155.3\text{ GB}$ (OOM) | $2\Psi$ (`ReduceScatter` + `AllGather`) |
| **ZeRO-3 / FSDP** | Optimizer + Grads + Params ($16\Psi$) | $\frac{16\Psi}{N_{\text{GPU}}}$ | **$17.5\text{ GB}$ (Fits easily)** | $3\Psi$ ($1.5\times$ baseline for forward/backward `AllGather`) |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1910.02054" title="ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" subtitle="Rajbhandari et al. (SC 2020) • Introduced optimizer, gradient, and parameter sharding across data-parallel ranks." >}}
  {{< card link="https://arxiv.org/abs/2211.17192" title="Fast Inference from Transformers via Speculative Decoding" subtitle="Leviathan et al. (ICML 2023) • Proved lossless parallel rejection sampling for accelerating memory-bound autoregressive decoding." >}}
  {{< card link="https://arxiv.org/abs/2306.00978" title="AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" subtitle="Lin et al. (MLSys 2024) • Demonstrated that scaling 1% salient activation channels preserves INT4 accuracy without mixed precision." >}}
  {{< card link="https://arxiv.org/abs/2309.06180" title="Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)" subtitle="Kwon et al. (SOSP 2023) • Eliminated KV-cache fragmentation via virtual memory paging, multiplying serving batch size." >}}
{{< /cards >}}
