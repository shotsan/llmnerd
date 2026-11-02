---
title: Native Any-to-Any Multimodality
toc: true
type: docs
math: true
weight: 3
---

Early multimodal models were built by bolting frozen, pre-trained unimodal encoders (such as a CLIP Vision Transformer or Whisper audio encoder) onto a frozen text LLM via a shallow linear adapter. While effective for basic visual question answering, late-fusion adapters are fundamentally **one-way**: they project images and audio into the LLM's input embedding space, but the model cannot natively synthesize images, interleaved diagrams, or expressive speech.

Frontier multimodal architectures are converging on **Native Any-to-Any Sequence Modeling** (GPT-4o, Gemini, Meta **Chameleon**, **Transfusion**), where text, images, video, and audio are tokenized into a single unified sequence and processed jointly across all layers of a single foundation backbone.

{{< visual/multimodal-blueprint >}}

---

## 1. Visual Encoding, SigLIP, & 2D Pixel-Shuffle Compression

Before an image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ enters a Transformer, it is partitioned into a grid of non-overlapping $P \times P$ patches (typically $P = 14$ or $P = 16$), yielding $N_{\text{patches}}$ visual tokens:

$$N_{\text{patches}} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$

### From Softmax CLIP to Sigmoid Contrastive Loss (SigLIP)
Classic CLIP (Radford et al. 2021) aligns image embeddings $\mathbf{u}_i$ and text embeddings $\mathbf{v}_j$ using a symmetric cross-entropy loss over an entire global batch $B \times B$, which requires expensive `AllGather` synchronization across thousands of GPUs and suffers from numerical instability at small batch sizes.

**SigLIP** (Zhai et al. 2023)—the vision backbone of PaliGemma, InternVL, and Qwen2-VL—replaces global softmax normalization with an independent **pairwise binary sigmoid loss** over every pair $(i, j)$ with label $y_{ij} = +1$ for matched pairs and $y_{ij} = -1$ for negatives:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B} \log \frac{1}{1 + \exp\!\left(-y_{ij}\left(t \cdot \mathbf{u}_i^\top \mathbf{v}_j + b\right)\right)}$$

where $t = \exp(t')$ is a learnable temperature and $b$ is a learnable negative-pair bias initialized to $b = -10$ so the initial loss starts near zero despite the $B(B-1)$ negative pairs outnumbering the $B$ positive pairs.

### 2D Pixel-Shuffle & Multimodal RoPE (M-RoPE)
At high document resolutions ($H = W = 896$, $P = 14$), a single image produces $64 \times 64 = 4{,}096$ visual tokens. To prevent quadratic $\mathcal{O}(N^2)$ attention blowup inside the LLM, modern vision-language models apply a **2D Pixel-Shuffle** (spatial-to-channel space-to-depth) transformation with factor $s = 2$:

$$\operatorname{PixelShuffle}_s : \mathbb{R}^{(H/P) \times (W/P) \times d_v} \longrightarrow \mathbb{R}^{\left(\frac{H}{sP}\right) \times \left(\frac{W}{sP}\right) \times (s^2 d_v)} \xrightarrow{\mathbf{W}_{\text{proj}}} \mathbb{R}^{\left(\frac{N_{\text{patches}}}{s^2}\right) \times d_{\text{model}}}$$

By folding each $2 \times 2$ block of adjacent visual tokens into the channel dimension before the 2-layer MLP projector, the sequence length drops by $s^2 = 4\times$ and attention FLOPs drop by $s^4 = 16\times$. To preserve 2D spatial geometry across dynamic aspect ratios, **Multimodal RoPE (M-RoPE, Qwen2-VL)** partitions the rotary frequency vector $\boldsymbol{\theta} \in \mathbb{R}^{d_k/2}$ into three orthogonal coordinate slices—$(t_{\text{temporal}}, h_{\text{height}}, w_{\text{width}})$—which reduce to identical 1D positions $(t, t, t)$ for pure text tokens.

{{< visual/multimodal-interactive >}}

---

## 2. Discrete Token Unification: VQGAN & Residual Vector Quantization

How can a standard autoregressive language model *generate* images and audio alongside text? The **Discrete Codebook Paradigm** (**Chameleon**, Team 2024; **SoundStream / EnCodec**) converts continuous visual and acoustic signals into discrete integer indices from a finite codebook $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_{|\mathcal{C}|}\} \subset \mathbb{R}^{d_c}$ so that every modality shares the exact same cross-entropy objective $\mathcal{L}_{\text{CE}} = -\sum_t \log P_\theta(x_t \mid x_{<t})$.

### Vector Quantization & Straight-Through Estimator (STE)
Given a continuous encoder feature vector $\mathbf{z}_e(x) \in \mathbb{R}^{d_c}$, a Vector Quantizer (VQ-VAE / VQGAN) snaps $\mathbf{z}_e(x)$ to its nearest Euclidean neighbor in the codebook:

$$q(\mathbf{z}_e(x)) = \mathbf{e}_k, \qquad \text{where } k = \arg\min_{j \in \{1,\dots,|\mathcal{C}|\}} \|\mathbf{z}_e(x) - \mathbf{e}_j\|_2$$

Because $\arg\min$ has zero gradient everywhere, the **Straight-Through Estimator (STE)** copies gradients directly from decoder input $\mathbf{z}_q$ to encoder output $\mathbf{z}_e$ during backpropagation while training the codebook via commitment loss:

$$\mathcal{L}_{\text{VQ}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \|\operatorname{sg}[\mathbf{z}_e(x)] - \mathbf{e}_k\|_2^2 + \beta_{\text{commit}}\|\mathbf{z}_e(x) - \operatorname{sg}[\mathbf{e}_k]\|_2^2$$

where $\operatorname{sg}[\cdot]$ denotes the stop-gradient operator. For high-fidelity audio ($24\text{ kHz}$ speech in GPT-4o / Moshi), **Residual Vector Quantization (RVQ)** cascades $M = 8$ codebooks sequentially, where codebook $m$ quantizes the residual error $\mathbf{r}_m = \mathbf{r}_{m-1} - \mathbf{e}_{k_{m-1}}^{(m-1)}$ left over by codebook $m-1$.

### Why Early-Fusion Requires QK-LayerNorm
When Chameleon trained a $34\text{B}$ early-fusion Transformer over interleaved text and image VQ tokens, training diverged at scale due to **cross-modal logit drift**: because image tokens have much higher local entropy and different norm statistics than text tokens, competition inside the shared softmax caused attention logits $\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}$ to grow unboundedly in BF16. Applying **QK-LayerNorm** ($\mathbf{q}' = \operatorname{LN}(\mathbf{q}), \mathbf{k}' = \operatorname{LN}(\mathbf{k})$) before the attention dot product bounds the maximum attention logit and stabilizes early-fusion pre-training.

---

## 3. Worked Numerical Walkthrough: 2D Pixel-Shuffle & VQ Quantization

### Part A: High-Resolution Token & FLOP Audit
Suppose we feed a $1344 \times 896$ document page into a vision-language model with patch size $P = 14$:
1. **Raw ViT Grid:** $\frac{1344}{14} \times \frac{896}{14} = 96 \times 64 = \mathbf{6{,}144\text{ patches}}$.
2. **After $2 \times 2$ Pixel-Shuffle ($s = 2$):** Each $2 \times 2$ spatial block is concatenated along the channel axis ($4 \times d_v$) and projected to $d_{\text{model}}$, yielding $\frac{96}{2} \times \frac{64}{2} = 48 \times 32 = \mathbf{1{,}536\text{ visual tokens}}$.
3. **Self-Attention Prefill Speedup:** Because causal self-attention scales as $N^2$, compressing $6{,}144 \to 1{,}536$ tokens reduces visual prefill attention FLOPs by $(6144 / 1536)^2 = \mathbf{16.0\times}$.

### Part B: 2-Stage Residual Vector Quantization (RVQ)
Suppose an audio encoder outputs a $2\text{D}$ feature vector $\mathbf{z}_e = (2.70, \, -1.30)$.
* **Stage 1 Codebook:** Nearest neighbor in $\mathcal{C}^{(1)}$ is $\mathbf{e}_{k_1}^{(1)} = (2.50, \, -1.00)$ (squared error $= 0.04 + 0.09 = 0.13$).
* **Residual Vector:** $\mathbf{r}_1 = \mathbf{z}_e - \mathbf{e}_{k_1}^{(1)} = (2.70 - 2.50, \, -1.30 - (-1.00)) = (+0.20, \, -0.30)$.
* **Stage 2 Codebook:** Nearest neighbor in $\mathcal{C}^{(2)}$ to $\mathbf{r}_1$ is $\mathbf{e}_{k_2}^{(2)} = (+0.18, \, -0.28)$.
* **Two-Codebook Reconstruction:** $\hat{\mathbf{z}} = \mathbf{e}_{k_1}^{(1)} + \mathbf{e}_{k_2}^{(2)} = (2.68, \, -1.28)$, reducing quantization MSE from $0.1300$ down to $(0.02)^2 + (-0.02)^2 = \mathbf{0.0008}$ ($162\times$ lower distortion).

---

## 4. Autoregressive-Diffusion Unification: Transfusion & Show-o

Quantizing continuous images into discrete VQGAN tokens discards fine-grained high-frequency detail compared to continuous latent diffusion (VAE latents). **Transfusion** (Zhou et al. 2024) and **Show-o** (Xie et al. 2024) eliminate visual quantization altogether by training a **single Transformer backbone** under a hybrid objective:

$$\mathcal{L}_{\text{Transfusion}}(\theta) = \mathcal{L}_{\text{CE}}(\text{discrete text tokens}) + \lambda \cdot \mathbb{E}_{t, \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \text{context})\right\|_2^2\right]$$

1. **Hybrid Causal-Bidirectional Attention Mask:** Text tokens attend causally ($j \le i$). Whenever the model encounters or generates an image start token `<BOI>`, $N$ continuous VAE patch latents $\mathbf{z}_t$ are inserted into the sequence with **full bidirectional attention** among the $N$ patches of that image, while attending causally to all preceding text and images.
2. **Unified Representations:** By sharing 98% of Transformer parameters across both discrete next-token prediction and continuous flow/diffusion denoising, Transfusion matches specialized diffusion models (SDXL) on image synthesis while matching LLaMA on text reasoning.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2303.15343" title="Sigmoid Loss for Language Image Pre-Training (SigLIP)" subtitle="Zhai et al. (ICCV 2023) • Replaced global softmax contrastive normalization with pairwise sigmoid loss for scalable vision-language alignment." >}}
  {{< card link="https://arxiv.org/abs/2405.09818" title="Chameleon: Mixed-Modal Early-Fusion Foundation Models" subtitle="Meta Chameleon Team (2024) • Trained a unified token-based early-fusion model over interleaved text and image VQ codebooks stabilized by QK-Norm." >}}
  {{< card link="https://arxiv.org/abs/2408.11039" title="Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" subtitle="Zhou et al. (2024) • Unified discrete causal language modeling with continuous bidirectional image diffusion inside a single Transformer." >}}
  {{< card link="https://arxiv.org/abs/2409.12191" title="Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" subtitle="Wang et al. (2024) • Introduced Naive Dynamic Resolution with 2×2 pixel-shuffle compression and 3D Multimodal RoPE (M-RoPE)." >}}
{{< /cards >}}
