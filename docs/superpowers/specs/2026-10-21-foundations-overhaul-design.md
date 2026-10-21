# Foundations Section Overhaul Design Specification

## Overview
This specification details the comprehensive educational overhaul of the **Foundations** section of the `llmnerd` Hugo documentation website.

The goal is to provide a clean, slow-paced, and rigorous learning environment for large language model architecture. Each page will feature:
1. **Organic Narrative Context**: Deep prose detailing the computational bottlenecks and representational friction that motivated each technique (avoiding artificial "Why this exists" boilerplate).
2. **Manim-Style Vector Illustrations**: High-fidelity, mathematically precise SVGs with dark-canvas aesthetics (`#0f172a`), neon accents (cyan `#38bdf8`, gold `#facc15`, purple `#c084fc`, emerald `#34d399`, coral `#f87171`), and clean typography.
3. **Concrete Toy Examples**: Step-by-step arithmetic and tensor transformations using small, tangible tokens and matrix dimensions ($d=2, 4$, $T=3$).
4. **Deep-Dive Technical Mechanics**: Precise formulations, tensor shapes, gradient flow, and modern production variations (e.g., Tiktoken byte-fallback, RoPE frequency scaling, FlashAttention, SwiGLU, RMSNorm).
5. **Architectural Lineage (Model Architecture)**: An evolutionary journey from the original 2017 Transformer to BERT, GPT, T5, MoEs (Mixtral/Switch), DeepSeek (MLA & DeepSeekMoE), and continuous/discrete Diffusion language models.
6. **Academic Paper Badges**: Formatted citation cards linking to original seminal publications with authors, year, arXiv/DOI links, and foundational contributions.

---

## Site Assets & Directory Structure
```
content/
└── foundations/
    ├── _index.md
    ├── tokenization.md
    ├── embeddings.md
    ├── positional_encoding.md
    ├── attention_mechanisms.md
    └── model_architecture.md

assets/
└── images/
    └── foundations/
        ├── tokenization_pipeline.svg
        ├── embedding_lookup.svg
        ├── positional_encoding_rope.svg
        ├── scaled_dot_product_attention.svg
        ├── model_architecture_evolution.svg
        └── deepseek_mla_moe.svg
```

---

## Detailed Specifications by Page

### 1. Tokenization (`content/foundations/tokenization.md`)
* **Organic Context**: How neural networks operate on continuous vector spaces while human language is unbounded discrete symbolic sequences. The tension between character-level representation (short vocabularies, but prohibitively long sequence lengths that explode quadratic attention compute) and word-level representation (manageable sequence length, but catastrophic out-of-vocabulary (OOV) rates, morphological blindness, and massive embedding tables). Subword algorithms as the optimal information-theoretic compression frontier.
* **Manim Illustration**: `tokenization_pipeline.svg` — Visual flow of the text `"understanding"` decomposing into character bytes, evaluating bigram frequency distributions, iteratively merging common pairs, and outputting token IDs.
* **Concrete Toy Example**:
  * Toy Corpus: `["low", "lower", "newest", "widest"]` with word frequencies.
  * Step 1: Base vocabulary initialization (characters + end-of-word marker `</w>`).
  * Step 2: Frequency tally of adjacent pairs (`('l', 'o')`, `('e', 's')`, etc.).
  * Step 3: Top-ranked pair merge into new subword token.
  * Step 4: Iteration table showing vocabulary expansion and tokenization of unseen word `"lowest"`.
* **Deep Dive Mechanics**:
  * Byte-Pair Encoding (BPE) vs. WordPiece (likelihood maximization) vs. SentencePiece (Unigram language model / byte fallback).
  * Byte-level BPE (GPT-2, Tiktoken, Llama 3): Eliminating OOV by falling back to UTF-8 raw bytes ($256$ base tokens).
  * Tokenizer pathologies: Sensitivity to leading whitespaces, arithmetic degradation (splitting numbers into inconsistent chunks like `100` vs `10` + `0`), and multilingual fertility disparity (non-English scripts consuming 3-5x more tokens per concept).
* **Seminal Paper Badges**:
  * Sennrich et al. (2015): *Neural Machine Translation of Rare Words with Subword Units* (arXiv:1508.07909).
  * Kudo & Richardson (2018): *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing* (arXiv:1808.06226).
  * Radford et al. (2019): *Language Models are Unsupervised Multitask Learners* (GPT-2 Byte-level BPE).

---

### 2. Embeddings (`content/foundations/embeddings.md`)
* **Organic Context**: Transforming discrete integer indices into a geometry of meaning. Why one-hot vectors ($e_i \in \mathbb{R}^V$) are computationally prohibitive ($V \times V$ matrices) and represent all words as mutually orthogonal ($e_i \cdot e_j = 0$), destroying semantic affinity. Continuous embeddings project discrete tokens onto a low-dimensional manifold ($\mathbb{R}^d$) where linear spatial directions encode linguistic properties.
* **Manim Illustration**: `embedding_lookup.svg` — Matrix lookup from token ID to vector row, projected into 3D geometric embedding space showing semantic clusters, cosine similarity vectors, and linear analogies.
* **Concrete Toy Example**:
  * Vocabulary $V=5$, Embedding dimension $d=3$.
  * Weight matrix $\mathbf{W}_E \in \mathbb{R}^{5 \times 3}$.
  * Token indices `[1, 4]` mapped to matrix rows $\mathbf{e}_1, \mathbf{e}_4$.
  * Calculation of dot product $\mathbf{e}_1 \cdot \mathbf{e}_4$ and cosine similarity $\cos(\theta) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_4}{\|\mathbf{e}_1\| \|\mathbf{e}_4\|}$ to quantify semantic proximity.
* **Deep Dive Mechanics**:
  * Weight Tying (Press & Wolf): Binding input embedding $\mathbf{W}_E$ to final pre-softmax output projection $\mathbf{W}_{\text{head}} = \mathbf{W}_E^T$, slashing parameter counts and regularizing representations.
  * Embedding Scaling: The mathematical purpose of multiplying input embeddings by $\sqrt{d_{\text{model}}}$ in Transformer architectures (preserving signal variance against positional encoding additions).
  * Normalization dynamics: Post-embedding LayerNorm vs. RMSNorm ($y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$).
* **Seminal Paper Badges**:
  * Mikolov et al. (2013): *Distributed Representations of Words and Phrases and their Compositionality* (Word2Vec, arXiv:1310.4546).
  * Press & Wolf (2016): *Using the Output Embedding to Improve Language Models* (arXiv:1608.05859).
  * Zhang & Sennrich (2019): *Root Mean Square Layer Normalization* (RMSNorm, arXiv:1910.07467).

---

### 3. Positional Encoding (`content/foundations/positional_encoding.md`)
* **Organic Context**: Self-attention is inherently permutation-invariant ($\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P}\,\text{Attention}(\mathbf{X})$ for any permutation matrix $\mathbf{P}$). Without explicit positional signaling, the sentence *"Dog bites man"* is mathematically indistinguishable from *"Man bites dog"*. How models break permutation symmetry by injecting sequence ordering into the geometric space.
* **Manim Illustration**: `positional_encoding_rope.svg` — Dual comparison diagram showing Sinusoidal harmonic waveforms across dimension channels alongside 2D complex plane rotation vectors for Rotary Position Embedding (RoPE).
* **Concrete Toy Example**:
  * Toy dimension $d=4$, Sequence positions $m=0, 1$.
  * Sinusoidal computation: evaluating $\sin(m / 10000^{0/4})$ and $\cos(m / 10000^{2/4})$.
  * RoPE computation: Taking a 2D query vector $\mathbf{q} = [1.0, 0.0]^T$ and key vector $\mathbf{k} = [0.0, 1.0]^T$, applying orthogonal rotation matrix $\mathbf{R}_{\theta, m}$, and computing their inner product $\mathbf{q}^T \mathbf{R}_{\theta, m-n} \mathbf{k}$ to prove it depends purely on relative distance $m-n$.
* **Deep Dive Mechanics**:
  * Absolute (Sinusoidal, Learned Absolute) vs Relative (T5, ALiBi) vs Rotary (RoPE).
  * RoPE Mathematical Proof: Formulating positional information as a complex multiplication $\mathbf{q}_m = \mathbf{W}_q \mathbf{x}_m e^{i m \theta}$, ensuring inner products $\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re}(\mathbf{q}_m \mathbf{k}_n^*)$ decay naturally with relative token separation.
  * Long-context extrapolation techniques: Why naive RoPE breaks beyond training length; NTK-aware scaling, YaRN (Yet another RoPE extensioN), and base frequency $\theta$ adjustment (from 10,000 to 500,000 in Llama 3).
* **Seminal Paper Badges**:
  * Vaswani et al. (2017): *Attention Is All You Need* (arXiv:1706.03762).
  * Su et al. (2021): *RoFormer: Enhanced Transformer with Rotary Position Embedding* (arXiv:2104.09864).
  * Press et al. (2021): *Train Short, Test Long: Attention with Linear Biases (ALiBi)* (arXiv:2108.12409).
  * Peng et al. (2023): *YaRN: Efficient Context Window Extension of Large Language Models* (arXiv:2309.00071).

---

### 4. Attention Mechanisms (`content/foundations/attention_mechanisms.md`)
* **Organic Context**: The historical computational deadlock of sequence transduction. Recurrent models forced sequential computation ($O(N)$ sequential steps), which acted as an insurmountable barrier to parallelizing training across thousands of GPUs, while fixed-size recurrent hidden states created an information compression bottleneck on long sequences. Attention resolved both by treating contextualization as a parallelized soft content-based associative retrieval operation.
* **Manim Illustration**: `scaled_dot_product_attention.svg` — Query ($Q$), Key ($K$), Value ($V$) matrix pipeline: projection $\to$ dot product $QK^T \to$ scale factor $1/\sqrt{d_k} \to$ causal upper-triangle masking $\to$ softmax distribution $\to$ weighted sum aggregation with $V$, along with multi-head splitting and concatenation.
* **Concrete Toy Example**:
  * 3 Tokens: `"The"`, `"cat"`, `"sat"`. Sequence length $T=3$, head dimension $d=2$.
  * Explicit numeric matrices for $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{3 \times 2}$.
  * Step 1: Raw attention scores $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$.
  * Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$ (explaining why unscaled dot products push softmax into saturated zero-gradient regions for large dimensions).
  * Step 3: Causal masking (setting upper triangle to $-\infty$).
  * Step 4: Row-wise softmax yielding attention weights matrix $\mathbf{A}$.
  * Step 5: Final output context representation $\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{3 \times 2}$.
* **Deep Dive Mechanics**:
  * Attention Variants: Multi-Head Attention (MHA) vs. Multi-Query Attention (MQA) vs. Grouped-Query Attention (GQA).
  * The Inference Memory-Bandwidth Wall: Analysis of KV-cache memory consumption during autoregressive generation and how GQA (e.g. in Llama 2/3, Mistral) reduces KV-cache bandwidth requirements by $8\times$.
  * Hardware-aware attention (FlashAttention-1/2/3): Why naive attention is memory-bandwidth bound ($O(N^2)$ reads/writes to slow GPU High Bandwidth Memory HBM) and how tiling/kernel fusion within on-chip SRAM computes exact softmax in $O(N)$ HBM memory accesses.
* **Seminal Paper Badges**:
  * Vaswani et al. (2017): *Attention Is All You Need* (arXiv:1706.03762).
  * Shazeer (2019): *Fast Transformer Decoding: One Write-Head is All You Need* (MQA, arXiv:1911.02150).
  * Ainslie et al. (2023): *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (arXiv:2305.13245).
  * Dao et al. (2022): *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (arXiv:2205.14135).

---

### 5. Model Architecture (`content/foundations/model_architecture.md`)
* **Organic Context**: The architectural journey of LLMs as an empirical search for the ideal balance between representation power, parallel training efficiency, and inference throughput. How the field evolved through distinct epochs driven by compute efficiency and scaling behavior.
* **Manim Illustrations**:
  * `model_architecture_evolution.svg`: Timeline/family tree mapping the divergence from the 2017 Transformer into Encoder (BERT), Unified Encoder-Decoder (T5), Causal Decoder (GPT/LLaMA), Sparse Router (MoE/DeepSeek), and Non-Autoregressive Diffusion Text Models.
  * `deepseek_mla_moe.svg`: Manim vector diagram illustrating Multi-Head Latent Attention (compressing $K, V$ into low-rank latent $c^{KV}$) alongside fine-grained routed and shared experts.
* **Chronological Evolutionary Sections**:
  1. **The Original Transformer (2017)**: Dual-stack Encoder-Decoder. Cross-attention bridge between bidirectional encoder context and autoregressive decoder generation. Toy translation sequence.
  2. **The Branching Era (2018–2020)**:
     * *BERT (Encoder-Only)*: Deep bidirectional self-attention via Masked Language Modeling (MLM). Strengths in classification/extraction, structural inability to generate open-ended text.
     * *GPT (Decoder-Only)*: Autoregressive next-token prediction with causal triangular masking. Why empirical scaling laws (Kaplan et al., Chinchilla) proved that simple causal prediction absorbs world knowledge and generalizes across tasks without task-specific architectural heads.
     * *T5 (Unified Encoder-Decoder)*: Text-to-text framing of every problem with span corruption pre-training.
  3. **The Modern Decoder-Only Foundation**:
     * Pre-LayerNorm vs. Post-LayerNorm: Removing learning rate warmup requirements and opening unobstructed gradient highways.
     * RMSNorm replacing LayerNorm for a 10–50% speedup.
     * SwiGLU activation ($\text{Swish}(x W) \odot (x V) W_2$) replacing ReLU/GELU with gating mechanisms.
  4. **Mixture of Experts (MoE)**:
     * The parameter-compute dilemma: Why dense models cannot scale parameters indefinitely without linear increases in FLOPs per token.
     * Sparse gating routers (Switch Transformer, Mixtral): Activating Top-$k$ out of $E$ experts per token.
     * Router load-balancing challenges and auxiliary loss dynamics.
  5. **DeepSeek Architectural Breakthroughs**:
     * *Multi-Head Latent Attention (MLA)*: Compressing the Key-Value cache into low-dimensional latent vectors ($c_t^{KV}$) via down-projection matrices, reducing KV-cache memory footprints by over $80\%$ without losing expressive power.
     * *DeepSeekMoE & Auxiliary-Loss-Free Balancing*: Fine-grained expert segmentation (64+ smaller experts with shared dedicated experts) and dynamic router bias adjustments that achieve balance without degrading model performance.
  6. **Continuous & Diffusion-Style Language Models**:
     * The limits of left-to-right autoregressive decoding: Exposure bias, sequential inference bottleneck ($N$ forward passes for $N$ tokens), inability to iteratively revise global plans.
     * Discrete and continuous diffusion generation: Formulating text generation as iterative denoising from Gaussian or categorical noise (Diffusion-LM, Plaid, Plaid-v2, score-based discrete generation).
     * Performance comparisons: Parallel generation speed vs. probability calibration and autoregressive parity.
* **Seminal Paper Badges**:
  * Vaswani et al. (2017): *Attention Is All You Need* (arXiv:1706.03762).
  * Devlin et al. (2018): *BERT: Pre-training of Deep Bidirectional Transformers* (arXiv:1810.04805).
  * Radford et al. (2018): *Improving Language Understanding by Generative Pre-Training* (GPT-1).
  * Raffel et al. (2019): *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)* (arXiv:1910.10683).
  * Fedus et al. (2021): *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (arXiv:2101.03961).
  * Jiang et al. (2024): *Mixtral of Experts* (arXiv:2401.04088).
  * DeepSeek-AI (2024): *DeepSeek-V2 / DeepSeek-V3 Technical Report* (arXiv:2405.04434, arXiv:2412.19437).
  * Li et al. (2022): *Diffusion-LM Improves Controllable Text Generation* (arXiv:2205.14217).

---

## Verification & Local Rendering
* Site builds cleanly with `hugo --gc --minify`.
* Dev server runs on `http://localhost:1313/` for instant live reload and visual inspection of rendered pages, SVGs, formulas, and navigation links.
