# Foundations Section Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the Foundations section of `llmnerd` with rich contextual narrative, high-fidelity Manim-style vector illustrations, concrete step-by-step toy examples, deep mathematical and systems mechanics, architectural evolution (Transformer -> T5 -> BERT -> GPT -> MoE -> DeepSeek -> Diffusion), and academic paper citation cards.

**Architecture:** Hugo static site with Hextra theme. Standalone Manim-style vector SVGs (`#0f172a` dark background, neon geometric accents) placed in `assets/images/foundations/` and referenced cleanly in Markdown content under `content/foundations/`.

**Tech Stack:** Hugo v0.168.0, Go modules (Hextra v0.11.1), SVG vector graphics, KaTeX math typesetting, Markdown/Goldmark.

## Global Constraints
- Account: Always use GitHub account `shotsan` (never `santosh-agebold`).
- Tone: Slow-paced, rigorous, intuitive, and authoritative. Avoid boilerplate like "Why this exists" or brief 2-sentence summaries; motivate each concept organically through computational and mathematical trade-offs.
- Structure for all 5 pages:
  1. Deep organic contextual opening
  2. Manim-style hero vector illustration (`assets/images/foundations/*.svg`)
  3. Concrete toy example with step-by-step calculations
  4. Deep dive into mechanics, tensor shapes, and modern production variants
  5. Seminal research paper badges with links and key takeaways
- Local Preview: Dev server running at `http://localhost:1313`. All pages must render with HTTP 200 and build without error via `hugo --gc --minify`.

---

### Task 1: Generate Manim-Style Vector SVG Illustrations

**Files:**
- Create: `assets/images/foundations/tokenization_pipeline.svg`
- Create: `assets/images/foundations/embedding_lookup.svg`
- Create: `assets/images/foundations/positional_encoding_rope.svg`
- Create: `assets/images/foundations/scaled_dot_product_attention.svg`
- Create: `assets/images/foundations/model_architecture_evolution.svg`
- Create: `assets/images/foundations/deepseek_mla_moe.svg`

**Interfaces:**
- Produces: 6 self-contained SVG files in `assets/images/foundations/` styled with dark background (`#0f172a`), gridlines, glowing neon accents (cyan `#38bdf8`, gold `#facc15`, purple `#c084fc`, emerald `#34d399`, coral `#f87171`), clear mathematical labels, and responsive viewBox.

- [ ] **Step 1: Create `assets/images/foundations/tokenization_pipeline.svg`**
  Generate vector illustration showing string `"understanding"` -> byte splits -> pair frequencies -> merges -> token IDs.

- [ ] **Step 2: Create `assets/images/foundations/embedding_lookup.svg`**
  Generate vector illustration showing discrete token ID lookup into weight matrix $\mathbf{W}_E$ projecting into continuous vector space with semantic clusters and cosine distances.

- [ ] **Step 3: Create `assets/images/foundations/positional_encoding_rope.svg`**
  Generate vector illustration comparing Sinusoidal harmonic waves across dimensions vs. Rotary Position Embedding (RoPE) 2D complex plane rotation vectors $\mathbf{R}_{\Theta, m}$.

- [ ] **Step 4: Create `assets/images/foundations/scaled_dot_product_attention.svg`**
  Generate vector illustration showing Query, Key, Value matrix pipeline ($Q K^T / \sqrt{d_k}$, causal mask, softmax, $\times V$, multi-head projection).

- [ ] **Step 5: Create `assets/images/foundations/model_architecture_evolution.svg`**
  Generate evolutionary lineage tree: 2017 Transformer -> BERT (Encoder) / T5 (Encoder-Decoder) / GPT (Decoder) -> MoE (Switch/Mixtral) -> DeepSeek (MLA/DeepSeekMoE) -> Diffusion text models.

- [ ] **Step 6: Create `assets/images/foundations/deepseek_mla_moe.svg`**
  Generate vector illustration detailing DeepSeek Multi-Head Latent Attention (down-projection to compressed latent $c^{KV}$) and fine-grained routed/shared MoE architecture.

- [ ] **Step 7: Verify all SVGs are valid and build with Hugo**
  Run: `hugo --gc --minify`
  Expected: PASS with 0 errors.

- [ ] **Step 8: Commit**
  Run: `git add assets/images/foundations/ && git commit -m "feat: add Manim-style vector SVG illustrations for Foundations"`

---

### Task 2: Overhaul `tokenization.md`

**Files:**
- Modify: `content/foundations/tokenization.md`

**Interfaces:**
- Consumes: `assets/images/foundations/tokenization_pipeline.svg`
- Produces: Fully comprehensive guide to tokenization with organic narrative context, concrete BPE toy walkthrough, deep-dive on byte-level fallback & multilingual fertility, and academic paper cards.

- [ ] **Step 1: Write comprehensive content for `tokenization.md`**
  - Organic narrative on the symbol-vector boundary and information density trade-offs.
  - Embed `tokenization_pipeline.svg`.
  - Concrete toy BPE example with corpus `["low", "lower", "newest", "widest"]`, iteration counts, merge rules, and tokenizing `"lowest"`.
  - Deep dive: Byte-level BPE (GPT-2/Tiktoken/Llama 3) with UTF-8 byte fallback ($256$ base tokens), WordPiece vs SentencePiece, tokenization pathologies (whitespace sensitivity, arithmetic splitting, multilingual fertility disparity).
  - Paper cards: Sennrich et al. (2015), Kudo & Richardson (2018), Radford et al. (2019).

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/tokenization/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/tokenization.md && git commit -m "feat: overhaul tokenization page with toy example and deep dive"`

---

### Task 3: Overhaul `embeddings.md`

**Files:**
- Modify: `content/foundations/embeddings.md`

**Interfaces:**
- Consumes: `assets/images/foundations/embedding_lookup.svg`
- Produces: Comprehensive guide to continuous vector representations, lookup mechanics, weight tying, normalization, and semantic geometry.

- [ ] **Step 1: Write comprehensive content for `embeddings.md`**
  - Organic narrative on the failure of one-hot orthogonality and the emergence of continuous semantic manifolds.
  - Embed `embedding_lookup.svg`.
  - Concrete toy example: Vocabulary $V=5$, dimension $d=3$, matrix $\mathbf{W}_E$, token lookup, dot product and cosine similarity calculations.
  - Deep dive: Weight tying ($\mathbf{W}_{\text{head}} = \mathbf{W}_E^T$), embedding scaling factor $\sqrt{d_{\text{model}}}$, LayerNorm vs RMSNorm ($y = \frac{x}{\text{RMS}(x)} \odot \gamma$).
  - Paper cards: Mikolov et al. (2013), Press & Wolf (2016), Zhang & Sennrich (2019).

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/embeddings/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/embeddings.md && git commit -m "feat: overhaul embeddings page with toy example and deep dive"`

---

### Task 4: Overhaul `positional_encoding.md`

**Files:**
- Modify: `content/foundations/positional_encoding.md`

**Interfaces:**
- Consumes: `assets/images/foundations/positional_encoding_rope.svg`
- Produces: Comprehensive guide to order injection, permutation symmetry breaking, Sinusoidal vs RoPE vs ALiBi, complex inner products, and long-context extension.

- [ ] **Step 1: Write comprehensive content for `positional_encoding.md`**
  - Organic narrative explaining attention's permutation invariance ($\text{Attention}(\mathbf{P}X) = \mathbf{P}\text{Attention}(X)$) and why order must be geometric.
  - Embed `positional_encoding_rope.svg`.
  - Concrete toy example: Sinusoidal evaluation ($pos=0, 1$, $d=4$) and 2D complex plane RoPE rotation matrix applied to query/key.
  - Deep dive: RoPE mathematical proof ($\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = g(\mathbf{q}, \mathbf{k}, m-n)$), ALiBi linear bias, and context extension methods (NTK-aware scaling, YaRN, $\theta$ frequency scaling).
  - Paper cards: Vaswani et al. (2017), Su et al. (2021), Press et al. (2021), Peng et al. (2023).

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/positional_encoding/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/positional_encoding.md && git commit -m "feat: overhaul positional encoding page with toy example and deep dive"`

---

### Task 5: Overhaul `attention_mechanisms.md`

**Files:**
- Modify: `content/foundations/attention_mechanisms.md`

**Interfaces:**
- Consumes: `assets/images/foundations/scaled_dot_product_attention.svg`
- Produces: Comprehensive guide to soft associative retrieval, QKV matrix algebra, causal masking, MHA vs MQA vs GQA, and FlashAttention IO awareness.

- [ ] **Step 1: Write comprehensive content for `attention_mechanisms.md`**
  - Organic narrative on sequential recurrence bottlenecks in RNNs and parallel soft dictionary lookups.
  - Embed `scaled_dot_product_attention.svg`.
  - Concrete toy example: 3 tokens (`"The"`, `"cat"`, `"sat"`), $d=2$, explicit $Q, K, V$ matrices, scaled dot product, causal masking with $-\infty$, row softmax, and value aggregation.
  - Deep dive: MHA vs MQA vs GQA KV-cache memory bandwidth comparison; FlashAttention SRAM tiling and memory hierarchy.
  - Paper cards: Vaswani et al. (2017), Shazeer (2019), Ainslie et al. (2023), Dao et al. (2022).

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/attention_mechanisms/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/attention_mechanisms.md && git commit -m "feat: overhaul attention mechanisms page with toy example and deep dive"`

---

### Task 6: Overhaul `model_architecture.md` (Evolutionary Lineage)

**Files:**
- Modify: `content/foundations/model_architecture.md`

**Interfaces:**
- Consumes: `assets/images/foundations/model_architecture_evolution.svg`, `assets/images/foundations/deepseek_mla_moe.svg`
- Produces: Definitive evolutionary survey across the 6 major architectural epochs (Transformer -> BERT/GPT/T5 -> Pre-LN/LLaMA -> MoE -> DeepSeek -> Diffusion).

- [ ] **Step 1: Write comprehensive content for `model_architecture.md`**
  - Organic narrative on empirical scaling laws, parameter-compute trade-offs, and architecture shifts.
  - Embed `model_architecture_evolution.svg`.
  - Section 1: The Original Transformer (Vaswani 2017, Encoder-Decoder, cross-attention).
  - Section 2: The Branching Era (BERT bidirectional encoder vs GPT causal decoder vs T5 text-to-text).
  - Section 3: The Modern Decoder-Only Foundation (Pre-RMSNorm, SwiGLU gating, GQA).
  - Section 4: Mixture of Experts (MoE) (Switch Transformer, Mixtral 8x7B, sparse routing, load balancing).
  - Embed `deepseek_mla_moe.svg`.
  - Section 5: DeepSeek Architectural Breakthroughs (Multi-Head Latent Attention KV-cache compression, DeepSeekMoE fine-grained routed/shared experts, auxiliary-loss-free balancing).
  - Section 6: Continuous & Diffusion-Style Language Models (Diffusion-LM, Plaid, iterative denoising, parallel non-autoregressive decoding).
  - Paper cards: Vaswani (2017), Devlin (2018), Radford (2018), Raffel (2019), Fedus (2021), Jiang (2024), DeepSeek-AI (2024), Li (2022).

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/model_architecture/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/model_architecture.md && git commit -m "feat: overhaul model architecture page with full evolutionary lineage"`

---

### Task 7: Update `content/foundations/_index.md` & Overview Cards

**Files:**
- Modify: `content/foundations/_index.md`

**Interfaces:**
- Produces: Polished overview page for Foundations linking to all 5 upgraded topics with crisp descriptions and consistent layout.

- [ ] **Step 1: Update `content/foundations/_index.md`**
  - Provide an inviting overview highlighting the 5 core conceptual pillars.
  - Verify links and descriptions.

- [ ] **Step 2: Verify page build and rendering**
  Run: `hugo --gc --minify && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:1313/foundations/`
  Expected: 200

- [ ] **Step 3: Commit**
  Run: `git add content/foundations/_index.md && git commit -m "feat: update foundations index page with curated overview"`

---

### Task 8: End-to-End Build and Verification

**Files:**
- Review all touched files in `content/foundations/` and `assets/images/foundations/`.

- [ ] **Step 1: Execute full production Hugo build**
  Run: `hugo --gc --minify`
  Expected: Site builds with 0 errors.

- [ ] **Step 2: Verify all 5 pages and illustrations over HTTP**
  Run automated curl checks across all 5 endpoints:
  - `http://localhost:1313/foundations/`
  - `http://localhost:1313/foundations/tokenization/`
  - `http://localhost:1313/foundations/embeddings/`
  - `http://localhost:1313/foundations/positional_encoding/`
  - `http://localhost:1313/foundations/attention_mechanisms/`
  - `http://localhost:1313/foundations/model_architecture/`
  Expected: All return HTTP 200.

- [ ] **Step 3: Final Git status check**
  Run: `git status`
  Expected: Working tree clean, everything committed under account `shotsan`.
