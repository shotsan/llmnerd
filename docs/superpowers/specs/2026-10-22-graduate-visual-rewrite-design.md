# Design Spec: Graduate Visual & Intuitive Foundations Rewrite

**Date:** 2026-10-22  
**Target Audience:** 1st and 2nd year graduate students in Computer Science & AI  
**Pedagogical Style:** 3Blue1Brown / Jay Alammar (Visual geometric intuition first $\to$ conceptual mechanics $\to$ transparent math $\to$ minimal PyTorch)

---

## 1. Problem Statement & Motivation

The current Foundation articles are dense, jargon-heavy, and structured like compressed reference manuals. While mathematically accurate, they lack pedagogical scaffolding:
- Concepts are introduced abstractly before students build an intuitive mental model.
- Diagrams are static flowcharts rather than interactive, dynamic visual explanations.
- The prose uses opaque nomenclature instead of clear, conversational, and illuminating explanations that graduate students can absorb and remember.

This design establishes a student-centered curriculum pairing rigorous mathematical foundations with rich visual intuition, interactive Canvas/SVG widgets, Motion Canvas animation player support, and clean PyTorch implementations.

---

## 2. Visual Architecture & Animation Components

### 2.1 Motion Canvas Player Support
A custom Hugo shortcode `{{< motion-canvas src="..." >}}` will embed the official `@motion-canvas/player` web component.
- **Shortcode:** `layouts/shortcodes/motion-canvas.html`
- **Engine:** `@motion-canvas/player` loaded via CDN (`https://unpkg.com/@motion-canvas/player/lib/index.js`).
- **Features:** Supports interactive playback, scrubbing, frame looping, and full-screen inspection of programmatic animations.

### 2.2 Interactive HTML5 / Canvas Visualizers (Built-in Shortcodes)
To ensure immediate interactivity without requiring external video assets, we will provide interactive canvas components:

1. **`{{< visual/bpe-interactive >}}` (Tokenization)**:
   - An interactive toy tokenizer widget.
   - Students type a word (e.g. `"lowest"`, `"understanding"`, `"newer"`) and click "Step Merge" to watch subword tokens highlight, fuse, and map to integer IDs in real time.
2. **`{{< visual/rope-interactive >}}` (Positional Encoding)**:
   - 3Blue1Brown-style interactive 2D unit circle.
   - Sliders for Query position $m$ and Key position $n$.
   - Live visual rotating vectors on the complex plane, displaying $(n-m)\theta$ relative displacement and the exact inner product $\mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$ invariance.
3. **`{{< visual/attention-interactive >}}` (Attention Mechanisms)**:
   - Interactive $Q \times K^T$ matrix visualizer.
   - Shows the dot product computation, the $\sqrt{d_k}$ temperature scaling slider, the softmax probability distribution bar chart, and the weighted value blend.

---

## 3. Article-by-Article Rewrite Specifications

### Article 1: Tokenization (`content/foundations/tokenization.md`)
- **Visual Metaphor:** The "Language Compression" Dilemma. Why text is not pixels or audio, and why characters are too long ($\mathcal{O}(N^2)$ attention penalty) while full words are infinite.
- **Core Visuals:**
  - 3-stage pipeline flowchart: Ingestion $\to$ Subword Merge $\to$ Tensor IDs.
  - Interactive BPE merge demo (`{{< visual/bpe-interactive >}}`).
- **Pedagogical Flow:**
  1. *The Dilemma:* Character vs. Word vs. Subword tradeoffs with concrete token count calculations.
  2. *The BPE Algorithm:* Worked step-by-step example on a 4-word corpus ($\text{low}, \text{lower}, \text{newest}, \text{widest}$).
  3. *Byte-Level Fallback:* Why modern models never throw `[UNK]` (UTF-8 byte space).
  4. *Graduate Insights:* Digit splitting in math reasoning, token fertility across languages.
  5. *Code:* 15-line clean BPE tokenizer from scratch in Python.

### Article 2: Embeddings (`content/foundations/embeddings.md`)
- **Visual Metaphor:** The High-Dimensional Semantic Map.
- **Core Visuals:**
  - Geometric coordinate space showing vector projection and cosine angle similarity.
  - Weight tying diagram: Input lookup table $\leftrightarrow$ Output logit projection.
- **Pedagogical Flow:**
  1. *From One-Hot to Continuous:* Why discrete integers fail gradient descent and how dense manifolds capture similarity.
  2. *The Geometry of Embeddings:* Dot product as semantic similarity, vector arithmetic ($v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$).
  3. *Weight Tying & RMSNorm:* Parameter efficiency and representation collapse prevention.
  4. *Code:* PyTorch `nn.Embedding` with manual weight tying and cosine similarity verification.

### Article 3: Positional Encoding (`content/foundations/positional_encoding.md`)
- **Visual Metaphor:** The Clock-Hand & Compass Rotation (RoPE).
- **Core Visuals:**
  - Sinusoidal wave harmonics across dimensions (binary counter analog).
  - Interactive RoPE 2D Complex Unit Circle (`{{< visual/rope-interactive >}}`).
  - ALiBi linear slope bias visualization.
- **Pedagogical Flow:**
  1. *Permutation Invariance Problem:* Why pure self-attention treats "Dog bites man" and "Man bites dog" as identical bags of words.
  2. *Sinusoidal Encodings (Vaswani 2017):* Multi-scale frequencies explained simply with clock hands.
  3. *Rotary Position Embeddings (RoPE):* 2D rotation matrix derivation, complex numbers intuition, and proof of relative displacement invariance.
  4. *Context Extrapolation:* RoPE frequency scaling, YaRN, and Long-Context modeling.
  5. *Code:* PyTorch RoPE module implementation with rotary dimension slicing.

### Article 4: Attention Mechanisms (`content/foundations/attention_mechanisms.md`)
- **Visual Metaphor:** The Search Engine / Dynamic Spotlight.
- **Core Visuals:**
  - Query, Key, Value interaction diagram.
  - Causal attention mask triangular matrix heatmap.
  - KV-Cache timeline showing memory bandwidth bottleneck.
  - Interactive Attention Calculator (`{{< visual/attention-interactive >}}`).
- **Pedagogical Flow:**
  1. *The Query-Key-Value Mental Model:* Database lookup intuition (Query = search term, Key = database tag, Value = retrieved content).
  2. *The Scaled Dot-Product Math:* Why $\sqrt{d_k}$ is mathematically required to prevent softmax gradient saturation.
  3. *Multi-Head & Grouped-Query Attention (MHA vs GQA vs MQA):* Memory bandwidth trade-offs.
  4. *The KV Cache & Systems Bottleneck:* Why LLM generation is memory-bandwidth bound, not compute bound.
  5. *Code:* Self-attention in pure PyTorch with explicit tensor shapes.

### Article 5: Model Architecture (`content/foundations/model_architecture.md`)
- **Visual Metaphor:** The Transformer Highway (Residual Stream).
- **Core Visuals:**
  - Modern Pre-LN Decoder-Only Block diagram with residual highway.
  - SwiGLU / MLP expansion and gating block.
- **Pedagogical Flow:**
  1. *The Residual Stream:* Viewing the transformer not as a rigid hierarchy, but as an iterative refinement highway.
  2. *Post-LN vs Pre-LN:* Why early transformers struggled with deep training and why Pre-LN unlocked 100B+ models.
  3. *The Modern Decoder Stack:* RMSNorm + RoPE + GQA + SwiGLU.
  4. *Code:* Minimal 1-block Transformer in PyTorch.

---

## 4. Verification & Rollout Plan

1. **Step 1:** Create `layouts/shortcodes/motion-canvas.html` for Motion Canvas player integration.
2. **Step 2:** Build the interactive visualizer shortcodes (`layouts/shortcodes/visual/...`).
3. **Step 3:** Incrementally rewrite each article, starting with `tokenization.md` $\to$ `positional_encoding.md` $\to$ `embeddings.md` $\to$ `attention_mechanisms.md` $\to$ `model_architecture.md`.
4. **Step 4:** Verify site builds locally (`hugo --gc --minify`) and confirm zero regressions.
5. **Step 5:** Commit and push to GitHub `shotsan/llmnerd` (`main`) to deploy to Netlify.
