# Unified Visual Components & Motion Graphics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken ad-hoc Tailwind shortcodes and cramped tables with a unified, self-contained CSS design system (`assets/css/custom.css`) and reusable Hugo components (2-column Math Definition Cards, Architecture Blueprints, 3-column MHA/GQA/MQA Comparison Grid, and 60fps Play/Pause/Scrub Motion Canvas Players) tested visually across all foundation pages.

**Architecture:** 
- All styling lives in `assets/css/custom.css` under semantic `.llm-*` classes so components never depend on uncompiled Tailwind utility classes.
- Fix KaTeX copy-paste duplication (`.katex-mathml { display: none !important; }`) and document scroll container (`body { overflow-x: hidden; }` only, not `html, body`).
- Each visual shortcode is self-contained, responsive, supports light and dark themes, and is verified via headless Chrome screenshots before deployment.

**Tech Stack:** Hugo templates/shortcodes, vanilla CSS3 (`assets/css/custom.css`), SVG blueprints, 60fps HTML5 Canvas / SVG animation engines.

---

### Task 1: Core Design System CSS & Global Fixes (`assets/css/custom.css`)

**Files:**
- Modify: `assets/css/custom.css`

- [ ] **Step 1: Fix scroll bug and KaTeX duplication in `assets/css/custom.css`**
  - Remove `html` from `html, body { overflow-x: hidden; }` so anchor scrolling works properly.
  - Add `.katex-mathml { display: none !important; }` so copying equations or inline math never outputs `htht` or duplicated equations.
- [ ] **Step 2: Define the Unified Component CSS Design System in `assets/css/custom.css`**
  - `.llm-card` / `.llm-blueprint`: Outer container with header bar, badge pill, title, subtitle, and body.
  - `.llm-math-grid` & `.llm-math-card`: 2-column responsive grid for Mathematical Definitions where symbols (`white-space: nowrap`) and tensor shape badges sit cleanly in a top header bar, and the explanation sits in the card body.
  - `.llm-grid-3` & `.llm-variant-card`: 3-column side-by-side comparison grid for MHA vs. GQA vs. MQA with head diagrams, routing indicators, and KV-cache reduction pills.
  - `.llm-motion-player`: Unified 60fps animation player frame with stage canvas, pedagogical caption bar, Play/Pause button, Step Back/Forward buttons, timeline scrubber slider, step pill indicators, and `0.5× / 1× / 2×` speed toggle.

---

### Task 2: Mathematical Definition Cards Component & RNN Intro Fix (`attention_mechanisms.md`)

**Files:**
- Create: `layouts/shortcodes/visual/math-defs.html` (or `math-card.html`)
- Modify: `content/foundations/attention_mechanisms.md`

- [ ] **Step 1: Rewrite the opening RNN motivation in `content/foundations/attention_mechanisms.md`**
  - Replace the abrupt equation dump with a clear, intuitive explanation of sequential memory bottlenecks vs. parallel associative lookup.
- [ ] **Step 2: Build the 2-column Math Definition Cards shortcode**
  - Create `layouts/shortcodes/visual/attention-math-defs.html` rendering all 12 symbols ($\mathbf{X}, \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{S}, \sqrt{d_k}, \mathbf{M}, \mathbf{A}, \mathbf{O}$) in the `.llm-math-grid` 2-column layout with generous symbol width and tensor shape badges.
- [ ] **Step 3: Screenshot & Verify**
  - Capture screenshot of the Math Definition Cards and verify zero line-wrapping on symbols, balanced card heights, and crisp typography.

---

### Task 3: Scaled Dot-Product Attention & MHA/GQA/MQA Blueprints

**Files:**
- Modify: `layouts/shortcodes/visual/attention-diagram.html`
- Create: `layouts/shortcodes/visual/mha-gqa-mqa.html`
- Modify: `content/foundations/attention_mechanisms.md`

- [ ] **Step 1: Upgrade `layouts/shortcodes/visual/attention-diagram.html` to `.llm-blueprint`**
  - Replace all broken Tailwind classes with `.llm-blueprint` CSS classes.
  - Enhance the SVG pipeline so $\mathbf{X} \to (\mathbf{Q}, \mathbf{K}, \mathbf{V}) \to \mathbf{Q}\mathbf{K}^T \to \text{Scale } (\div\sqrt{d_k}) \to \text{Causal Mask} \to \text{Softmax } (\mathbf{A}) \to \mathbf{A}\mathbf{V} \to \mathbf{O}$ is crystal clear, well-proportioned, and labels every wire's tensor shape.
- [ ] **Step 2: Create `layouts/shortcodes/visual/mha-gqa-mqa.html` (Option 2: 3-Column Side-by-Side Grid)**
  - Build the 3-column blueprint comparing **Multi-Head Attention (MHA)**, **Grouped-Query Attention (GQA)**, and **Multi-Query Attention (MQA)** side-by-side with SVG head-routing diagrams, memory bandwidth formulas, and KV-cache reduction badges (`100% Baseline`, `25% (4× Reduction)`, `12.5% (8× Reduction)`).
- [ ] **Step 3: Screenshot & Verify**
  - Capture screenshots of both blueprints on `attention_mechanisms` and verify color, alignment, and readability.

---

### Task 4: Unified 60fps Motion Graphics Players Across Foundation Pages

**Files:**
- Modify: `layouts/shortcodes/visual/bpe-interactive.html`
- Create: `layouts/shortcodes/visual/embedding-interactive.html`
- Modify: `layouts/shortcodes/visual/rope-interactive.html`
- Modify: `layouts/shortcodes/visual/attention-interactive.html`
- Create: `layouts/shortcodes/visual/architecture-interactive.html`
- Modify: `content/foundations/embeddings.md`, `content/foundations/model_architecture.md`

- [ ] **Step 1: Rebuild `bpe-interactive.html` as a `.llm-motion-player` Animation + Experiment**
  - Pedagogical Goal: Watch character pairs in `"understanding"` and the training corpus highlight, gravitate together, and snap into subword tokens step-by-step as sequence length drops from 13 $\to$ 3 tokens. Includes Play/Pause, Step Back/Forward, timeline scrubber, and live token counter.
- [ ] **Step 2: Rebuild `rope-interactive.html` as a `.llm-motion-player` Animation + Experiment**
  - Pedagogical Goal: Animate $\mathbf{q}_m$ and $\mathbf{k}_n$ rotating in the 2D complex plane as token positions shift ($m \to m+\Delta, n \to n+\Delta$), proving visually and numerically that the angle $(n-m)\theta$ and dot product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ remain strictly invariant to absolute position.
- [ ] **Step 3: Rebuild `attention-interactive.html` as a `.llm-motion-player` Animation + Experiment**
  - Pedagogical Goal: Animate the 4 stages of Attention ($\mathbf{Q}\mathbf{K}^T$ dot-product sweep $\to$ $\div\sqrt{d_k}$ temperature scaling $\to$ causal mask wipe $\to$ softmax probability bars & $\mathbf{A}\mathbf{V}$ value blending), with interactive $\sqrt{d_k}$ temperature slider and causal mask toggle.
- [ ] **Step 4: Add Motion Players for `embeddings.md` and `model_architecture.md`**
  - `embedding-interactive.html`: Animate discrete one-hot IDs projecting into continuous 2D semantic clusters where cosine similarity arrows dynamically measure semantic alignment.
  - `architecture-interactive.html`: Animate a token flowing through the Pre-RMSNorm residual stream highway ($x_{l+1} = x_l + \text{Sublayer}(\text{RMSNorm}(x_l))$), showing how SwiGLU gating and Top-$K$ MoE routing process the vector.
- [ ] **Step 5: Screenshot & Verify All 5 Pages**
  - Verify every page in headless Chrome, build with `hugo --gc --minify`, and push to `origin main` using `shotsan`.
