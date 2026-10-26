# Unified Visual Components & Pedagogical Design Spec

**Date:** 2026-10-26  
**Target Audience:** 1st and 2nd-year graduate students (intuitive motivation + rigorous mathematical clarity)

---

## 1. Root Cause Diagnosis of Previous Visual Failures

1. **Uncompiled Tailwind Classes in Shortcodes**:
   - Hextra ships a pre-compiled CSS stylesheet (`main.css`) with `hx:` prefixes. It does **not** run a Tailwind JIT compiler on user shortcode HTML.
   - Because previous shortcodes (`bpe-interactive.html`, `rope-interactive.html`, `attention-interactive.html`, `attention-diagram.html`) used standard unprefixed Tailwind classes (`flex`, `grid`, `p-6`, `rounded-2xl`, `w-3.5`, `h-3.5`, `bg-slate-50`), **none of those styles applied**.
   - As a result, button icons (`<svg class="w-3.5 h-3.5">`) expanded to full 800px screen width, grids collapsed into unpadded vertical lists, and card headers rendered as raw unstyled text.
2. **Unused Motion Canvas Stub**:
   - `layouts/shortcodes/motion-canvas.html` was created as an empty player wrapper pointing to an external script without any actual animations or usage on any page.
3. **KaTeX Dual-Stream Duplication (`htht`, `tt`)**:
   - Hugo's `transform.ToMath` outputs `htmlAndMathml`. Without `.katex-mathml { display: none !important; }`, selecting or copying text captures both the MathML and HTML streams simultaneously, doubling every symbol and equation.
4. **Cramped Markdown Tables**:
   - Standard markdown tables auto-shrink the first column, causing expressions like $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ to wrap awkwardly across lines.

---

## 2. Design Principles

1. **Zero Uncompiled Utility Classes**:
   - All custom components will use **dedicated, semantic BEM-style CSS classes** (`llm-card`, `llm-grid-2`, `llm-grid-3`, `llm-badge`, `llm-sim-container`, etc.) defined explicitly in `assets/css/custom.css`.
   - Every SVG will have explicit `width`, `height`, and `viewBox` attributes so it can never blow up in size.
2. **Pedagogy First ("What is the student learning, and how?")**:
   - Every visual or interactive experiment must answer **one concrete conceptual question** with minimal UI clutter and immediate visual feedback.
3. **Component-First Verification**:
   - Build each component template first, render it in headless Chrome (light and dark modes), verify color, typography, proportions, and layout via screenshots, and only then deploy across pages.

---

## 3. Component Templates Specification

### Component 1: Mathematical Symbol Cards (Option 2 — 2-Column Grid Cards)
- **Purpose**: Replace cramped markdown tables with a clean 2-column grid (`.llm-math-grid`).
- **Structure of Each Card (`.llm-math-card`)**:
  - **Top Header Bar**:
    - Left: Generous symbol box (`.llm-math-symbol`) rendering the full expression (e.g., $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$) on a single line with `white-space: nowrap`.
    - Right: Monospace tensor shape pill badge (`.llm-math-shape`, e.g., `ℝ^(T × d_k)`).
  - **Body**:
    - Bold concept title (e.g., **Query Projection Matrix**) + concise 1–2 sentence explanation of its role in the computation.

### Component 2: Architecture Blueprint Card (`.llm-blueprint`)
- **Purpose**: Standardized container and SVG design system for block diagrams (e.g., Scaled Dot-Product Attention pipeline).
- **Visual Standards**:
  - Styled exclusively via `assets/css/custom.css` + self-contained SVG styles.
  - Consistent semantic color palette:
    - **Queries ($\mathbf{Q}$)**: Indigo (`#4f46e5` / `#eef2ff`)
    - **Keys ($\mathbf{K}$)**: Sky Blue (`#0284c7` / `#e0f2fe`)
    - **Values ($\mathbf{V}$)**: Emerald (`#059669` / `#d1fae5`)
    - **Attention Weights ($\mathbf{A}$)**: Violet (`#7c3aed` / `#f3e8ff`)
    - **Masking / Control**: Amber (`#d97706` / `#fef3c7`)
  - Every stage block shows **Operation Name**, **Mathematical Formula**, and **Output Tensor Shape** (`[T × T]`, `[T × d_v]`).

### Component 3: MHA vs. GQA vs. MQA Side-by-Side Blueprint Grid (Option 2 — 3-Column Grid)
- **Purpose**: Teach *why* modern LLMs replaced Multi-Head Attention (MHA) with Grouped-Query Attention (GQA) or Multi-Query Attention (MQA) to solve the KV-cache memory bandwidth bottleneck.
- **Structure (`.llm-grid-3`)**:
  - **Column 1: Multi-Head Attention (MHA)**
    - Visual: 8 Query heads $\leftrightarrow$ 8 Key/Value heads (1:1 mapping).
    - Metric Badge: `KV Cache: 100% (Baseline)` • High memory bandwidth cost.
  - **Column 2: Grouped-Query Attention (GQA — Llama-3 / Mistral)**
    - Visual: 8 Query heads grouped into pairs sharing 2 Key/Value heads (4:1 mapping).
    - Metric Badge: `KV Cache: 25% (4× Reduction)` • Near-zero perplexity loss.
  - **Column 3: Multi-Query Attention (MQA — PaLM / Falcon)**
    - Visual:All 8 Query heads sharing 1 single Key/Value head (8:1 mapping).
    - Metric Badge: `KV Cache: 12.5% (8× Reduction)` • Maximum throughput.

### Component 4: Simple, Purpose-Built Interactive Experiments
Instead of bloated, broken dashboards, each page gets one focused, clean interactive experiment styled via `assets/css/custom.css`:
1. **Tokenization (`bpe-interactive.html`)**:
   - **What the student learns**: How greedy pair merging compresses character sequences into subwords and segments unseen words.
   - **Fix**: Replace broken Tailwind classes with clean `.llm-sim` CSS, fix the 800px SVG arrow bug, and show a clean step slider (`Step 0` $\to$ `Step 4`) with highlighted merges.
2. **Positional Encoding (`rope-interactive.html`)**:
   - **What the student learns**: Why rotating Query at position $m$ and Key at position $n$ by angle $\theta$ makes their dot product depend *only* on relative distance $(n - m)$.
   - **Fix**: Clean 2-column layout (2D unit circle on the left, live angle & dot-product readout on the right) with explicit CSS rules so shifting $m$ and $n$ together visibly preserves the angle between vectors.
3. **Attention (`attention-interactive.html`)**:
   - **What the student learns**: How temperature scaling ($\sqrt{d_k}$) controls distribution sharpness (entropy) and how the causal mask zeroes out future tokens ($j > i$).
   - **Fix**: Clean 3-matrix horizontal pipeline ($\mathbf{Q}\mathbf{K}^T \to \text{Scaled & Masked} \to \text{Softmax }\mathbf{A}$) with a temperature slider and causal mask toggle.
