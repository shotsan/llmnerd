# LLMNerd Strict Visual, Typography & Contrast Rules

Every page, CSS rule, table, diagram, and interactive visualization on `llmnerd.dev` MUST obey these rules without exception.

---

## 1. Maximum 4-Color Palette Rule (Zero Rainbow, Zero Grey Text)

The entire site—including all prose, tables, SVG diagrams, and interactive animations—is restricted to **4 core colors**:

| Role | Token | Hex | Allowed Usage | Contrast vs. Surface |
| :--- | :--- | :--- | :--- | :--- |
| **1. Canvas & Card Surface** | `--llm-bg-card`<br>`--llm-bg-subtle` | `#ffffff`<br>`#f5f4f0` | Page background, card backgrounds, table headers, inactive matrix cells. | Base Surface |
| **2. Structural Border** | `--llm-border` | `#c2bfb6` | Table rules, card borders, matrix gridlines. **Never use for text.** | `1.85 : 1` (Non-text stroke) |
| **3. Primary Ink (All Text)** | `--llm-ink` | `#141413` | **All** prose, headings, figure captions, table cells, matrix numbers, sub-labels (`Token Mixing`, `3 × 2`), and diagram arrows. | **`18.8 : 1` (WCAG AAA)** on `#ffffff`<br>**`17.1 : 1` (WCAG AAA)** on `#f5f4f0` |
| **4. Pedagogical Highlight** | `--llm-focus`<br>`--llm-focus-tint` | `#991b1b`<br>`#fef2f2` | Active mathematical focus ONLY (e.g., active Query row, Residual Highway, active merge pair). | **`8.9 : 1` (WCAG AAA)** on `#ffffff`<br>**`8.1 : 1` (WCAG AAA)** on `#fef2f2` |

### Strictly Banned Colors:
- **NO Grey Text (`#5e5c57`, `#78716c`, `#88857e`, `#6b7280`):** Every word, caption, dimension label `(3 × 2)`, and sub-label (`Token Mixing`, `Channel MLP`) is meant to be read and MUST use `#141413` Primary Ink (`>17:1` AAA contrast).
- **NO Blue, Green, Yellow, or Purple (`#1d4ed8`, `#2563eb`, `#10b981`, etc.):** Never introduce secondary accent colors for boxes, badges, or sublayers. Use `#ffffff` or `#f5f4f0` fills with `#141413` borders and `#141413` text, reserving `#991b1b` / `#fef2f2` strictly for the active pedagogical element.
- **NO Header Pill Badges:** Do not put decorative badges (`Identity Gradient Highway`, `Exact Tensor Grids`, `Ingestion Pipeline`) in figure headers.

---

## 2. Strict 4-Size Typography Rule

Only **4 font sizes** are permitted across the entire site:

1. **Article Title (`H1`):** `2.15rem` (`34px`), weight `750`, line-height `1.25`, color `#141413`.
2. **Section Heading (`H2`):** `1.45rem` (`23px`), weight `700`, line-height `1.3`, top margin `2.75rem`, bottom border `1.5px solid #c2bfb6`, color `#141413`.
3. **Subsection / Figure Title (`H3`):** `1.05rem` (`17px`), weight `700`, color `#141413`.
4. **Body / Table / Caption Text:** `1.0625rem` (`17px`, line-height `1.8`) for prose; `0.95rem` (`15.2px`, line-height `1.55`, color `#141413`) for tables and figure captions.
   - **Inside SVG Diagrams:** Only `14px` (bold titles/numbers) and `13px` (sub-labels/dimensions), always in `#141413` or `#991b1b`. Never use `9px`–`11px` micro-text or `text-transform: uppercase`.

---

## 3. Mathematical Typography (Overleaf / IEEE Standard)

- **Inline Math (`.katex`):** `1.02em` (matches surrounding `17px` body copy; overrides KaTeX's bloated `1.21em` default).
- **Display Equations (`.katex-display > .katex`):** `1.05em`, centered inside a clean callout (`background: #ffffff`, `border: 1px solid #c2bfb6`, `border-left: 3.5px solid #991b1b`).
- **Subscripts & Superscripts:** First-level scripts (`\scriptstyle`) fixed at `0.64em`; second-level scripts (`\scriptscriptstyle`) fixed at `0.52em`. No artificial `translateY` hacks.

---

## 4. Scientific Visuals & Content Structure Rules

1. **Notation Tables Instead of Multi-Card Grids:**
   - Summarize symbols, tensor shapes, and geometric roles in a single, full-width horizontal-rule table (`.llm-notation-table`), never in 2-column or 3-column grids of small cards.
2. **True Tensor & Architectural Diagrams (Never Text-in-Boxes):**
   - Every static figure must show actual **tensor grids** (rows, columns, and cell values contracting across matrix multiplications) or **true architectural schematics** (continuous residual stream highway with branching sublayers and $+$ nodes).
3. **Meaningful Step-by-Step Mathematical Animations:**
   - Every interactive visualizer must step through the exact numerical worked example from the article text, highlighting the active row/column operation in `#991b1b` on `#fef2f2` while all other elements remain in clean `#141413` ink on `#ffffff`.
4. **Consistent 5-Part Article Flow:**
   - Every article follows: **(1) Core Problem & Intuition $\to$ (2) Governing Equation & Notation Table $\to$ (3) Visual Tensor/Architecture Diagram & Worked Numerical Trace $\to$ (4) Interactive Step-by-Step Mathematical Visualizer $\to$ (5) Production & Hardware Reality Table + References.**
