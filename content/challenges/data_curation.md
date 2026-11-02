---
title: Data Curation & Scaling Laws
toc: true
type: docs
math: true
weight: 1
---

When scaling laws first emerged, pre-training was treated primarily as a raw compute race: scrape larger snapshots of the public web and train larger parameter matrices. Today, frontier pre-training is fundamentally **data-constrained**. High-quality human text on the open web is finite, and raw web crawls (such as CommonCrawl) consist of over 85% HTML boilerplate, SEO keyword spam, machine-translated filler, and duplicated templates.

Training a foundation model therefore requires solving two coupled mathematical problems:

1. **Compute-Optimal vs. Inference-Optimal Allocation**: Given a fixed floating-point budget $C$, how should we partition compute between model parameters $N$ and training tokens $D$?
2. **Sub-Quadratic Corpus Curation**: How do we filter and deduplicate $10^{11}$ raw web documents down to $15\text{ trillion}$ high-signal tokens without quadratic $\mathcal{O}(M^2)$ pairwise comparisons or distribution collapse?

---

## Compute-Optimal & Inference-Aware Scaling Laws

In a standard dense Transformer, every training token requires approximately $2N$ floating-point operations (FLOPs) for the forward pass and $4N$ FLOPs for the backward pass, yielding the canonical pre-training compute identity:

$$
C \approx 6 N D
$$

Hoffmann et al. (2022) (*Chinchilla*) modeled the pre-training cross-entropy loss $L(N, D)$ as a sum of three terms: the irreducible entropy of natural text $E$, the finite-parameter approximation error, and the finite-data estimation error:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

where empirical fits across four orders of magnitude of compute give $E \approx 1.69\text{ nats}$, $A \approx 406.4$, $B \approx 410.7$, $\alpha \approx 0.34$, and $\beta \approx 0.34$.

Minimizing $L(N, D)$ subject to the constraint $6ND = C$ via Lagrange multipliers yields the power-law scaling allocation:

$$
N_{\text{opt}}(C) \propto C^{\frac{\beta}{\alpha + \beta}} \approx C^{0.50}, \qquad D_{\text{opt}}(C) \propto C^{\frac{\alpha}{\alpha + \beta}} \approx C^{0.50}
$$

Because $\alpha \approx \beta$, model parameters and training tokens must scale in **1:1 proportion** (~20 training tokens per parameter). However, because inference cost over a deployed model's lifetime scales linearly with parameter count $N$, modern frontier models (LLaMA 3, Qwen, Gemma) intentionally **over-train** smaller parameter matrices up to $D/N \in [200, 1800]$ (e.g., LLaMA 3 8B trained on $15\text{T}$ tokens, or $1{,}875$ tokens per parameter):

{{< visual/data-curation-blueprint >}}

---

## MinHash LSH Deduplication Walkthrough

Duplicate documents in pre-training corpora degrade generalization, allow models to memorize verbatim passages, and waste FLOPs. However, comparing $M = 10^9$ documents pairwise for fuzzy overlap requires $\binom{M}{2} \approx 5 \times 10^{17}$ comparisons.

Frontier pipelines (FineWeb, Dolma, RedPajama) achieve $\mathcal{O}(M)$ near-deduplication using **MinHash with Locality-Sensitive Hashing (LSH)**:

### 1. Document Shingling & Jaccard Similarity
Each document is converted into a set $S$ of character or word $k$-grams (typically $k = 5$). The overlap between documents $d_A$ and $d_B$ is measured by their Jaccard similarity:

$$
J(S_A, S_B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|} \in [0, 1]
$$

### 2. MinHash Signature Compression
Given $K$ independent random hash permutations $\pi_1, \dots, \pi_K$, the MinHash signature of a set $S$ is the vector of minimum hash values:

$$
h_{\min}^{(i)}(S) = \min_{x \in S} \pi_i(x)
$$

By a foundational property of random permutations, the probability that two documents share the exact same minimum hash equals their Jaccard similarity:

$$
\Pr\!\left[ h_{\min}^{(i)}(S_A) = h_{\min}^{(i)}(S_B) \right] = J(S_A, S_B)
$$

### 3. Banded LSH S-Curve Calculation
We partition a signature of length $K = 100$ into $b = 20$ bands of $r = 5$ rows each. Two documents are flagged as candidate duplicates if they match on **all $r$ rows in at least one band**:

1. Probability that all $r$ rows in a single band match: $s^r$ (where $s = J(S_A, S_B)$).
2. Probability that at least one row in that band differs: $1 - s^r$.
3. Probability that **none** of the $b$ bands match: $(1 - s^r)^b$.
4. Probability of colliding as a candidate duplicate pair:

$$
P_{\text{collision}}(s) = 1 - (1 - s^r)^b
$$

Let us evaluate $P_{\text{collision}}(s)$ for $b = 20, r = 5$ (threshold $\tau \approx (1/20)^{1/5} \approx 0.549$):
* **Unrelated Documents ($s = 0.20$)**: $P_{\text{collision}}(0.20) = 1 - (1 - 0.20^5)^{20} = 1 - (0.99968)^{20} \approx \mathbf{0.0064}$ ($0.6\%$ false positive rate).
* **Near-Duplicate Clones ($s = 0.80$)**: $P_{\text{collision}}(0.80) = 1 - (1 - 0.80^5)^{20} = 1 - (1 - 0.32768)^{20} = 1 - (0.67232)^{20} \approx \mathbf{0.9996}$ ($99.96\%$ detection rate).

---

## Interactive Chinchilla & Over-Training Simulator

Compare the three historical scaling regimes below under an identical pre-training compute budget ($C = 3.4 \times 10^{23}\text{ FLOPs}$) to observe how shifting from parameter-heavy to token-heavy allocation transforms serving VRAM and decode throughput:

{{< visual/data-curation-interactive >}}

---

## Synthetic Data & The Model Collapse Boundary

As organic human text is exhausted, frontier pipelines increasingly incorporate synthetic data (math reasoning traces, code unit tests, textbook rewrites). However, training generation $t+1$ naively on the unfiltered output distribution of generation $t$ triggers **Model Collapse** (Shumailov et al. 2024).

If each generation fits a Gaussian distribution $\mathcal{N}(\mu_t, \sigma_t^2)$ to $M$ finite samples drawn from the previous model, the variance evolves as a random walk that converges to zero:

$$
\mathbb{E}[\sigma_{t+1}^2] \to 0 \quad \text{as } t \to \infty, \qquad \operatorname{Var}(\mu_t) \propto \sum_{k=1}^t \frac{\sigma_0^2}{M_k}
$$

Low-probability tail events vanish first, collapsing the model into repetitive mode-seeking outputs. Modern pipelines prevent collapse through **Verifier-Guided Filtering**: synthetic candidates are only admitted into the training mix if they pass deterministic external verifiers (compiler execution, Lean 4 proof checkers, or unit test suites) while always preserving a fixed anchor proportion $\alpha > 0$ of grounded human text.

---

## Comparing Frontier Open & Proprietary Corpora

| Corpus / Pipeline | Scale (Tokens) | Deduplication Strategy | Quality Classifier & Filtering |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu** (HuggingFace 2024) | $15\text{T}$ ($1.3\text{T}$ Edu) | Per-dump MinHash LSH ($5\text{-gram}$) | LLaMA-3-70B educational rubric annotations distilled into a fast linear classifier |
| **Dolma v1.7** (AI2 OLMo) | $3.0\text{T}$ | Bloom filter exact + paragraph span dedup | Rule-based Gopher/C4 heuristics + fastText perplexity thresholds |
| **DataComp-LM (DCLM)** | $2.6\text{T}$ | Intra-crawl + cross-crawl MinHash | FastText classifier trained on OpenHermes-2.5 & ELI5 high-score examples |
| **LLaMA 3 Pre-Training** | $15.6\text{T}$ | Multi-level URL, line, and MinHash LSH | Distilled LLaMA-2 quality scorers + 4× code & mathematical annealing in final phase |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2203.15556" title="Training Compute-Optimal Large Language Models (Chinchilla)" subtitle="Hoffmann et al. (NeurIPS 2022) • Established the 1:1 parameter-to-token scaling law and parametric loss decomposition." >}}
  {{< card link="https://arxiv.org/abs/2406.17557" title="The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" subtitle="Penedo et al. (2024) • Systematic ablation of MinHash deduplication and educational classifier filtering at 15T tokens." >}}
  {{< card link="https://arxiv.org/abs/2305.17493" title="The Curse of Recursion: Training on Generated Data Makes Models Forget" subtitle="Shumailov et al. (Nature 2024) • Proved how recursive synthetic training erodes tail distributions and induces model collapse." >}}
  {{< card link="https://arxiv.org/abs/2406.11794" title="DataComp-LM: In Search of the Next Generation of Training Sets" subtitle="Li et al. (2024) • Demonstrated that model-based quality classifiers outperform heuristic filters by over 6% on MMLU." >}}
{{< /cards >}}
