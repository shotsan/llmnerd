---
title: Continual Learning & Knowledge Updating
toc: true
type: docs
math: true
weight: 5
---

A pre-trained language model is a frozen snapshot of the world up to its training cutoff date. When deployed into production, however, facts change, APIs evolve, and enterprises require domain adaptation without spending millions of GPU hours pre-training from scratch. Attempting to update a foundation model via naive gradient descent on a new dataset $\mathcal{D}_B$ triggers **catastrophic forgetting**: gradient updates overwrite the distributed weight representations that encode general reasoning and prior knowledge $\mathcal{D}_A$.

Solving continual adaptation requires navigating the **stability-plasticity dilemma** through three complementary paradigms: **Fisher-regularized continual learning (EWC)**, **low-rank subspace isolation (LoRA)**, and **architectural routing between non-parametric retrieval (RAG), prefix KV-caching, and localized weight editing (ROME/MEMIT)**.

{{< visual/continual-blueprint >}}

---

## 1. The Stability-Plasticity Dilemma & Elastic Weight Consolidation

From a Bayesian perspective, after training on a prior corpus $\mathcal{D}_A$, learning a new domain $\mathcal{D}_B$ corresponds to maximizing the log-posterior $\log P(\theta \mid \mathcal{D}_A, \mathcal{D}_B)$. Applying Bayes' rule assuming conditional independence of $\mathcal{D}_A$ and $\mathcal{D}_B$ given $\theta$:

$$\log P(\theta \mid \mathcal{D}_A, \mathcal{D}_B) = \log P(\mathcal{D}_B \mid \theta) + \log P(\theta \mid \mathcal{D}_A) - \log P(\mathcal{D}_B \mid \mathcal{D}_A)$$

Here, $-\log P(\mathcal{D}_B \mid \theta) = \mathcal{L}_B(\theta)$ is the empirical cross-entropy loss on the new domain $\mathcal{D}_B$, while the prior posterior $\log P(\theta \mid \mathcal{D}_A)$ summarizes everything learned on Task $A$. Because the exact posterior $P(\theta \mid \mathcal{D}_A)$ is intractable over billions of parameters, Kirkpatrick et al. (2017) apply a **Laplace approximation** around the converged Task $A$ optimum $\theta_A^*$. Since $\nabla_{\theta}\mathcal{L}_A(\theta_A^*) = \mathbf{0}$, a second-order Taylor expansion of $\mathcal{L}_A(\theta)$ around $\theta_A^*$ yields the **Elastic Weight Consolidation (EWC)** objective:

$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2}\sum_{i=1}^{|\theta|} F_i \left(\theta_i - \theta_{A,i}^*\right)^2$$

where $\lambda > 0$ controls the trade-off between stability (preserving Task $A$) and plasticity (fitting Task $B$), and $F_i$ is the $i$-th diagonal entry of the **Fisher Information Matrix** evaluated at $\theta_A^*$:

$$F_i = \mathbb{E}_{x \sim \mathcal{D}_A}\!\left[\left(\frac{\partial \log P_{\theta}(x)}{\partial \theta_i}\right)^2\Bigg|_{\theta = \theta_A^*}\right]$$

By a fundamental identity of information geometry, the Fisher Information Matrix equals the expected second-order Hessian of the negative log-likelihood at the mode. Thus, each parameter $\theta_i$ is anchored to $\theta_{A,i}^*$ by a mechanical spring whose stiffness is proportional to $F_i$: coordinates with high curvature ($F_i \gg 0$) are locked in place, forcing Task $B$ updates into the null-space or flat low-curvature directions ($F_i \approx 0$) of Task $A$.

{{< visual/continual-interactive >}}

---

## 2. Subspace Isolation: Low-Rank Adaptation (LoRA)

Even with Fisher regularization, full-parameter continual fine-tuning requires storing optimizer moments ($m_t, v_t$) and reference weights $\theta_A^*$ across all $N$ parameters. Aghajanyan et al. (2021) and Hu et al. (2022) demonstrated that while pre-training requires high rank across $\mathbb{R}^{d \times k}$, adapting a pre-trained model to a downstream task has a low **intrinsic dimension**.

**Low-Rank Adaptation (LoRA)** freezes the pre-trained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ completely ($\nabla \mathbf{W}_0 = \mathbf{0}$) and parameterizes the weight update $\Delta \mathbf{W}$ as the outer product of two low-rank matrices $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$:

$$\mathbf{h} = \mathbf{W}'\mathbf{x} = \left(\mathbf{W}_0 + \frac{\alpha}{r}\Delta\mathbf{W}\right)\mathbf{x} = \mathbf{W}_0\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$$

1. **Zero-Drift Initialization:** At step $t = 0$, $\mathbf{A}$ is initialized from $\mathcal{N}(0, \sigma^2)$ and $\mathbf{B}$ is initialized to $\mathbf{0}$, guaranteeing $\Delta \mathbf{W} = \mathbf{0}$ so the adapted model starts identically at $\theta_A^*$.
2. **Hyperparameter Decoupling ($\alpha / r$):** Scaling the low-rank bypass by $\alpha / r$ stabilizes the forward activation variance when sweeping rank $r \in \{8, 16, 64\}$ without retuning the Adam learning rate.
3. **Zero Inference Latency & Modular Routing:** Because the update is linear, $\mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ can be folded directly into the base matrix for zero-overhead single-tenant serving, or kept separate to dynamically hot-swap thousands of domain-specific LoRA adapters $(\mathbf{B}_m, \mathbf{A}_m)$ over a single shared frozen base model $\mathbf{W}_0$.

---

## 3. Worked Numerical Walkthrough: EWC Penalty & LoRA Compression

Let us trace both a two-coordinate Fisher-constrained update and the exact memory savings of a LoRA adapter on a $d = 4096$ Transformer layer.

### Part A: Two-Parameter EWC Gradient Step

Suppose a model converges on Task $A$ at $\theta_A^* = (\theta_{A,1}^*, \theta_{A,2}^*) = (1.00, 2.00)$ with diagonal Fisher sensitivities $F_1 = 0.20$ (flat direction) and $F_2 = 50.0$ (stiff core feature). During Task $B$ adaptation at current weights $\theta = (1.50, 2.20)$, suppose the raw Task $B$ loss gradient is $\nabla_{\theta}\mathcal{L}_B(\theta) = (-0.80, -0.80)$ and we set $\lambda = 10$.

The total EWC gradient for coordinate $i$ is $\frac{\partial \mathcal{L}_{\text{EWC}}}{\partial \theta_i} = \frac{\partial \mathcal{L}_B}{\partial \theta_i} + \lambda F_i (\theta_i - \theta_{A,i}^*)$:

$$\frac{\partial \mathcal{L}_{\text{EWC}}}{\partial \theta_1} = -0.80 + 10(0.20)(1.50 - 1.00) = -0.80 + 1.00 = +0.20$$

$$\frac{\partial \mathcal{L}_{\text{EWC}}}{\partial \theta_2} = -0.80 + 10(50.0)(2.20 - 2.00) = -0.80 + 100.0 = +99.20$$

On the stiff coordinate $\theta_2$, the $+100.0$ restorative spring force overwhelms the $-0.80$ Task $B$ pull, snapping $\theta_2$ back toward $\theta_{A,2}^* = 2.00$ while permitting $\theta_1$ to absorb the new domain update.

### Part B: LoRA Parameter & VRAM Compression

Consider a single attention projection matrix $\mathbf{W}_0 \in \mathbb{R}^{4096 \times 4096}$ containing $d \times k = 16{,}777{,}216$ parameters. With rank $r = 16$:

$$|\theta_{\text{LoRA}}| = r(d + k) = 16 \times (4096 + 4096) = 131{,}072\text{ parameters}$$

$$\text{Compression Ratio} = \frac{131{,}072}{16{,}777{,}216} = \frac{1}{128} \approx 0.781\%$$

Because AdamW optimizer states ($8\text{ bytes/param}$ in mixed precision) are allocated only for the $0.781\%$ trainable LoRA weights, optimizer VRAM drops by $128\times$ per adapted matrix.

---

## 4. Direct Knowledge Editing: ROME & MEMIT

When a single discrete fact changes (e.g., *"The Prime Minister of the UK is..."*), neither continual pre-training nor LoRA is targeted enough. Geva et al. (2021) and Meng et al. (2022) established that two-layer MLP blocks inside mid-network Transformer layers act as **linear associative key-value memories**:

$$\mathbf{m}_i^{(l)} = \mathbf{W}_{\text{proj}}^{(l)} \cdot \sigma\!\left(\mathbf{W}_{\text{fc}}^{(l)} \mathbf{h}_i^{(l)}\right) = \mathbf{W}_{\text{proj}}^{(l)} \mathbf{k}_i$$

Here, the intermediate activation $\mathbf{k}^* \in \mathbb{R}^{d_{\text{ff}}}$ produced at the last token of a subject entity acts as a lookup **key**, and the down-projection matrix $\mathbf{W}_{\text{proj}}^{(l)}$ maps $\mathbf{k}^*$ to a **value** vector $\mathbf{v}^*$ encoding the factual attribute.

**Rank-One Model Editing (ROME)** inserts a new factual association $(\mathbf{k}^* \to \mathbf{v}^*)$ while preserving existing associations $\mathbf{W}\mathbf{K} \approx \mathbf{V}$ (where $\mathbf{C} = \mathbf{K}\mathbf{K}^{\top}$ is the uncentered covariance of pre-existing keys) via a constrained least-squares rank-one update:

$$\min_{\hat{\mathbf{W}}} \|\hat{\mathbf{W}}\mathbf{K} - \mathbf{W}\mathbf{K}\|_F^2 \quad \text{subject to} \quad \hat{\mathbf{W}}\mathbf{k}^* = \mathbf{v}^* \implies \hat{\mathbf{W}} = \mathbf{W} + \frac{(\mathbf{v}^* - \mathbf{W}\mathbf{k}^*)\left(\mathbf{C}^{-1}\mathbf{k}^*\right)^{\top}}{\left(\mathbf{C}^{-1}\mathbf{k}^*\right)^{\top}\mathbf{k}^*}$$

**MEMIT** (Meng et al., 2023) generalizes this closed-form normal equation across multiple contiguous MLP layers simultaneously, scaling from single-fact edits to $10{,}000+$ concurrent factual updates without gradient descent.

---

## 5. System Architecture Trade-Offs: RAG vs. Context Caching vs. LoRA

In production systems, updating model knowledge is an architectural choice between **non-parametric external memory** and **parametric weight modification**:

| Architectural Paradigm | Knowledge Update Latency | Per-Query Compute Cost | Hallucination & Citation Audit | Ideal Engineering Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval-Augmented Generation (RAG)** | Milliseconds (vector index upsert + delete) | Low ($\mathcal{O}(K_{\text{top}} \cdot d)$ retrieval + short prompt) | Exact document provenance & deterministic citations | High-churn factual corpora, user-permissioned ACL data, strict compliance |
| **Long-Context KV-Cache Prefix** | Seconds (re-prefill shared corpus KV cache) | Moderate (amortized via prefix KV caching) | High fidelity over medium corpora ($32\text{K}\text{--}1\text{M}$ tokens) | Static codebases, legal/financial books queried repeatedly across a session |
| **LoRA / Continual Fine-Tuning** | Hours to Days (GPU training + eval pipeline) | Zero extra context tokens at inference | Cannot guarantee factual deletion; opaque provenance | Teaching new output schemas, domain DSL syntax, reasoning style, or tone |
| **Model Editing (ROME / MEMIT)** | Seconds (closed-form rank-$r$ matrix update) | Zero extra inference cost | Surgical single-hop edits; brittle on multi-hop reasoning ripples | Patching specific outdated named entities or redacting toxic associations |

> **The Golden Rule of Continual LLM Engineering:** Use **non-parametric retrieval (RAG or Prefix KV-Caching)** to inject *dynamic facts* that require provenance or frequent deletion; use **parametric adaptation (LoRA + Experience Replay)** to teach *new skills, syntax, or domain reasoning patterns*.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1612.00796" title="Overcoming Catastrophic Forgetting in Neural Networks (EWC)" subtitle="Kirkpatrick et al. (PNAS 2017) • Introduced Laplace Fisher Information regularization to anchor parameters critical to prior tasks." >}}
  {{< card link="https://arxiv.org/abs/2106.09685" title="LoRA: Low-Rank Adaptation of Large Language Models" subtitle="Hu et al. (ICLR 2022) • Demonstrated that freezing base weights and training rank-r decomposition matrices matches full fine-tuning at <1% parameters." >}}
  {{< card link="https://arxiv.org/abs/2202.05262" title="Locating and Editing Factual Associations in GPT (ROME)" subtitle="Meng et al. (NeurIPS 2022) • Causal tracing of factual associations to mid-layer MLP modules and closed-form rank-one weight updates." >}}
  {{< card link="https://arxiv.org/abs/2210.07229" title="Mass-Editing Memory in a Transformer (MEMIT)" subtitle="Meng et al. (ICLR 2023) • Scaled closed-form associative memory updates to 10,000+ simultaneous factual edits across MLP layers." >}}
{{< /cards >}}
