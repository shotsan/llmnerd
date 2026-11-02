---
title: Evaluation & Contamination
toc: true
type: docs
math: true
weight: 4
---

Evaluating frontier language models is fundamentally harder than evaluating classical supervised classifiers. Because an autoregressive language model is trained on trillions of web tokens and outputs open-ended sequences, static accuracy metrics break down under three failure modes: **tokenizer dependence** in likelihood metrics, **combinatorial variance** in multi-sample generation, and **benchmark contamination** where test problems leak into the pre-training corpus.

Rigorous model evaluation requires combining information-theoretic likelihood normalization, unbiased combinatorial estimators ($\operatorname{Pass@}k$), pairwise Bradley-Terry human/judge rankings, and log-probability outlier auditing ($\operatorname{Min-K\%}$) to separate genuine out-of-distribution generalization from verbatim memorization.

{{< visual/evaluation-blueprint >}}

---

## 1. Likelihood Evaluation & Bits-Per-Byte Normalization

At the pre-training layer, an autoregressive model $P_{\theta}$ is evaluated by its cross-entropy loss over a held-out sequence $X = (x_1, x_2, \dots, x_T)$ of $T$ tokens. Exponentiating the average negative log-likelihood (measured in natural base $e$) yields **Perplexity** ($\operatorname{PPL}$):

$$\operatorname{PPL}(X) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P_{\theta}(x_t \mid x_{<t})\right)$$

Perplexity represents the effective branching factor of the model at each step: a perplexity of $8.0$ indicates that the model is as uncertain on average as if it were choosing uniformly among $8$ equiprobable tokens.

### The Tokenizer Fertility Trap

Perplexity is normalized by the **token count** $T$, which makes $\operatorname{PPL}$ incomparable across models that use different vocabularies $|\mathcal{V}|$. Suppose Model A uses a $32\text{K}$ BPE vocabulary that segments a document of $B_{\text{UTF8}} = 1{,}000$ UTF-8 bytes into $T_A = 250$ tokens, whereas Model B uses a $128\text{K}$ vocabulary (e.g., LLaMA-3) that compresses the exact same document into $T_B = 200$ tokens. Even if both models assign the exact same total log-probability $\sum_t \log P(x_t \mid x_{<t}) = -400\text{ nats}$ to the document:

$$\operatorname{PPL}_A = \exp\!\left(\frac{400}{250}\right) = e^{1.60} \approx 4.95, \qquad \operatorname{PPL}_B = \exp\!\left(\frac{400}{200}\right) = e^{2.00} \approx 7.39$$

Model B appears to have worse perplexity purely because each of its tokens spans more characters. To compare models across tokenizers, we normalize total negative log-likelihood by the raw UTF-8 byte length $B_{\text{UTF8}}$ and convert from natural logs to base-$2$ bits, yielding **Bits-Per-Byte ($\operatorname{BPB}$)**:

$$\operatorname{BPB}(X) = -\frac{1}{B_{\text{UTF8}} \ln 2}\sum_{t=1}^{T}\log P_{\theta}(x_t \mid x_{<t}) = \left(\frac{T}{B_{\text{UTF8}} \ln 2}\right)\ln\operatorname{PPL}(X)$$

For both models above, $\operatorname{BPB}(X) = \frac{400}{1000 \times 0.6931} \approx 0.577\text{ bits/byte}$, restoring invariance to vocabulary size.

---

## 2. Functional Synthesis & Unbiased Pass@k Estimation

For reasoning, mathematics, and code synthesis (e.g., HumanEval, SWE-bench), likelihood does not measure whether generated code compiles and passes unit tests. Instead, we evaluate functional correctness across $k$ independent stochastic rollouts sampled at temperature $\tau > 0$: a problem is solved under $\operatorname{Pass@}k$ if **at least one** of the $k$ generated candidates passes all unit tests.

Naively drawing exactly $k$ samples per problem and checking if any succeed produces high variance. Conversely, if we generate $n \ge k$ total candidate rollouts (e.g., $n = 20$ or $n = 200$) and observe that $c \in \{0, \dots, n\}$ candidates pass the test suite, estimating $\operatorname{Pass@}k$ via the plug-in empirical success rate $1 - (1 - c/n)^k$ is **upwardly biased** for finite $n$.

Following Chen et al. (2021), the unique minimum-variance unbiased estimator (MVUE) for $\operatorname{Pass@}k$ draws $k$ samples without replacement from the $n$ generated rollouts and computes the exact hypergeometric complement—one minus the probability that all $k$ selected rollouts belong to the $n - c$ failing candidates:

$$\operatorname{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1}\left(1 - \frac{c}{n - i}\right)$$

Whenever $n - c < k$, by the pigeonhole principle every subset of size $k$ contains at least one passing solution, so $\binom{n-c}{k} = 0$ and $\operatorname{Pass@}k = 1$.

{{< visual/evaluation-interactive >}}

---

## 3. Worked Numerical Walkthrough: Pass@k vs. Plug-In Bias

Suppose we evaluate a code synthesis model on a benchmark task by generating $n = 20$ independent rollouts at temperature $\tau = 0.8$, and exactly $c = 4$ rollouts pass the unit tests.

### Step 1: Comparing Unbiased Pass@5 to Naive Plug-In

The empirical single-sample pass rate is $\hat{p} = c/n = 4/20 = 0.20$. The naive plug-in approximation assumes sampling with replacement:

$$\widehat{\operatorname{Pass@}5}_{\text{naive}} = 1 - (1 - \hat{p})^5 = 1 - (0.80)^5 = 1 - 0.32768 = 67.23\%$$

Now evaluate the exact unbiased hypergeometric estimator using the product identity to avoid factorial overflow:

$$\operatorname{Pass@}5 = 1 - \frac{\binom{16}{5}}{\binom{20}{5}} = 1 - \left(\frac{16}{20} \times \frac{15}{19} \times \frac{14}{18} \times \frac{13}{17} \times \frac{12}{16}\right)$$

$$\operatorname{Pass@}5 = 1 - \frac{4{,}368}{15{,}504} = 1 - 0.28173 = 71.83\%$$

Here, the naive plug-in estimator misestimates the finite-sample combinatorial probability by $4.60$ percentage points.

### Step 2: Modeling Contamination Inflation

Suppose a competing model has weaker true generalization ($c = 3$ out of $n = 20$, so true $\operatorname{Pass@}1_{\text{clean}} = 15.0\%$), but $\alpha_{\text{leak}} = 30\%$ of the benchmark problems were inadvertently included in its web pre-training corpus and memorized verbatim ($\operatorname{Pass@}1_{\text{leaked}} = 100\%$). Its reported aggregate $\operatorname{Pass@}1$ across the full benchmark is:

$$\operatorname{Pass@}1_{\text{reported}} = (1 - \alpha_{\text{leak}})\operatorname{Pass@}1_{\text{clean}} + \alpha_{\text{leak}}(1.0) = 0.70(0.15) + 0.30(1.0) = 40.5\%$$

Despite having lower true capability ($15.0\%$ vs. $20.0\%$), the contaminated model outranks the clean model by $+20.5\%$ on the static leaderboard—a classic **benchmark mirage**.

---

## 4. Pairwise Arena Ranking & LLM-as-a-Judge Biases

For open-ended instruction following, summarization, and dialogue where no deterministic unit test exists, evaluation relies on pairwise preference comparisons $(A \succ B)$ collected from blind human annotators (Chatbot Arena) or frontier judge models (LLM-as-a-Judge).

### Bradley-Terry Maximum Likelihood Scaling

Given $M$ models with latent skill parameters $\boldsymbol{\xi} = (\xi_1, \dots, \xi_M) \in \mathbb{R}^M$, the **Bradley-Terry model** parameterizes the probability that Model $i$ beats Model $j$ via the logistic link function. In standard Elo scaling (base $10$, scale factor $400$), rating $R_i$ satisfies:

$$P(i \succ j) = \frac{1}{1 + 10^{(R_j - R_i)/400}}$$

Unlike classical sequential Elo updates—which depend on match ordering—modern leaderboards (Chiang et al., 2024) fit all ratings simultaneously by maximizing the regularized pairwise log-likelihood over the empirical win matrix $W_{ij}$ (number of times $i$ beat $j$), yielding exact bootstrap confidence intervals:

$$\hat{\mathbf{R}} = \arg\max_{\mathbf{R}} \sum_{i=1}^{M}\sum_{j \neq i} W_{ij} \log\!\left(\frac{1}{1 + 10^{(R_j - R_i)/400}}\right)$$

A rating difference of $\Delta R = R_A - R_B = +100$ Elo points corresponds to an expected win rate of $P(A \succ B) = (1 + 10^{-100/400})^{-1} = (1 + 10^{-0.25})^{-1} \approx 64.0\%$.

### Systematic Judge Biases & Debiasing Protocols

When replacing human raters with an LLM judge (e.g., MT-Bench, AlpacaEval), three statistical biases distort win rates unless explicitly controlled:

| Judge Bias Mode | Empirical Manifestation | Mathematical / Experimental Mitigation |
| :--- | :--- | :--- |
| **Position Bias** | Judges systematically favor the first response $(A, B)$ over $(B, A)$ by $5\text{--}12\%$ | Evaluate both permutations $(A, B)$ and $(B, A)$; count a win only if consistent across both orders |
| **Length / Verbosity Bias** | Longer, markdown-formatted responses win even when factually redundant | Length-Controlled AlpacaEval (Dubois et al., 2024): fit a generalized linear model (GLM) with response length as a covariate and predict win rate at zero length delta |
| **Self-Preference Bias** | Models assign higher scores to outputs matching their own low-perplexity style | Use an ensemble of heterogeneous judge families and reference-guided grading rubrics |

---

## 5. Benchmark Contamination & Min-K% Detection

Because pre-training corpora scrape billions of GitHub repositories, ArXiv papers, and web forums, static benchmarks (GSM8K, MMLU, HumanEval) inevitably leak into training sets. Decontamination requires both **pre-training corpus filtering** and **post-hoc black-box detection**.

1. **N-Gram & Embedding Decontamination (Corpus Filtering):** During corpus curation (LLaMA-3, GPT-4), any training document sharing a $13$-gram (or $10$-gram for short prompts) with any evaluation test example is stripped or masked. However, $n$-gram matching fails against **paraphrased contamination**, translated benchmarks, or synthetic rewrites of test questions.
2. **Min-K% Prob (Post-Hoc Memorization Audit):** Shi et al. (2024) observe that if a model has memorized a test example $X = (x_1, \dots, x_T)$, even its rarest, highest-surprisal tokens will receive anomalously high log-probabilities compared to an unseen text of similar domain. Let $\ell_t = \log P_{\theta}(x_t \mid x_{<t})$ denote the token log-probabilities, and let $\operatorname{Min-K\%}(X)$ be the subset $E$ containing the $K\%$ tokens with the **lowest** log-probabilities (typically $K = 20$). The detection score is:

$$\operatorname{Min-K\%}(X) = \frac{1}{|E|}\sum_{x_t \in E} \log P_{\theta}(x_t \mid x_{<t})$$

If $\operatorname{Min-K\%}(X) > \gamma$ (where threshold $\gamma$ is calibrated on guaranteed post-cutoff text), the example is flagged as contaminated. Moving beyond static datasets, **dynamic temporal benchmarks** such as **LiveCodeBench** continuously harvest new Competitive Programming problems published *after* each model's training cutoff date, providing a contamination-free verification signal.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2107.03374" title="Evaluating Large Language Models Trained on Code (Codex / HumanEval)" subtitle="Chen et al. (2021) • Formulated functional synthesis evaluation and the unbiased hypergeometric Pass@k estimator." >}}
  {{< card link="https://arxiv.org/abs/2403.04132" title="Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" subtitle="Chiang et al. (ICML 2024) • Established pairwise Bradley-Terry maximum likelihood Elo scaling over blind human crowdsourcing." >}}
  {{< card link="https://arxiv.org/abs/2310.17680" title="Detecting Pretraining Data from Large Language Models (Min-K% Prob)" subtitle="Shi et al. (ICLR 2024) • Reference-free contamination detection using bottom-K% outlier token log-probabilities." >}}
  {{< card link="https://arxiv.org/abs/2403.07974" title="LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code" subtitle="Jain et al. (2024) • Temporal cutoff isolation using continuously harvested post-cutoff competitive programming problems." >}}
{{< /cards >}}
