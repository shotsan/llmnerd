---
title: Alignment & Post-Training
toc: true
type: docs
math: true
weight: 2
---

A base language model trained purely on next-token cross-entropy loss ($\mathcal{L}_{\text{CE}} = -\sum_t \log P_\theta(x_t \mid x_{<t})$) is a statistical distribution simulator, not an aligned assistant. Given a prompt like *"How do I fix a memory leak in C?"*, a pre-trained base model is just as likely to continue with five more unanswered forum questions as it is to output a solution.

**Post-training and alignment** reshape the base distribution $\pi_{\text{base}}$ into a controllable policy $\pi_\theta$ through three distinct mathematical stages:

1. **Supervised Fine-Tuning (SFT)**: Behavioral cloning on curated `(prompt, response)` demonstrations to establish formatting and instruction following, producing the reference anchor $\pi_{\text{ref}}$.
2. **Preference Optimization (RLHF / DPO)**: Shifting probability mass away from plausible-sounding errors ($y_l$) and toward preferred completions ($y_w$) under a KL-divergence trust region.
3. **Reinforcement Learning with Verifiable Rewards (RLVR / GRPO)**: Training directly against deterministic math/code verifiers to induce multi-step chain-of-thought self-correction.

---

## From Bradley-Terry Rewards to Closed-Form DPO

### 1. The Bradley-Terry Preference Model
Given a prompt $x$ and two candidate completions $(y_w, y_l)$ where human or verifier evaluation prefers $y_w \succ y_l$, the Bradley-Terry model defines the probability of this pairwise preference via a latent scalar reward function $r_\phi(x, y)$:

$$
P(y_w \succ y_l \mid x) = \sigma\!\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) = \frac{1}{1 + \exp\!\left(-(r_\phi(x, y_w) - r_\phi(x, y_l))\right)}
$$

### 2. The KL-Regularized RLHF Objective
In classical Reinforcement Learning from Human Feedback (Ouyang et al. 2022), we train a policy $\pi_\theta$ to maximize expected reward while penalizing reverse KL-divergence from the frozen reference policy $\pi_{\text{ref}}$:

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)} \left[ r_\phi(x, y) \right] - \beta \, D_{\mathrm{KL}}\!\left( \pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x) \right)
$$

### 3. Direct Preference Optimization (DPO) Derivation
Instead of optimizing this objective with unstable online Proximal Policy Optimization (PPO), Rafailov et al. (2023) observed that the KL-constrained maximization has an **exact analytical solution** $\pi^*(y \mid x)$:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left( \frac{1}{\beta} r(x, y) \right)
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)$ is the partition function. Solving for the reward $r(x, y)$ algebraically yields:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

When we substitute this expression into the Bradley-Terry preference formula $r(x, y_w) - r(x, y_l)$, the intractable partition function $\beta \log Z(x)$ **cancels out identically**, giving the closed-form DPO loss:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

{{< visual/alignment-blueprint >}}

---

## Numerical Walkthrough: DPO Margin & Gradient Weighting

To see how DPO updates token probabilities without a reward model, consider a single preference pair $(x, y_w, y_l)$ with KL temperature $\beta = 0.20$:

1. **Sequence Log-Probabilities under $\pi_\theta$ and $\pi_{\text{ref}}$**:
   * Winning response $y_w$: $\log \pi_\theta(y_w \mid x) = -12.0$, $\quad \log \pi_{\text{ref}}(y_w \mid x) = -14.0$
   * Losing response $y_l$: $\log \pi_\theta(y_l \mid x) = -11.0$, $\quad \log \pi_{\text{ref}}(y_l \mid x) = -10.5$

2. **Log-Ratio Rewards ($\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$)**:
   * $\hat{r}_\theta(x, y_w) = 0.20 \times (-12.0 - (-14.0)) = 0.20 \times (+2.0) = \mathbf{+0.40}$
   * $\hat{r}_\theta(x, y_l) = 0.20 \times (-11.0 - (-10.5)) = 0.20 \times (-0.5) = \mathbf{-0.10}$

3. **Implicit Preference Margin & Loss**:
   * Margin $\Delta \hat{r}_\theta = 0.40 - (-0.10) = \mathbf{+0.50}$
   * Preference probability: $\sigma(0.50) = \frac{1}{1 + e^{-0.50}} \approx \mathbf{0.622}$
   * DPO Loss: $\mathcal{L}_{\text{DPO}} = -\log(0.622) \approx \mathbf{0.474\text{ nats}}$

4. **Per-Example Gradient Weight**:
   Taking the gradient with respect to $\theta$:
   $$
   \nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \, \underbrace{\sigma\!\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right)}_{1 - 0.622 = \mathbf{0.378}} \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]
   $$
   Notice that the factor $\sigma(-\Delta \hat{r}_\theta) = 0.378$ automatically scales the gradient in proportion to how strongly the current policy **misranks** the pair.

---

## Interactive Reward Hacking & Goodhart Simulator

In online RL, the reward model $R_{\text{proxy}}$ is only an imperfect statistical proxy for true human preference $R_{\text{gold}}$. Gao et al. (2023) established that as policy drift $d = \sqrt{D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})}$ grows, true utility follows an inverted-U curve:

$$
R_{\mathrm{proxy}}(d) = \alpha d, \qquad R_{\mathrm{gold}}(d) = \alpha d - \gamma d^2
$$

Use the simulator below to observe how pushing $D_{\mathrm{KL}}$ past the optimal frontier causes the policy to game length and sycophancy biases:

{{< visual/alignment-interactive >}}

---

## Group Relative Policy Optimization (GRPO)

For mathematical and code reasoning (DeepSeek-R1), maintaining a separate Value Critic $V_\psi$ of the same size as the policy doubles GPU memory consumption. **Group Relative Policy Optimization (GRPO)** (Shao et al. 2024) eliminates the critic entirely by sampling a group of $G$ outputs $\{o_1, o_2, \dots, o_G\}$ from $\pi_{\theta_{\text{old}}}$ for each prompt $q$ and computing the baseline directly from the group's reward statistics:

$$
\hat{A}_{i, t} = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

When paired with deterministic rule-based verifiers ($r_i \in \{0, 1\}$ for exact math answer or passing unit tests), GRPO avoids neural reward model hacking while enabling emergent self-verification and backtracking.

---

## Comparing Post-Training Algorithms

| Algorithm | Models in VRAM | Rollout Mode | Advantage / Margin Mechanism | Primary Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **RLHF (PPO)** | 4 ($\pi_\theta, \pi_{\text{ref}}, r_\phi, V_\psi$) | Online sampling | Generalized Advantage Estimation (GAE) via $V_\psi$ | Critic instability & high VRAM footprint |
| **DPO** | 2 ($\pi_\theta, \pi_{\text{ref}}$) | Offline pairs | Closed-form log-ratio margin $\beta \Delta \log(\pi_\theta/\pi_{\text{ref}})$ | Likelihood displacement on out-of-distribution tokens |
| **SimPO** | 1 ($\pi_\theta$ only) | Offline pairs | Length-normalized average log-prob $\frac{\beta}{|y|}\log \pi_\theta(y\mid x) - \gamma$ | Sensitive to target margin $\gamma$ tuning |
| **GRPO** | 2 ($\pi_\theta, \pi_{\text{ref}}$) | Online group ($G$) | Group z-score normalization $(r_i - \mu_G)/\sigma_G$ | Zero gradient when all $G$ samples fail ($r_i = 0$) |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2203.02155" title="Training language models to follow instructions with human feedback (InstructGPT)" subtitle="Ouyang et al. (NeurIPS 2022) • Established the three-stage SFT, Reward Model, and KL-PPO alignment pipeline." >}}
  {{< card link="https://arxiv.org/abs/2305.18290" title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model" subtitle="Rafailov et al. (NeurIPS 2023) • Derived the closed-form Bradley-Terry policy mapping, eliminating the PPO critic and reward model." >}}
  {{< card link="https://arxiv.org/abs/2210.10760" title="Scaling Laws for Reward Model Overoptimization" subtitle="Gao et al. (ICML 2023) • Quantified Goodhart's Law in RLHF as a function of square-root KL divergence." >}}
  {{< card link="https://arxiv.org/abs/2501.12948" title="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" subtitle="DeepSeek-AI (2025) • Demonstrated emergent reasoning and self-verification via large-scale GRPO on verifiable rewards." >}}
{{< /cards >}}
