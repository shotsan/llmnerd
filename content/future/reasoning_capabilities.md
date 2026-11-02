---
title: Test-Time Compute & Reasoning Scaling
toc: true
type: docs
math: true
weight: 2
---

For the first five years of the foundation model era, scaling laws focused almost exclusively on **pre-training FLOPs** ($C_{\text{train}} \approx 6ND$): to make a model smarter, researchers trained larger parameter matrices $N$ on larger token corpora $D$ while keeping inference computation fixed at a single greedy forward pass per output token.

However, a single forward pass through an $L$-layer Transformer has a fixed **computational circuit depth** $\mathcal{O}(L)$. Problems in theorem proving, competitive programming, and multi-hop planning require compositional search depths that exceed any fixed-depth feedforward circuit. Modern reasoning models (OpenAI **o1/o3**, **DeepSeek-R1**, **Qwen-QwQ**) unlock a second orthogonal scaling axis—**Test-Time Compute Scaling**—by dynamically allocating serial reasoning tokens and parallel tree expansions during inference.

{{< visual/reasoning-blueprint >}}

---

## 1. Latent Chain-of-Thought & Serial Circuit Depth

Given a complex problem $x$ and target answer $y$, a standard autoregressive model attempts to approximate $P_\theta(y \mid x)$ in $\mathcal{O}(L)$ sequential matrix layers. **Chain-of-Thought (CoT)** introduces a variable-length sequence of intermediate reasoning tokens $z = (s_1, s_2, \dots, s_H) \in \mathcal{V}^*$ before emitting $y$, marginalizing over latent deduction trajectories:

$$P_\theta(y \mid x) = \sum_{z \in \mathcal{Z}} P_\theta(y \mid z, x) P_\theta(z \mid x)$$

From a circuit complexity perspective (Feng et al. 2023; Merrill & Sabharwal 2024), a constant-depth Transformer restricted to direct output generation lies within the weak parallel complexity class $\mathsf{TC}^0$ and cannot solve inherently sequential problems such as boolean formula evaluation, permutation composition, or dynamic programming of length $T > L$. By writing intermediate states $s_h$ back into the context window and attending over them at step $h+1$, an autoregressive model with $H$ reasoning tokens expands its effective sequential circuit depth from $\mathcal{O}(L)$ to $\mathcal{O}(H \cdot L)$—achieving **Turing completeness** polynomially bounded by test-time token budget $H$.

---

## 2. Parallel Verification: Majority Voting, ORMs, and PRMs

When we scale test-time compute in **parallel** by sampling $N$ candidate reasoning trajectories $\{z^{(1)}, \dots, z^{(N)}\} \sim \pi_\theta(\cdot \mid x)$ at temperature $\tau > 0$, how do we select the correct final answer $\hat{y}$?

### 1. Self-Consistency Majority Voting (Unweighted Mode)
Wang et al. (2023) marginalize over the sampled reasoning chains $z^{(i)}$ by taking the empirical plurality vote over their extracted final answers $y^{(i)} = \operatorname{ans}(z^{(i)})$:

$$\hat{y}_{\text{SC}} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{I}\!\left(\operatorname{ans}(z^{(i)}) = y\right)$$

**Failure Mode:** Majority voting succeeds only when the correct answer is the unique mode of the model's answer distribution. On hard Olympiad problems where the model's true pass rate is $p_0 = 10\%$ and a common algebraic trap attracts $35\%$ of rollouts, increasing $N \to \infty$ causes Majority Voting to converge with $100\%$ certainty to the **wrong** answer.

### 2. Outcome Reward Models (ORM) vs. Process Reward Models (PRM)
Instead of unweighted voting, a learned verifier scores candidate trajectories:
* **Outcome Reward Model (ORM):** Assigns a single scalar score $R_{\text{ORM}}(x, z, y) \in [0, 1]$ to the complete trajectory based solely on whether the final answer is correct.
* **Process Reward Model (PRM, Lightman et al. 2023):** Decomposes a reasoning trace $z = (s_1, \dots, s_H)$ into $H$ discrete deduction steps and predicts the probability that each step $s_h$ is logically valid given the prefix $z_{<h}$, aggregating across the chain via the product or minimum rule:

$$\mathcal{R}_{\text{PRM}}(x, z_{1:H}) = \prod_{h=1}^{H} P_\phi\!\left(\text{valid}_h \mid x, s_1, \dots, s_h\right), \qquad \text{or} \qquad \min_{1 \le h \le H} P_\phi\!\left(\text{valid}_h \mid x, s_{\le h}\right)$$

Because a PRM localizes the exact step $h^*$ where a sign error or hallucinated lemma occurs, it solves the **credit assignment bottleneck** and guides stepwise **Beam Search** or **Monte Carlo Tree Search (MCTS)** via the PUCT selection rule:

$$a^* = \arg\max_{a} \left[ Q(s, a) + c_{\text{puct}} \cdot \pi_\theta(a \mid s) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$

{{< visual/reasoning-interactive >}}

---

## 3. Worked Numerical Walkthrough: ORM False-Positive Trap vs. PRM Pruning

Consider a $3$-step mathematical derivation $z = (s_1, s_2, s_3)$ where the model generates two competing candidate proofs $z^{(A)}$ and $z^{(B)}$:

* **Candidate A (Flawed Proof, Lucky Cancellation):** Step $s_1^{(A)}$ is valid ($P_{\text{valid}} = 0.95$), Step $s_2^{(A)}$ divides by zero ($P_{\text{valid}} = 0.08$), and Step $s_3^{(A)}$ guesses a plausible-looking integer answer ($P_{\text{valid}} = 0.85$).
* **Candidate B (Rigorous Proof):** Step $s_1^{(B)}$ ($0.92$), Step $s_2^{(B)}$ ($0.90$), and Step $s_3^{(B)}$ ($0.91$) are all algebraically sound.

Let us compare how an ORM vs. a PRM ranks the two trajectories:

1. **ORM Evaluation:** Because Candidate A ends with a confident, well-formatted boxed integer that superficially matches training distribution heuristics, an imperfect ORM assigns $R_{\text{ORM}}(z^{(A)}) = \mathbf{0.89} > R_{\text{ORM}}(z^{(B)}) = 0.86$, selecting the **flawed proof**.
2. **PRM Product & Minimum Evaluation:**
   $$\mathcal{R}_{\text{PRM}}^{\text{prod}}(z^{(A)}) = 0.95 \times 0.08 \times 0.85 = \mathbf{0.0646}, \qquad \mathcal{R}_{\text{PRM}}^{\text{min}}(z^{(A)}) = \min(0.95, 0.08, 0.85) = \mathbf{0.080}$$
   $$\mathcal{R}_{\text{PRM}}^{\text{prod}}(z^{(B)}) = 0.92 \times 0.90 \times 0.91 = \mathbf{0.7535}, \qquad \mathcal{R}_{\text{PRM}}^{\text{min}}(z^{(B)}) = \min(0.92, 0.90, 0.91) = \mathbf{0.900}$$
   By penalizing the single invalid transition at $s_2^{(A)}$ ($0.08$), the PRM rejects the lucky hallucination by an $11.6\times$ margin.

### Concrete PUCT Tree Expansion Step
Suppose at node $s_1$, the MCTS tree has visited parent $s_1$ a total of $N(s_1) = 16$ times ($\sqrt{16} = 4$) with exploration constant $c_{\text{puct}} = 1.5$:
* **Branch $a_1$ (Exploited path):** $Q(s_1, a_1) = 0.75$, prior $\pi_\theta(a_1 \mid s_1) = 0.50$, visit count $N(s_1, a_1) = 14$.
  $$\operatorname{PUCT}(s_1, a_1) = 0.75 + 1.5(0.50)\frac{4}{1 + 14} = 0.75 + 0.20 = \mathbf{0.950}$$
* **Branch $a_2$ (Under-explored promising path):** $Q(s_1, a_2) = 0.60$, prior $\pi_\theta(a_2 \mid s_1) = 0.30$, visit count $N(s_1, a_2) = 2$.
  $$\operatorname{PUCT}(s_1, a_2) = 0.60 + 1.5(0.30)\frac{4}{1 + 2} = 0.60 + 0.60 = \mathbf{1.200}$$
PUCT selects **Branch $a_2$** ($1.200 > 0.950$), systematically balancing PRM value estimates against policy exploration.

---

## 4. Reinforcement Learning with Verifiable Rewards (RLVR & DeepSeek-R1)

Training separate PRMs and running external MCTS trees at inference time introduces massive serving complexity. **DeepSeek-R1** (DeepSeek-AI, 2025) and OpenAI **o1** demonstrated that a model can **internalize tree search into a single linear autoregressive stream** using pure **Reinforcement Learning with Verifiable Rewards (RLVR)**.

Using **Group Relative Policy Optimization (GRPO)**, for each math or code problem $x$, we sample a group of $G$ full reasoning-and-answer trajectories $\{y_1, \dots, y_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$ and score each output using a **deterministic rule-based verifier** $R(x, y_i) \in \{0, 1\}$ (exact symbolic math match or unit-test compilation), with zero neural reward model to hack:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}_{i=1}^G}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\!\left(\rho_{i,t}\hat{A}_i,\, \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\right]$$

where $\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}$ and $\hat{A}_i = \frac{R(x, y_i) - \operatorname{mean}(\mathbf{R})}{\operatorname{std}(\mathbf{R})}$.

### Emergent Reasoning Behaviors ("The Aha Moment")
Without any supervised demonstrations of how to backtrack, RLVR optimization spontaneously discovers that trajectories containing **in-context verification, hypothesis testing, and self-correction** achieve higher expected reward $R(x, y)$. As training progresses, average `<think>` length scales from hundreds to tens of thousands of tokens, effectively compiling an MCTS search tree into a linearized stream of natural language thoughts.

---

## 5. Latent Continuous Reasoning (Coconut)

Must a model verbalize every intermediate reasoning step as a discrete English word in $\mathcal{V}$? Most words in a natural language CoT trace are syntactic glue rather than high-entropy logical decisions. **Coconut (Chain of Continuous Thought, Hao et al. 2024)** replaces discrete token decoding inside the `<think>` block by feeding the model's last-layer continuous hidden state $\mathbf{h}_t^{(L)} \in \mathbb{R}^d$ directly back as the next input embedding $\mathbf{e}_{t+1} = \mathbf{h}_t^{(L)}$.

Because a continuous vector $\mathbf{h}_t^{(L)} \in \mathbb{R}^{4096}$ can represent a **superposition of multiple candidate search branches simultaneously**, latent continuous reasoning performs breadth-first graph exploration in a fraction of the serial steps required by discrete vocabulary tokens.

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/2305.20050" title="Let's Verify Step by Step (Process Reward Models)" subtitle="Lightman et al. (ICLR 2024) • Proved that step-level PRM supervision significantly outperforms Outcome Reward Models on MATH." >}}
  {{< card link="https://arxiv.org/abs/2408.03314" title="Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" subtitle="Snell et al. (2024) • Established compute-optimal inference scaling laws across parallel verification and sequential revision." >}}
  {{< card link="https://arxiv.org/abs/2501.12948" title="DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" subtitle="DeepSeek-AI (2025) • Demonstrated emergent self-verification, backtracking, and test-time scaling via pure GRPO on verifiable rewards." >}}
  {{< card link="https://arxiv.org/abs/2412.06769" title="Training Large Language Models to Reason in a Continuous Latent Space (Coconut)" subtitle="Hao et al. (2024) • Replaced discrete CoT tokens with recurrent continuous hidden states enabling breadth-first superposition search." >}}
{{< /cards >}}
