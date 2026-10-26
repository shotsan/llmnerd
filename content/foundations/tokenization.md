---
title: Tokenization
toc: true
type: docs
math: true
weight: 1
---

Human language exists as discrete, variable-length text strings governed by syntax, grammar, and morphology. Deep neural networks, conversely, are differentiable mathematical functions that operate strictly on continuous numerical vectors. The bridge between these two representations is **tokenization**.

The design of every tokenizer is governed by a fundamental tradeoff in sequence modeling:

1. **Character-Level Representation**: Guarantees zero out-of-vocabulary (OOV) errors because every text can decompose into individual Unicode characters. However, it inflates sequence length by an order of magnitude. Because self-attention compute and memory scale quadratically ($\mathcal{O}(N^2)$) with sequence length $N$, processing text character-by-character imposes a massive computational tax on every attention layer.
2. **Word-Level Representation**: Keeps sequence lengths short and preserves obvious semantic units. However, human vocabulary is practically open-ended: proper names, technical jargon, slang, and typos are infinite. A fixed word vocabulary requires enormous embedding matrices ($V \times d$) and inevitably fails when encountering unseen words at inference time.

**Subword tokenization** resolves this dilemma. By treating words as compositional sequences of high-frequency subword chunks, common words remain single tokens, rare words break down into familiar morphemes, and sequence length remains computationally efficient.

---

## How Tokenization Works

Modern tokenizers transform raw text into integer tensors through three distinct phases:

```mermaid
flowchart LR
    subgraph S1["1. Character & Byte Ingestion"]
        direction TB
        A["<b>Raw Text Input</b><br/><code>&quot;understanding&quot;</code>"]
        --> B["<b>Pre-Tokenization</b><br/>Whitespace & byte split"]
    end

    subgraph S2["2. Subword Segmentation"]
        direction TB
        C["<b>BPE Merge Engine</b><br/>Greedy pair frequency rules"]
        --> D["<b>Subword Sequence</b><br/><code>[&quot; under&quot;,&nbsp;&quot;stand&quot;,&nbsp;&quot;ing&quot;]</code>"]
    end

    subgraph S3["3. Numerical Vectorization"]
        direction TB
        E["<b>Vocabulary Lookup</b><br/><code>[2104,&nbsp;3812,&nbsp;278]</code>"]
        --> F["<b>Model Input Tensor</b><br/>Embedding coordinate lookup"]
    end

    S1 --> S2 --> S3
```

---

## The BPE Algorithm Walkthrough

To see how Byte-Pair Encoding (BPE) constructs its vocabulary, consider building a tokenizer from scratch on a miniature training corpus consisting of four words:

$$
\mathcal{D} = \{ \text{"low": 5}, \quad \text{"lower": 2}, \quad \text{"newest": 6}, \quad \text{"widest": 3} \}
$$

### 1. Initial Character Vocabulary
We append a special end-of-word suffix `</w>` to each word to preserve word boundary information across merges. We split each word into individual character symbols:

* `l o w </w>` (frequency: 5)
* `l o w e r </w>` (frequency: 2)
* `n e w e s t </w>` (frequency: 6)
* `w i d e s t </w>` (frequency: 3)

The base vocabulary $V_0$ contains all unique symbols:
$$
V_0 = \{ \text{'d'}, \text{'e'}, \text{'i'}, \text{'l'}, \text{'n'}, \text{'o'}, \text{'r'}, \text{'s'}, \text{'t'}, \text{'w'}, \text{'</w>'} \} \quad (|V_0| = 11)
$$

### 2. Counting Symbol Pairs
We scan the corpus and count all adjacent pairs of symbols:

| Adjacent Symbol Pair | Occurrences Across Corpus | Total Frequency |
| :--- | :--- | :--- |
| `('e', 's')` | $(6 \times 1) + (3 \times 1)$ | **9** |
| `('s', 't')` | $(6 \times 1) + (3 \times 1)$ | **9** |
| `('l', 'o')` | $(5 \times 1) + (2 \times 1)$ | 7 |
| `('o', 'w')` | $(5 \times 1) + (2 \times 1)$ | 7 |
| `('w', 'e')` | $(6 \times 1)$ | 6 |
| `('n', 'e')` | $(6 \times 1)$ | 6 |
| `('w', '</w>')` | $(5 \times 1)$ | 5 |

Both `('e', 's')` and `('s', 't')` share the top frequency of 9. Breaking ties deterministically, we select `('e', 's')` to merge:

* **Merge Rule 1**: `('e', 's')` $\to$ `'es'`
* **Corpus state**:
  * `l o w </w>` (5)
  * `l o w e r </w>` (2)
  * `n e w es t </w>` (6)
  * `w i d es t </w>` (3)

### 3. Merging Most Frequent Pairs
Re-evaluating adjacent pair frequencies in the updated corpus:
* `('es', 't')`: appears $6 + 3 = \mathbf{9}$ times.
* `('l', 'o')`: appears $5 + 2 = 7$ times.

The top pair is `('es', 't')`:
* **Merge Rule 2**: `('es', 't')` $\to$ `'est'`
* **Corpus state**:
  * `l o w </w>` (5)
  * `l o w e r </w>` (2)
  * `n e w est </w>` (6)
  * `w i d est </w>` (3)

Next, `('l', 'o')` and `('o', 'w')` both occur 7 times:
* **Merge Rule 3**: `('l', 'o')` $\to$ `'lo'`
* **Merge Rule 4**: `('lo', 'w')` $\to$ `'low'`
* **Corpus state**:
  * `low </w>` (5)
  * `low e r </w>` (2)
  * `n e w est </w>` (6)
  * `w i d est </w>` (3)

### 4. Tokenizing Unseen Words
Suppose the model encounters the previously unseen word **`"lowest"`**:
1. Split into base characters: `['l', 'o', 'w', 'e', 's', 't', '</w>']`
2. Check merge rulebook in chronological training order:
   * Rule 1 applies: `['l', 'o', 'w', 'es', 't', '</w>']`
   * Rule 2 applies: `['l', 'o', 'w', 'est', '</w>']`
   * Rule 3 applies: `['lo', 'w', 'est', '</w>']`
   * Rule 4 applies: `['low', 'est', '</w>']`
3. Final token output: **`["low", "est</w>"]`**

Despite `"lowest"` never appearing in the training corpus, the tokenizer decomposes it into two semantically meaningful root morphemes without throwing an unknown token error.

---

## Interactive BPE Simulator

Step through each merge below to watch the vocabulary grow and see how `"lowest"` is segmented:

{{< visual/bpe-interactive >}}

---

## Python Implementation

Here is an educational implementation of Byte-Pair Encoding in pure Python:

```python
import collections

def get_stats(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """Counts frequencies of adjacent symbol pairs across the corpus."""
    pairs = collections.defaultdict(int)
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pairs[(word_tuple[i], word_tuple[i + 1])] += freq
    return pairs

def merge_vocab(pair: tuple[str, str], v_in: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """Merges all occurrences of a specific symbol pair in the vocabulary."""
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word_tuple in v_in:
        word_str = " ".join(word_tuple)
        new_word_str = word_str.replace(bigram, replacement)
        v_out[tuple(new_word_str.split())] = v_in[word_tuple]
    return v_out

# Example usage:
corpus = {
    ("l", "o", "w", "</w>"): 5,
    ("l", "o", "w", "e", "r", "</w>"): 2,
    ("n", "e", "w", "e", "s", "t", "</w>"): 6,
    ("w", "i", "d", "e", "s", "t", "</w>"): 3,
}

num_merges = 4
for i in range(num_merges):
    pairs = get_stats(corpus)
    best_pair = max(pairs, key=pairs.get)
    corpus = merge_vocab(best_pair, corpus)
    print(f"Merge {i + 1}: {best_pair} -> {''.join(best_pair)}")
```

---

## Practical Edge Cases & Realities

### Byte-Level BPE & No Out-of-Vocabulary Tokens
Early BPE models operated on Unicode characters. If an input contained an unencountered emoji, rare mathematical symbol, or character from an unsupported script, the tokenizer emitted the destructive unknown token `[UNK]`.

Modern frontier tokenizers (Tiktoken for GPT-4, LLaMA 3, Gemma) eliminate `[UNK]` completely using **Byte-Level BPE**:
* The base vocabulary is initialized with all **256 raw bytes** ($0\text{x}00$ through $0\text{xFF}$) of the UTF-8 specification.
* Because every Unicode symbol decomposes into 1 to 4 UTF-8 bytes, any input string is representable without exceptions.

```
Raw Input: "✨" (Sparkles Emoji)
UTF-8 Hex:  0xE2 0x9C 0xA8
Fallback:   Byte(226), Byte(156), Byte(168) -> Always representable!
```

### Numbers & Digit Splitting
Standard BPE trained on raw text merges numbers haphazardly:
* `1234` might tokenize as `["12", "34"]`
* `12345` might tokenize as `["123", "45"]`
* `123456` might tokenize as `["1", "234", "56"]`

This unpredictable chunking forces attention layers to waste capacity inferring decimal alignment before performing arithmetic. Modern models enforce regex pre-tokenization rules that split every individual digit into its own token (`split_digits: true`), stabilizing mathematical reasoning.

### Multilingual Fertility & Compute Cost
Tokenizer vocabularies trained predominantly on English corpora dedicate most entries to English morphemes.
* In English, **fertility** (tokens per character) is roughly **0.25 to 0.30** (~4 characters per token).
* In non-Latin scripts (Hindi, Arabic, Japanese, Telugu), text frequently splits into individual bytes, causing fertility to surge to **0.80 to 1.20** tokens per character.

This means non-English prompts consume 3x to 4x more tokens for the exact same semantic content, driving up API costs, inference latency, and memory consumption.

---

## Comparing Modern Tokenizers

| Algorithm | Used By | Optimization Objective | Boundary Handling |
| :--- | :--- | :--- | :--- |
| **Byte-Pair Encoding (BPE)** | GPT-4, LLaMA 3 | Co-occurrence frequency ($\max \text{count}(p_1, p_2)$) | Regex pre-tokenization with byte fallback |
| **WordPiece** | BERT | Likelihood of unigram language model | Explicit prefix marking (e.g. `##ing`) |
| **SentencePiece / Unigram** | T5, Mistral | Prunes seed vocabulary via unigram loss | Treats whitespace as a standard symbol (` `) |

---

## Key Papers & References

{{< cards >}}
  {{< card link="https://arxiv.org/abs/1508.07909" title="Neural Machine Translation of Rare Words with Subword Units" subtitle="Sennrich, Haddow, Birch (ACL 2016) • Introduced BPE to NLP, solving the open-vocabulary translation problem." >}}
  {{< card link="https://arxiv.org/abs/1808.06226" title="SentencePiece: A Simple and Language Independent Subword Tokenizer" subtitle="Kudo & Richardson (EMNLP 2018) • Direct raw text tokenization without language-specific pre-tokenizers." >}}
  {{< card link="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" title="Language Models are Unsupervised Multitask Learners (GPT-2)" subtitle="Radford et al. (OpenAI 2019) • Introduced Byte-level BPE, guaranteeing 100% token coverage over all UTF-8 strings." >}}
{{< /cards >}}
