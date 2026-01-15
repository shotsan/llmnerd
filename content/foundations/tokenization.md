---
title: Tokenization
toc: true
---

How text is converted into tokens that LLMs process.

## Overview

Tokenization is the first step in the LLM pipeline, converting raw text into discrete tokens.

## Key Concepts

### Tokenization Strategies

#### Byte-Pair Encoding (BPE)
- Most common approach (GPT, LLaMA)
- Merges frequent byte pairs iteratively

#### WordPiece
- Used by BERT
- Similar to BPE with greedy longest-match first

#### SentencePiece
- Language-agnostic
- Handles raw text without pre-tokenization

### Vocabulary Size
- Typical sizes: 32K, 64K, 100K tokens
- Trade-off between compression and expressivity
