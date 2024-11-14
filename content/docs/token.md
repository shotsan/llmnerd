---
title: Token
type: docs
prev: docs/_index.md
math: true
next: docs/token
---

### What is a token
The fundamental unit of a language is a token, its a way to represent building elements of words and/or sentences. Irrespective of the language of choice or one research group or another, it might differ. 
For example, we may be think of English words as token, or chunks of words as tokens. There might be a wide range of choices, although not quite relevant at this point in time, its better to keep Token definition simple and clean.

### What to do with tokens
We need to establish relationship between the tokens to develop a language model. Through embeddings we create relationships across tokens. Simply, an embedding is a vector, and using simple Euclidian or $L^2$ norm, we can 
establish distances or metric to measure how *related* two tokens are.

### Creating an Embedding
Probably, the most important step towards developing a language model is creating an embedding. An embedding is a vector representation of a token.
Ideally, upon applying a distance metric on an embedding, similar words or words with similar context must be closer. 

