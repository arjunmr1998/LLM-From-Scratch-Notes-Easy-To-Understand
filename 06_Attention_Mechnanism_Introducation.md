# Attention Mechanism 


Why was Attention introduced?
Before Transformers, machine translation relied on Encoder--Decoder
RNNs.
These models worked well for short sentences but struggled with long
sequences because the decoder depended on a single hidden state from the
encoder.
Attention solved this by allowing the decoder to dynamically focus on
relevant input tokens while generating each output token.
Four Types of Attention
Type                        Purpose
---
Simplified Self-Attention   Basic intuition
Self-Attention              Learns relationships within one sequence
Causal Attention            Prevents looking at future tokens
Multi-Head Attention        Learns multiple relationships simultaneously
Why Word-by-Word Translation Fails
German:
> Kannst du mir helfen, diesen Satz zu übersetzen?
Correct English:
> Can you help me translate this sentence?
Translation requires context and grammar rather than literal
word-by-word replacement.
Encoder--Decoder Architecture
``` text
Input Sentence
      │
      ▼
   Encoder
      │
      ▼
 Hidden State
      │
      ▼
   Decoder
      │
      ▼
Output Sentence
```
The encoder processes the sentence sequentially and builds a hidden
representation.
The decoder generates the output using that representation.
RNN Limitation
``` text
Word1 → Hidden1
          │
Word2 → Hidden2
          │
Word3 → Hidden3
```
As sentences become longer, earlier information is compressed and may be
lost.
How Attention Helps
Instead of relying on one hidden state, the decoder can revisit every
input token.
``` text
Input Tokens
A  B  C  D  E
│  │  │  │  │
└──┴──┼──┴──┘
      ▼
    Decoder
```
The model learns attention weights that determine which words matter
most.
Dynamic Focus
At each decoding step the model shifts its focus.
Example:
Step 1 → Subject
Step 2 → Verb
Step 3 → Object
Self-Attention
Self-attention allows every token to attend to every other token in the
same sequence.
Example:
"The cat sat on the mat."
While processing sat, the model can attend to The, cat,
on, and mat.
Why is it called Self?
The sequence attends to itself.
``` text
A ↔ B
A ↔ C
A ↔ D
B ↔ C
...
```
Traditional Attention vs Self-Attention
Traditional Attention   Self-Attention
---
Encoder ↔ Decoder       One sequence
Two sequences           One sequence
Used in translation     Core of Transformers
Progression to GPT
``` text
RNN
 ↓
Encoder–Decoder
 ↓
Attention
 ↓
Self-Attention
 ↓
Transformer
 ↓
GPT
```
Key Takeaways
Encoder--Decoder RNNs struggled with long sentences.
Attention lets the decoder revisit the entire input sequence.
Attention weights determine token importance.
Self-attention allows every token to interact with every other
token.
Self-attention is the foundation of Transformer-based LLMs such as
GPT.


## Attention Types

1.  Simplified Self-Attention
2.  Self-Attention
3.  Causal Attention
4.  Multi-Head Attention

-   Simplified: no trainable weights.
-   Self-attention: uses trainable weights.
-   Causal: cannot see future tokens.
-   Multi-head: learns multiple relationships.

## Why Attention?

Word-by-word translation fails because grammar and word order differ.

RNN encoder-decoder models compress an entire sentence into one hidden
state, which struggles with long-range dependencies.

## Self-Attention

Every token can attend to every other token.

Example: "Your journey starts with one step."

## Attention Scores

Dot product measures similarity:

score = query · token

Higher score means greater relevance.

## Softmax

Softmax converts scores into probabilities that sum to 1.

PyTorch improves numerical stability by subtracting the maximum score
before exponentiation.

## Context Vector

The context vector is a weighted sum of token embeddings using attention
weights.

## Simplified Pipeline

1.  Token embeddings
2.  Dot-product scores
3.  Softmax
4.  Weighted sum
5.  Context vector

## Why Trainable Weights?

Example: "The cat sat on the mat because it was warm."

The model learns that "warm" should attend more strongly to "mat".

## Key Takeaways

-   Attention lets models revisit relevant tokens.
-   Self-attention compares tokens within the same sequence.
-   Dot products create attention scores.
-   Softmax normalizes them.
-   Context vectors combine information from the whole sequence.
-   Trainable weights improve contextual understanding.
