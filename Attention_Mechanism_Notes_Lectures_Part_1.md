# LLM From Scratch --- Attention Mechanism Notes (Lectures 12--14)

Clean GitHub-friendly notes.

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
