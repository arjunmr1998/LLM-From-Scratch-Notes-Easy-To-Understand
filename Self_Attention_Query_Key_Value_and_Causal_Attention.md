# LLM From Scratch --- Self-Attention with Query, Key & Value 



## Lecture 15 --- Scaled Dot-Product Attention

### Goal

Convert input embeddings into richer context embeddings using trainable
Query, Key and Value matrices.

### Trainable matrices

-   W_Q → Query
-   W_K → Key
-   W_V → Value

For an input embedding matrix X:

-   Q = XW_Q
-   K = XW_K
-   V = XW_V

### Attention scores

Unscaled attention:

QK\^T

Scaled attention:

QK\^T / sqrt(d_k)

### Why divide by sqrt(d_k)?

1.  Prevents Softmax from becoming overly peaked.
2.  Keeps dot-product variance stable as dimensions grow.

### Softmax

Softmax converts scores into attention weights that sum to one.

### Context vectors

Context = Attention Weights × Value Matrix

Pipeline:

1.  Input embeddings
2.  Q, K, V
3.  Attention scores
4.  Scale
5.  Softmax
6.  Multiply by V
7.  Context vectors

### Database analogy

-   Query: what the current token is looking for.
-   Key: information used for matching.
-   Value: information retrieved after matching.

------------------------------------------------------------------------

##  --- Causal Self-Attention

Causal attention (masked self-attention) allows a token to attend only
to previous and current tokens.

Future tokens are hidden.

### Efficient masking

Instead of masking after Softmax:

1.  Compute attention scores.
2.  Add a triangular mask with negative infinity above the diagonal.
3.  Apply Softmax.

Because exp(-inf)=0, future tokens receive zero attention.

### Dropout

Dropout improves generalization by randomly ignoring some values during
training.

Common locations:

-   after attention weights
-   after multiplying attention weights with value vectors

### Key Takeaways

-   W_Q, W_K and W_V are learned projection matrices.
-   Scaling by sqrt(d_k) stabilizes training.
-   Softmax produces attention weights.
-   Context vectors are weighted sums of value vectors.
-   Causal masking prevents information leakage from future tokens.
