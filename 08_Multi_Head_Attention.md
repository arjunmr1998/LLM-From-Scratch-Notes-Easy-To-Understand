# Multi-Head Attention

>
> Main idea: Instead of running **one self-attention mechanism**, Transformers run **multiple self-attention mechanisms in parallel**, each learning different relationships in the same input sequence.

---

# Why Multi-Head Attention?

The term **Multi-Head** refers to dividing the attention mechanism into **multiple independent heads**.

Each head:

- Has its own **Query weight matrix**
- Has its own **Key weight matrix**
- Has its own **Value weight matrix**
- Computes attention independently.
- Produces its own context vectors.

Finally, all heads are combined together to produce one final output.

This makes the model much better at learning different types of relationships simultaneously.

---

# Intuition

A single attention head can only learn one way of looking at a sentence.

Multiple heads allow the model to learn different viewpoints at the same time.

For example, in the sentence

> "The cat sat on the mat because it was warm."

Different heads may learn:

- One head tracks grammatical relationships.
- One head tracks subject-object relationships.
- One head connects "warm" with "mat".
- One head captures positional relationships.

Instead of one understanding, the model gets multiple complementary understandings.

---

# Overall Architecture

```text
Embedded Input Tokens (X)
            │
            ▼
    ┌────────────────────┐
    │   Query Weights     │ WQ
    └────────────────────┘
            │
            ▼
         Queries

    ┌────────────────────┐
    │    Key Weights      │ WK
    └────────────────────┘
            │
            ▼
           Keys

    ┌────────────────────┐
    │   Value Weights     │ WV
    └────────────────────┘
            │
            ▼
          Values
```

Unlike single-head attention:

- the **embedded input tokens remain unchanged**
- only different learned projections are created.

---

# Instead of One Matrix

Single-head attention uses

- one WQ
- one WK
- one WV

Multi-head attention uses many.

```text
Head 1:
WQ₁  WK₁  WV₁

Head 2:
WQ₂  WK₂  WV₂

Head 3:
WQ₃  WK₃  WV₃

...

Each head learns independently.
```

---

# Complete Workflow

```text
                 Input X
                    │
      ┌─────────────┼─────────────┐
      │             │             │
      ▼             ▼             ▼
     WQ            WK            WV
      │             │             │
      ▼             ▼             ▼
      Q             K             V
      │             │             │
      └──── Split into Multiple Heads ────┐
                                          │
                                          ▼
                              Scaled Dot-Product
                                  Attention
                                          │
                             Causal Mask (GPT)
                                          │
                                   Softmax
                                          │
                                   Dropout
                                          │
                                          ▼
                                Context Vectors
                                          │
                                 Combine Heads
                                          │
                                          ▼
                                  Final Output
```

---

# Main Idea Written Simply

The attention mechanism is executed **multiple times in parallel**.

Instead of multiplying the input once by one weight matrix, we multiply it by multiple learned weight matrices.

Every head learns a different linear projection of the same input.

---

# Running Attention in Parallel

```text
Input X

├── WQ₁ → Query₁
├── WQ₂ → Query₂
├── WQ₃ → Query₃
└── ...

Each query belongs to a different attention head.
```

The same happens for Keys and Values.

---

# Example

Suppose

- Batch size = 1
- Tokens = 3
- Embedding dimension = 6

Input shape

```text
(1,3,6)
```

Example tensor

```python
X = torch.tensor([[
    [1.,2.,3.,4.,5.,6.],
    [6.,5.,4.,3.,2.,1.],
    [0.,1.,0.,1.,0.,1.]
]])
```

Here

- batch = 1
- tokens = 3
- embedding dimension = 6

---

# Step 1 — Decide Output Dimension

Choose

- `d_out = 6`
- `num_heads = 2`

Each head receives

<math value="head\\_dim=\\frac{d_{out}}{num\\_heads}=\\frac{6}{2}=3"/>

So every head works with **3-dimensional vectors**.

---

# Step 2 — Initialize Trainable Matrices

Create three trainable matrices.

```text
WQ
WK
WV
```

Each has shape

```text
(6×6)
```

because

```text
input_dim = 6
output_dim = 6
```

These matrices are learned during training.

---

# Step 3 — Compute Queries, Keys and Values

Multiply the input by each matrix.

<math value="Q=XW_Q"/>

<math value="K=XW_K"/>

<math value="V=XW_V"/>

Shapes

| Tensor | Shape |
|---------|--------|
| Input | (1,3,6) |
| Query | (1,3,6) |
| Key | (1,3,6) |
| Value | (1,3,6) |

---

# Step 4 — Split Into Heads

Current shape

```text
(1,3,6)
```

Split the last dimension.

```text
(1,3,6)
        │
        ▼
(1,3,2,3)
```

Meaning

```text
(batch,
 tokens,
 heads,
 head_dim)
```

Example

```text
(1,3,2,3)
```

Each token now contains

```text
Head1 → 3 values
Head2 → 3 values
```

instead of one six-dimensional vector.

---

# Step 5 — Group By Heads

Transpose

```text
(1,3,2,3)
      │
      ▼
(1,2,3,3)
```

Now the dimensions become

```text
(batch,
 heads,
 tokens,
 head_dim)
```

Each head now contains every token.

This makes attention computation much easier.

---

# Step 6 — Attention Scores

Every head performs

<math value="QK^T"/>

The Key matrix is transposed.

```text
Query
(1,2,3,3)

×

Keyᵀ
(1,2,3,3)

↓

Attention Scores
(1,2,3,3)
```

Every token compares itself with every other token inside the same head.

---

# What Does One Score Mean?

Suppose

```text
ω₂₁
```

This means

> How much should Token 2 attend to Token 1?

Each row asks

> "Where should I look?"

Each column answers

> "I contain this information."

---

# Step 7 — Scale the Scores

Before Softmax divide every score by

<math value="\\sqrt{head\\_dim}"/>

Here

<math value="\\sqrt3"/>

Formula

<math value="\\frac{QK^T}{\\sqrt{d_k}}"/>

This is why it is called

# Scaled Dot-Product Attention

---

# Why Divide by sqrt(head_dim)?

Two important reasons.

## Reason 1 — Stable Softmax

Large dot products produce huge exponentials.

Example

```text
Softmax(40,38,35)
```

becomes almost

```text
(1,0,0)
```

This creates saturated gradients.

Scaling prevents this.

---

## Reason 2 — Stable Variance

As vector dimension grows,

the dot product variance also grows.

Dividing by

<math value="\\sqrt{d_k}"/>

keeps variance approximately constant.

Training becomes much more stable.

---

# Step 8 — Apply Causal Mask

GPT uses **causal attention**.

Future tokens must not be visible.

Mask

```text
1.0   -∞   -∞

0.4   0.9  -∞

0.2   0.3  0.8
```

Everything above the diagonal becomes

```text
-∞
```

instead of zero.

Why?

Because after Softmax,

<math value="e^{-\\infty}=0"/>

Future tokens receive zero attention.

No information leakage occurs.

---

# Efficient Masking Order

Instead of

```text
Attention Scores
      │
Softmax
      │
Mask
```

Transformers do

```text
Attention Scores
      │
Add -∞ Mask
      │
Softmax
```

This is more efficient.

---

# Step 9 — Softmax

Softmax converts scores into probabilities.

Formula

<math value="\\text{Softmax}(x_i)=\\frac{e^{x_i}}{\\sum_j e^{x_j}}"/>

Properties

- all weights are positive
- rows sum to one
- masked values become zero.

---

# Numerical Stability Trick

PyTorch actually computes

<math value="\\frac{e^{x_i-\\max(x)}}{\\sum e^{x_j-\\max(x)}}"/>

Subtracting the maximum prevents

- overflow
- underflow

without changing the result.

---

# Step 10 — Dropout

Dropout is commonly applied

after computing attention weights.

Purpose

- prevents overfitting
- improves generalization

Common locations

1. after Softmax
2. after multiplying Values.

---

# Step 11 — Compute Context Vectors

Now multiply

```text
Attention Weights

×

Value Matrix
```

Formula

<math value="\\text{Context}=\\text{Attention}\\times V"/>

Shapes

```text
Attention
(1,2,3,3)

×

Values
(1,2,3,3)

↓

Context
(1,2,3,3)
```

Each head produces its own context vectors.

---

# Step 12 — Reformat

Current shape

```text
(1,2,3,3)
```

Transpose

```text
(1,3,2,3)
```

Now tokens are grouped together again.

---

# Step 13 — Combine Heads

Flatten

```text
(1,3,2,3)
      │
      ▼
(1,3,6)
```

The outputs from every head are concatenated.

This becomes the final output of Multi-Head Attention.

---

# Tensor Shape Summary

| Stage | Shape |
|--------|--------|
| Input | (1,3,6) |
| Q,K,V | (1,3,6) |
| Split | (1,3,2,3) |
| Transpose | (1,2,3,3) |
| Attention Scores | (1,2,3,3) |
| Context | (1,2,3,3) |
| Merge | (1,3,6) |

---

# Complete Flow Diagram

```text
Input X
(1,3,6)

      │

      ▼

Q=XWQ
K=XWK
V=XWV

      │

      ▼

Split Heads
(1,3,2,3)

      │

      ▼

Transpose
(1,2,3,3)

      │

      ▼

QKᵀ

      │

      ▼

Scale by √head_dim

      │

      ▼

Add Causal Mask

      │

      ▼

Softmax

      │

      ▼

Dropout

      │

      ▼

Attention × Values

      │

      ▼

Context

      │

      ▼

Transpose Back

      │

      ▼

Flatten

      │

      ▼

Final Output
(1,3,6)
```

---

# Why Multiple Heads Are Better

A single head tries to capture every relationship using one representation.

Multiple heads divide the learning process.

Different heads can simultaneously learn

- grammar
- long-range dependencies
- positional relationships
- semantic meaning
- object interactions
- contextual references

This parallel representation learning is one of the biggest reasons why Transformers outperform older RNN-based architectures.

---

# Final Pipeline (Everything Together)

```text
Input Embeddings
      │
      ▼
Linear Projections (WQ, WK, WV)
      │
      ▼
Split into Multiple Heads
      │
      ▼
QKᵀ
      │
      ▼
Scale by √dk
      │
      ▼
Add Causal Mask
      │
      ▼
Softmax
      │
      ▼
Dropout
      │
      ▼
Attention × Values
      │
      ▼
Context per Head
      │
      ▼
Transpose Back
      │
      ▼
Concatenate Heads
      │
      ▼
Final Multi-Head Attention Output
```

---