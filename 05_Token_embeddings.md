# Token Embeddings

ALse called "Vector embeddings" or "Word embeddings". The problem with using random no's.
"cat" -> 34
"book" -> 2.9
"tablet" -> -20
"itte" -> "13"

-> Cat and kitten are semantically related. However the associated number cannot capture their relation. One hot encoding fails to get semantic relationship.
Semantically similar words should have similar vectors.

IDs are meaningless.

Example

```
Dog = 12
Cat = 53
```

Numbers themselves have no semantic meaning.

Embedding layer converts

```
12

↓

Dense vector
```

Example

```
12

↓

[0.31,
-0.18,
0.91]
```

---

# Embedding Layer

```python
Embedding(
vocab_size,
embedding_dimension
)
```

Learns during training.

Similar words become nearby vectors.

---

# Positional Embeddings

Transformers don't understand order.

Need position information.

Sentence

```
Dog bites man
```

vs

```
Man bites dog
```

Same words

Different meaning.

---

Position Embedding

Position

0

↓

Vector

Position

1

↓

Another vector

---

Final Input Embedding

```
Input Embedding

=

Token Embedding

+

Position Embedding
```

This gives

```
Meaning

+

Position
```

which is fed into the Transformer.

---

# Complete Pipeline

Raw Text

↓

Tokenizer

↓

Tokens

↓

Vocabulary

↓

Token IDs

↓

Sliding Window

↓

Dataset

↓

DataLoader

↓

Token Embedding

+

Position Embedding

↓

Transformer

↓

Predict Next Token

---

# Key Takeaways

✔ Tokenization converts text into tokens.

✔ Vocabulary maps tokens to integer IDs.

✔ Reverse vocabulary converts IDs back to text.

✔ Unknown words are handled using `<|unk|>` in simple tokenizers.

✔ GPT uses Byte Pair Encoding instead of `<|unk|>`.

✔ `<|endoftext|>` separates unrelated documents.

✔ Sliding windows generate many training examples.

✔ DataLoader batches training data efficiently.

✔ Token embeddings learn semantic meaning.

✔ Positional embeddings encode word order.

✔ Transformer receives:

```
Token Embedding
+
Position Embedding
```

not raw text.

---




## What are Token Embeddings?

A token embedding converts a token ID into a dense vector representation
that captures semantic meaning. 


### Why?

Token IDs are only integers:

``` text
Dog → 23
Cat → 31
Apple → 1
Banana → 38
```

The embedding layer maps them to vectors that the model can learn from.

## Similar words have similar vectors

During training, related words become close together in embedding space.

  Dog       Cat
  --------- ---------
  Similar   Similar

  Apple     Banana
  --------- ---------
  Similar   Similar

## How embeddings are created

1.  Randomly initialize the embedding matrix.
2.  Train with backpropagation.
3.  Optimize vectors during LLM training.

## Embedding Matrix

GPT-2 Small:

-   Vocabulary: **50,257**
-   Embedding dimension: **768**

Matrix shape:

``` text
50,257 × 768
```

Every row stores one token's vector.

## Word2Vec

Word2Vec learned 300-dimensional vectors from Google's news dataset and
introduced the idea that similar words have similar vectors.

## Embedding Layer vs Linear Layer

  Embedding Layer   Linear Layer
  ----------------- -----------------------
  Lookup            Matrix multiplication
  Efficient         Less efficient

------------------------------------------------------------------------

# --- Positional Embeddings

## Why positional information?

The same token receives the same embedding regardless of position.

Example:

``` text
"The cat sat."

"The cat slept."
```

The model therefore needs extra positional information.

## Two types

``` text
Positional Embeddings
├── Absolute
└── Relative
```

### Absolute

->For each postion in the input sequence , a unique embedding is added to the tokens embedding to convey it's exact location.
Toen embdeeings + Postional embeddings =>Input embeddings
The positional vectors have the same dimensions as the original token embeddings.

Each position has its own learnable embedding.

``` text
Input Embedding =
Token Embedding +
Position Embedding
```

Example shapes:

``` text
Token:    (8 × 256)
Position: (8 × 256)
Output:   (8 × 256)
```

PyTorch uses broadcasting to perform this efficiently.

### Relative

-> The emphasis is on the relative postion or distance between the tokens. The model learns the relationship in terms of "how far apart" rahter than exact postion.

The model learns distances between tokens instead of fixed positions.

Example:

``` text
cat is two tokens away from sat
```

This often generalizes better to long sequences.

## Which does GPT use?

GPT models use **learnable absolute positional embeddings** that are
optimized during training.

## Complete Pipeline

``` text
Token IDs
   ↓
Token Embedding
   +
Position Embedding
   ↓
Input Embedding
   ↓
Transformer Layers
```

## Key Takeaways

-   Token embeddings capture semantic meaning.
-   Similar words receive similar vectors.
-   Embedding matrices are learned during training.
-   GPT-2 Small uses a 50,257 × 768 embedding matrix.
-   Positional embeddings inject word-order information.
-   Absolute embeddings encode fixed positions.
-   Relative embeddings encode distances.
-   GPT uses learnable absolute positional embeddings.