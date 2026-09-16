# Creating a Large Language Model (LLM) from Scratch --- Notes

> Beginner-friendly notes on the ideas behind LLMs, transformers,
> GPT-style models, pretraining, and fine-tuning.

## 1. What is a Large Language Model?

A **Large Language Model (LLM)** is a deep neural network designed to
understand, generate, and respond to human-like text.

LLMs are generally:

-   **Large** --- they may contain millions or billions of trainable
    parameters.
-   **Language models** --- they learn statistical patterns in language
    from large amounts of text.
-   **Deep neural networks** --- modern LLMs are usually based on the
    **Transformer architecture**.

LLMs can perform a wide range of natural-language-processing tasks,
including:

-   Question answering
-   Translation
-   Summarization
-   Classification
-   Sentiment analysis
-   Text generation
-   Conversational assistance

A useful high-level idea is:

``` text
Large dataset + Transformer + language-model training → Foundation language model
```

The transformer architecture --- especially its **attention mechanism**
--- is one of the key reasons modern LLMs work so well.

------------------------------------------------------------------------

## 2. Stages of Building an LLM

At a high level, creating a useful LLM involves two major stages:

``` text
Creating an LLM
│
├── Pretraining
│   └── Learn general language patterns from a large, diverse corpus
│
└── Fine-tuning / post-training
    └── Adapt the pretrained model to tasks, instructions, or domains
```

### 2.1 Pretraining

During pretraining, the model learns from a **large and diverse text
dataset**.

Possible data sources include:

-   Books
-   Articles
-   Websites
-   Reference material
-   Other large text corpora

Before training, raw text normally goes through preprocessing such as:

``` text
Raw text
   ↓
Cleaning / preparation
   ↓
Tokenization
   ↓
Training sequences
   ↓
LLM pretraining
```

The resulting model is often called a **pretrained model** or
**foundation model**.

For a GPT-style model, pretraining commonly uses **next-token
prediction**: given previous tokens, predict the next one.

------------------------------------------------------------------------

## 3. Fine-Tuning

Pretraining teaches broad language capabilities. Fine-tuning adapts
those capabilities for a narrower purpose.

``` text
Pretrained LLM
      ↓
Fine-tuning / post-training
      ↓
Task- or instruction-adapted model
```

Examples include:

-   Classification
-   Summarization
-   Translation
-   Personal assistants
-   Domain-specific assistants

### Instruction Fine-Tuning

An instruction dataset contains examples of instructions/prompts and
desired responses.

Example:

``` text
Instruction: Translate "Good morning" into German.
Response: Guten Morgen.
```

Other examples could involve question answering or customer support.

### Classification Fine-Tuning

A classification dataset contains text together with a target label.

Example:

``` text
Email: "Congratulations! Claim your prize now."
Label: Spam
```

``` text
Email: "Please find the meeting notes attached."
Label: Not spam
```

So the important distinction is:

``` text
Instruction tuning → prompt/instruction + desired response
Classification    → input text + class label
```

------------------------------------------------------------------------

## 4. What is a Transformer?

Most modern LLMs are based on the **Transformer**, introduced in the
2017 paper *Attention Is All You Need*.

The original Transformer was designed for sequence-to-sequence tasks
such as machine translation.

For example:

``` text
"This is an example"
        ↓
     Encoder
        ↓
     Decoder
        ↓
"Das ist ein Beispiel"
```

A simplified original Transformer has two major parts:

### Encoder

The encoder converts the input sequence into contextual vector
representations.

``` text
Input text
   ↓
Tokenization
   ↓
Embeddings
   ↓
Encoder blocks
   ↓
Contextual representations
```

### Decoder

The decoder generates an output sequence using encoded information and
previously generated tokens.

``` text
Encoded representation
        +
previous output tokens
        ↓
      Decoder
        ↓
probability distribution over next token
```

------------------------------------------------------------------------

## 5. Attention

The central idea behind transformers is **attention**.

Attention allows the model to determine which tokens are important
relative to other tokens while constructing representations or
generating predictions.

For example, in:

``` text
The animal didn't cross the street because it was tired.
```

the model needs to understand what **"it"** refers to. Attention helps
connect related words even when they are separated in the sequence.

This is important because language often contains **long-range
dependencies**.

------------------------------------------------------------------------

## 6. Self-Attention

**Self-attention** (sometimes called intra-attention) relates different
positions within the **same sequence** to build contextual
representations.

Conceptually:

``` text
Token 1 ─┐
Token 2 ─┼──> Self-Attention ──> Context-aware token representations
Token 3 ─┤
Token 4 ─┘
```

Instead of representing each word independently, the representation of
each token is influenced by other relevant tokens.

This allows the model to capture relationships such as:

-   Subject ↔ verb
-   Pronoun ↔ noun
-   Modifier ↔ object
-   Words separated by many tokens

------------------------------------------------------------------------

## 7. BERT vs GPT

Transformers can be adapted in different ways.

### BERT

**BERT --- Bidirectional Encoder Representations from Transformers**

BERT is based primarily on the **encoder** side of the Transformer.

A key feature is bidirectional context: a token representation can use
information from both the left and right sides of the sequence.

Example:

``` text
"I deposited money in the bank."
"The boat reached the river bank."
```

The surrounding words help the model distinguish the meaning of
**bank**.

BERT was originally pretrained using objectives including **masked
language modeling**, where hidden/masked tokens are predicted.

### GPT

**GPT --- Generative Pre-trained Transformer**

GPT-style models use a **decoder-only Transformer architecture**.

They are autoregressive:

``` text
Previous tokens → predict next token
```

Example:

``` text
Input:  "This is"
Output: "an"

Input:  "This is an"
Output: "example"

Input:  "This is an example"
Output: "..."
```

The newly generated token becomes part of the input context for the next
prediction.

------------------------------------------------------------------------

## 8. Autoregressive Language Modeling

An **autoregressive model** uses previous outputs/tokens as context for
future predictions.

Suppose the sequence is:

``` text
This is an example
```

Training can conceptually create prediction pairs such as:

``` text
"This"             → "is"
"This is"          → "an"
"This is an"       → "example"
```

The model learns a probability distribution:

``` text
P(next token | previous tokens)
```

During generation:

``` text
Prompt
  ↓
Predict next token
  ↓
Append token to context
  ↓
Predict another token
  ↓
Repeat
```

This simple training objective, when combined with huge datasets and
sufficiently capable models, can produce surprisingly broad
capabilities.

------------------------------------------------------------------------

## 9. Self-Supervised Learning

GPT-style pretraining is commonly described as **self-supervised
learning**.

We do not need humans to manually label every sentence. Instead, the
structure of the text provides the learning target.

Given:

``` text
Transformers are useful for language modeling
```

the training process can automatically derive targets from shifted
versions of the same sequence.

Conceptually:

``` text
Input:   Transformers are useful for language
Target:  are useful for language modeling
```

So the data effectively creates its own supervision.

------------------------------------------------------------------------

## 10. Tokens

An LLM does not directly operate on sentences as humans see them.

Text is first converted into **tokens**.

A token can be:

-   A whole word
-   Part of a word
-   Punctuation
-   Another frequently occurring text unit

Example:

``` text
"unbelievable"
```

might be represented as something like:

``` text
["un", "believ", "able"]
```

depending on the tokenizer.

The processing pipeline is roughly:

``` text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
 ↓
Token probabilities
```

------------------------------------------------------------------------

## 11. Zero-Shot and Few-Shot Learning

### Zero-Shot

**Zero-shot** means attempting a task without providing task-specific
examples in the prompt.

Example:

``` text
Classify the sentiment:
"The movie was fantastic."
```

The model may perform the task even without an example demonstrating the
expected answer format.

### Few-Shot

**Few-shot prompting** provides a small number of examples in the
prompt.

Example:

``` text
Text: "I loved the film."
Sentiment: Positive

Text: "The food was terrible."
Sentiment: Negative

Text: "The book was excellent."
Sentiment:
```

The model uses the examples in the context to infer the pattern.

> Note: few-shot prompting is different from updating the model's
> weights. The examples are supplied in the context rather than used as
> a conventional training dataset.

------------------------------------------------------------------------

## 12. Emergent Capabilities

A language model may display useful behaviors beyond the exact surface
form of its training objective.

For example, a GPT-style model is trained primarily to predict the next
token, yet sufficiently capable models can perform tasks such as:

-   Translation
-   Summarization
-   Question answering
-   Classification
-   Text transformation

These capabilities arise because successful next-token prediction
requires learning many patterns about language, concepts, and
relationships represented in the training data.

This phenomenon is often discussed under the broad term **emergent
capabilities/behavior**, although researchers debate exactly how
emergence should be measured and interpreted.

------------------------------------------------------------------------

## 13. Transformers vs LLMs

These terms are related, but they are not interchangeable.

``` text
Transformer = neural-network architecture
LLM         = large model trained primarily on language/text
```

Therefore:

-   Not every Transformer is an LLM.
-   Transformers can also be used for computer vision, audio, multimodal
    systems, and other sequence/modeling tasks.
-   Historically, language models have also used recurrent or
    convolutional architectures.
-   Most state-of-the-art modern LLMs use Transformer-based
    architectures.

------------------------------------------------------------------------

# Building a Small GPT-Style LLM from Scratch

A practical learning project can be organized into the following stages.

``` text
1. Collect text data
        ↓
2. Clean and prepare data
        ↓
3. Build/train a tokenizer
        ↓
4. Convert text to token IDs
        ↓
5. Create training sequences
        ↓
6. Build embeddings
        ↓
7. Implement self-attention
        ↓
8. Implement multi-head attention
        ↓
9. Build Transformer blocks
        ↓
10. Build a decoder-only GPT architecture
        ↓
11. Train using next-token prediction
        ↓
12. Evaluate the model
        ↓
13. Generate text
        ↓
14. Fine-tune / instruction-tune if desired
```

## 14. Data Preparation and Sampling

Start with a text corpus.

For a learning project, this does **not** need to contain billions of
words. A smaller dataset is enough to understand the mechanics.

Typical steps:

``` text
Raw text
 ↓
Cleaning
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Long token stream
 ↓
Fixed-length training windows
```

If the token sequence is:

``` text
[12, 41, 8, 90, 17, 22]
```

a training example might look like:

``` text
Input:  [12, 41, 8, 90, 17]
Target: [41, 8, 90, 17, 22]
```

Every target token is the **next token** corresponding to the input
position.

------------------------------------------------------------------------

## 15. Embeddings

Token IDs are integers, but neural networks need meaningful vector
representations.

An **embedding layer** maps each token ID to a vector:

``` text
Token ID
   ↓
Embedding lookup
   ↓
Dense vector
```

Transformers also need information about token order, because
self-attention alone does not inherently encode sequence position.

Therefore a GPT-style input representation includes token information
plus position information, conceptually:

``` text
Input representation = token embedding + positional information
```

------------------------------------------------------------------------

## 16. Attention Mechanism

A transformer attention layer builds three representations from each
input:

-   **Query (Q)**
-   **Key (K)**
-   **Value (V)**

A simplified scaled dot-product attention equation is:

``` text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Intuition:

``` text
Query → What information am I looking for?
Key   → What information does this token contain / advertise?
Value → What information should be passed forward?
```

For GPT-style causal language modeling, a **causal mask** prevents a
token from attending to future tokens.

``` text
Token 1 → can see token 1
Token 2 → can see tokens 1..2
Token 3 → can see tokens 1..3
...
```

This prevents the model from cheating during next-token training.

------------------------------------------------------------------------

## 17. Multi-Head Attention

Instead of computing only one attention pattern, transformers use
multiple attention heads.

``` text
Input
 ├── Attention Head 1
 ├── Attention Head 2
 ├── Attention Head 3
 └── ...
       ↓
Concatenate
       ↓
Linear projection
```

Different heads can learn different relationships in the data.

------------------------------------------------------------------------

## 18. Transformer Block

A simplified GPT-style Transformer block contains:

``` text
Input
 ↓
Layer Normalization
 ↓
Masked Multi-Head Self-Attention
 ↓
Residual Connection
 ↓
Layer Normalization
 ↓
Feed-Forward Network / MLP
 ↓
Residual Connection
 ↓
Output
```

Many such blocks are stacked:

``` text
Embeddings
   ↓
Transformer Block 1
   ↓
Transformer Block 2
   ↓
...
   ↓
Transformer Block N
```

------------------------------------------------------------------------

## 19. GPT-Style Architecture

A simplified decoder-only GPT model looks like:

``` text
Input text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token + positional representations
   ↓
Transformer blocks
   ↓
Final normalization
   ↓
Linear output layer
   ↓
Logits over vocabulary
   ↓
Next-token probabilities
```

Unlike the original encoder-decoder Transformer, GPT does not need a
separate encoder.

``` text
GPT = decoder-only Transformer language model
```

------------------------------------------------------------------------

## 20. Training Loop

The basic training loop is:

``` text
for each batch:
    input_tokens, target_tokens = batch

    logits = model(input_tokens)

    loss = cross_entropy(logits, target_tokens)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Conceptually:

``` text
Input tokens
    ↓
Model
    ↓
Predicted next-token distributions
    ↓
Compare with true next tokens
    ↓
Loss
    ↓
Backpropagation
    ↓
Update model parameters
```

Repeat this process over many batches.

------------------------------------------------------------------------

## 21. Evaluation

Training loss tells us how well the model predicts the training data,
but we also need a **validation dataset**.

``` text
Dataset
├── Training set
└── Validation set
```

The validation set helps detect overfitting and compare model
configurations.

A common language-model metric is **perplexity**, which is related to
cross-entropy loss.

Lower validation loss/perplexity generally indicates better next-token
prediction on unseen text.

------------------------------------------------------------------------

## 22. Text Generation

Once trained, the model can generate text autoregressively.

``` text
Prompt
  ↓
Model predicts token probabilities
  ↓
Choose/sample one token
  ↓
Append token
  ↓
Run model again
  ↓
Repeat
```

Generation behavior can be controlled using techniques such as:

-   Greedy decoding
-   Temperature
-   Top-k sampling
-   Top-p / nucleus sampling

------------------------------------------------------------------------

# Complete Mental Model

The entire process can be summarized as:

``` text
                        RAW TEXT
                           │
                           ▼
                       TOKENIZER
                           │
                           ▼
                       TOKEN IDs
                           │
                           ▼
              TOKEN + POSITION INFORMATION
                           │
                           ▼
              ┌────────────────────────┐
              │   TRANSFORMER BLOCK    │
              │                        │
              │  Masked Self-Attention │
              │          +             │
              │        MLP             │
              └────────────────────────┘
                           │
                         × N
                           │
                           ▼
                     OUTPUT LOGITS
                           │
                           ▼
                NEXT-TOKEN PROBABILITY
                           │
                           ▼
                     SAMPLE TOKEN
                           │
                           └──────────► repeat
```

During training:

``` text
Prediction
    ↓
Compare against actual next token
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Parameter update

```



# Goal

Before an LLM can understand text, it **cannot work directly with words or sentences**.

Everything must eventually become **numbers**.

The complete pipeline is:

Text
↓
Tokenization
↓
Vocabulary
↓
Token IDs
↓
Embeddings
↓
Transformer

---

# Step 1 - Read Raw Text

```python
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
```

Example

```
Hello world.
```

Initially this is simply one long string.

---

# Step 2 - Tokenization

## What is Tokenization?

Tokenization means

> Breaking text into smaller pieces called **tokens**.

Example

```
Hello, world!
```

becomes

```
["Hello", ",", "world", "!"]
```

Every punctuation mark becomes its own token.

---

# Why Tokenization?

Neural networks only understand numbers.

Words

```
Hello
World
Apple
```

must eventually become

```
41
82
500
```

before training.

---

# Regular Expressions

Python's `re.split()` is used.

Example

```python
re.split(r'(\s)', text)
```

splits on whitespace.

Example

```
Hello world
```

↓

```
["Hello"," ","world"]
```

---

## Split on punctuation too

```python
re.split(r'([,.]|\s)', text)
```

Output

```
Hello,
↓

["Hello",",","world","."]
```

---

## Remove Whitespace

```python
[item for item in result if item.strip()]
```

Result

```
["Hello",",","world","."]
```

instead of

```
["Hello"," ",","," ","world"]
```

---

# Why Remove Spaces?

Advantages

- Less memory
- Faster training
- Simpler tokenizer

However...

Some applications require spaces.

Examples

- Python code
- YAML
- Markdown
- Source code

because spacing carries meaning.

---

# Improved Regular Expression

The tokenizer later becomes

```python
r'([,.:;?_!"()\']|--|\s)'
```

It now handles

- commas
- periods
- quotes
- semicolons
- brackets
- question marks
- double dash

---

# Apply Tokenizer

```python
preprocessed = re.split(...)
```

Output

```
["It's",
"the",
"last",
"he",
"painted",
",",
...]
```

Thousands of tokens are produced.

---

# Step 3 - Create Vocabulary

Vocabulary means

> Every unique token appearing in the dataset.

Example

Dataset

```
I like tea.
I like coffee.
```

Unique words

```
I
like
tea
coffee
.
```

Vocabulary size = 5

---

Create Vocabulary

```python
all_words = sorted(set(preprocessed))
```

Sorting ensures deterministic ordering.

---

# Vocabulary Dictionary

```python
vocab = {
    token : integer
}
```

Example

```
{
"," : 0
"." : 1
"I" : 2
"coffee" : 3
"like" : 4
"tea" : 5
}
```

Every token gets a unique integer.

---

# Why Vocabulary?

Neural networks only understand integers.

Example

```
Hello world
```

↓

```
[15,302]
```

---

# Reverse Vocabulary

Need reverse lookup too.

```
15 → Hello

302 → world
```

Useful during generation.

---

# SimpleTokenizerV1

Contains two functions

## Encode

```
Text

↓

Split

↓

Lookup Vocabulary

↓

Token IDs
```

Example

```
Hello world

↓

[15,302]
```

---

## Decode

```
[15,302]

↓

Hello world
```

Uses reverse dictionary.

---

# Fixing Spaces Around Punctuation

Joining tokens creates

```
Hello , world !
```

Regular expression fixes this.

```
Hello, world!
```

---

# Problem With Unknown Words

Suppose vocabulary contains

```
apple
cat
dog
```

Now encode

```
banana
```

Tokenizer crashes.

Reason

```
banana
```

doesn't exist in vocabulary.

---

# Solution - Special Tokens

Add

```
<|unk|>
```

Unknown word.

Example

```
banana

↓

<|unk|>
```

---

Second Special Token

```
<|endoftext|>
```

Separates documents.

Example

```
Book A

<|endoftext|>

Book B
```

GPT models use this token extensively.

---

# SimpleTokenizerV2

Changes

Before lookup

```
if word not found

↓

replace by

<|unk|>
```

No crash occurs.

---

# Other Common Tokens

## BOS

Beginning of sequence

```
[BOS]
Hello
```

---

## EOS

End of sequence

```
Hello

[EOS]
```

---

## PAD

Padding shorter sequences.

Example

Sentence 1

```
I love AI
```

Sentence 2

```
Hello
```

Batch

```
I love AI

Hello PAD PAD
```

Allows equal length batches.

---

# GPT Doesn't Use UNK

GPT uses

## Byte Pair Encoding (BPE)

instead.

Unknown words are split into known pieces.

Example

```
playing
```

↓

```
play
ing
```

Unknown word

```
electroencephalography
```

↓

broken into many smaller subwords.

Therefore

No UNK token is required.

---

# tiktoken

Official tokenizer.

```python
tokenizer =
tiktoken.get_encoding("gpt2")
```

Encode

```python
ids = tokenizer.encode(text)
```

Decode

```python
tokenizer.decode(ids)
```

---

# Why BPE Is Better

Simple tokenizer

```
Hello

↓

Unknown
```

BPE

```
Hello

↓

Hel
lo
```

Model still understands.

Vocabulary becomes much smaller.

---

# Sliding Window

Language models predict

Next Token.

Example

```
The cat sat on the mat
```

Context length = 4

Input

```
The cat sat on
```

Target

```
cat sat on the
```

Shift by one token.

---

Example

```
x

[10,20,30,40]

y

[20,30,40,50]
```

Every target is the next token.

---

# Why Sliding Window?

Creates many training examples.

Instead of

One sentence

↓

Hundreds of overlapping samples.

---

# GPTDatasetV1

Custom PyTorch Dataset.

Stores

```
input_ids

target_ids
```

Every sample contains

Input

↓

Target

shifted by one token.

---

# DataLoader

Responsible for

- batching
- shuffling
- loading efficiently

```python
DataLoader(
dataset,
batch_size,
shuffle=True
)
```

---

# Important Parameters

## max_length

Maximum context size.

Example

```
256 tokens
```

---

## stride

How much the window moves.

Example

```
max_length = 256

stride = 128
```

Windows overlap.

---

# Token Embeddings

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


# Byte Pair Encoding (BPE) & DataLoader Notes

## What is BPE?

Byte Pair Encoding (BPE) is a **subword tokenization algorithm** used by
many modern LLMs (including GPT-family models).

Instead of storing every word as a separate token, BPE builds a
vocabulary consisting of: - Common words - Frequently occurring subwords

## Rules

### Rule 1

Do **not** split frequently occurring words.

Example:

``` text
boy -> ["boy"]
```

### Rule 2

Split rare words into meaningful subwords.

``` text
boys -> ["boy", "s"]
```

## Why Subword Tokenization?

### 1. Learns relationships

Words like

``` text
token
tokenizer
tokenization
```

share common roots, helping the model generalize.

### 2. Handles unseen words

Words such as

``` text
modern
modernize
modernization
```

can be represented using familiar subwords instead of adding entirely
new vocabulary entries.

------------------------------------------------------------------------

## Original Byte Pair Encoding

Originally introduced as a compression algorithm.

Algorithm:

1.  Find the most frequent adjacent pair.
2.  Replace it with a new symbol.
3.  Repeat until the stopping criterion is reached.

Example

``` text
aaabdaaabac
```

If **aa** is most frequent:

``` text
aaabdaaabac
↓

ZabdZabac
```

Continue merging until no useful merges remain.

------------------------------------------------------------------------

## BPE for LLMs

For language models, BPE repeatedly merges the most frequent character
pairs to create increasingly larger tokens.

Result:

-   Common words become single tokens.
-   Rare words become multiple subword tokens.

------------------------------------------------------------------------

## Training a BPE Vocabulary

Example corpus

``` text
old      : 7
older    : 3
finest   : 9
lowest   : 4
```

### Step 1

Append an end-of-word marker.

``` text
old</w>
older</w>
finest</w>
lowest</w>
```

### Step 2

Split every word into characters.

``` text
o l d </w>
```

### Step 3

Count frequencies of adjacent symbol pairs.

### Step 4

Merge the most frequent pair.

Repeat until:

-   Desired vocabulary size is reached, or
-   Maximum merge iterations are completed.

------------------------------------------------------------------------

## Vocabulary

The final vocabulary may contain tokens like

``` text
old
est
ing
tion
lowest
```

instead of every possible English word.

------------------------------------------------------------------------

## tiktoken

`tiktoken` is OpenAI's tokenizer library.

Features:

-   Fast BPE tokenizer
-   Encoder
-   Decoder
-   Rust implementation for high performance

------------------------------------------------------------------------

# Creating Input--Target Pairs

After tokenization, training examples are created using a sliding
window.

Token IDs

``` text
[11, 45, 82, 19, 6, 90]
```

Input

``` text
[11, 45, 82, 19, 6]
```

Target

``` text
[45, 82, 19, 6, 90]
```

Each target token is simply the **next token**.

------------------------------------------------------------------------

## Sliding Window

``` text
Input : This is an example
Target: is an example sentence

Input : is an example sentence
Target: an example sentence ...
```

------------------------------------------------------------------------

# DataLoader Pipeline

``` text
Raw Text
   ↓
Tokenizer (BPE)
   ↓
Token IDs
   ↓
Input–Target Pairs
   ↓
Dataset
   ↓
DataLoader
   ↓
Training Loop
```

------------------------------------------------------------------------

# Key Takeaways

-   BPE is a subword tokenization algorithm.
-   Frequent words remain whole.
-   Rare words are broken into meaningful pieces.
-   BPE builds vocabulary by repeatedly merging frequent symbol pairs.
-   GPT models typically use BPE-style tokenizers.
-   After tokenization, token sequences become input--target pairs for
    next-token prediction.
-   A DataLoader efficiently feeds these batches into the model during
    training.


    ## What is BPE?

Byte Pair Encoding (BPE) is a **subword tokenization algorithm** used by
many modern LLMs (including GPT-family models).

Instead of storing every word as a separate token, BPE builds a
vocabulary consisting of: - Common words - Frequently occurring subwords

## Rules

### Rule 1

Do **not** split frequently occurring words.

Example:

``` text
boy -> ["boy"]
```

### Rule 2

Split rare words into meaningful subwords.

``` text
boys -> ["boy", "s"]
```

## Why Subword Tokenization?

### 1. Learns relationships

Words like

``` text
token
tokenizer
tokenization
```

share common roots, helping the model generalize.

### 2. Handles unseen words

Words such as

``` text
modern
modernize
modernization
```

can be represented using familiar subwords instead of adding entirely
new vocabulary entries.

------------------------------------------------------------------------

## Original Byte Pair Encoding

Originally introduced as a compression algorithm.

Algorithm:

1.  Find the most frequent adjacent pair.
2.  Replace it with a new symbol.
3.  Repeat until the stopping criterion is reached.

Example

``` text
aaabdaaabac
```

If **aa** is most frequent:

``` text
aaabdaaabac
↓

ZabdZabac
```

Continue merging until no useful merges remain.

------------------------------------------------------------------------

## BPE for LLMs

For language models, BPE repeatedly merges the most frequent character
pairs to create increasingly larger tokens.

Result:

-   Common words become single tokens.
-   Rare words become multiple subword tokens.

------------------------------------------------------------------------

## Training a BPE Vocabulary

Example corpus

``` text
old      : 7
older    : 3
finest   : 9
lowest   : 4
```

### Step 1

Append an end-of-word marker.

``` text
old</w>
older</w>
finest</w>
lowest</w>
```

### Step 2

Split every word into characters.

``` text
o l d </w>
```

### Step 3

Count frequencies of adjacent symbol pairs.

### Step 4

Merge the most frequent pair.

Repeat until:

-   Desired vocabulary size is reached, or
-   Maximum merge iterations are completed.

------------------------------------------------------------------------

## Vocabulary

The final vocabulary may contain tokens like

``` text
old
est
ing
tion
lowest
```

instead of every possible English word.

------------------------------------------------------------------------

## tiktoken

`tiktoken` is OpenAI's tokenizer library.

Features:

-   Fast BPE tokenizer
-   Encoder
-   Decoder
-   Rust implementation for high performance

------------------------------------------------------------------------

# Creating Input--Target Pairs

After tokenization, training examples are created using a sliding
window.

Token IDs

``` text
[11, 45, 82, 19, 6, 90]
```

Input

``` text
[11, 45, 82, 19, 6]
```

Target

``` text
[45, 82, 19, 6, 90]
```

Each target token is simply the **next token**.

------------------------------------------------------------------------

## Sliding Window

``` text
Input : This is an example
Target: is an example sentence

Input : is an example sentence
Target: an example sentence ...
```

------------------------------------------------------------------------

# DataLoader Pipeline

``` text
Raw Text
   ↓
Tokenizer (BPE)
   ↓
Token IDs
   ↓
Input–Target Pairs
   ↓
Dataset
   ↓
DataLoader
   ↓
Training Loop
```

------------------------------------------------------------------------

# Key Takeaways

-   BPE is a subword tokenization algorithm.
-   Frequent words remain whole.
-   Rare words are broken into meaningful pieces.
-   BPE builds vocabulary by repeatedly merging frequent symbol pairs.
-   GPT models typically use BPE-style tokenizers.
-   After tokenization, token sequences become input--target pairs for
    next-token prediction.
-   A DataLoader efficiently feeds these batches into the model during
    training.


    # Lecture 10 & 11 --- Token Embeddings and Positional Embeddings


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

# Lecture 11 --- Positional Embeddings

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



Lecture 12 --- Attention Mechanism Introduction
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



