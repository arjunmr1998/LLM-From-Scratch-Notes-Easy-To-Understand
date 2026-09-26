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

- The transformer architecture  especially its **attention mechanism** is one of the key reasons modern LLMs work so well. The popular paper by google "Attention is all you need". 
- LLM + Deep Learning => Generative AI
- LLM's represent a specific application of deep learning techniques, leverging their ability to process and generate human like text.



------------------------------------------------------------------------

## 2. Stages of Building an LLM

At a high level, creating a useful LLM involves two major stages:

``` text
Creating an LLM
│
├── Pretraining
│   └── Learn general language patterns from a large, diverse dataset.
│
└── Fine-tuning / post-training
    └── Adapt the pretrained model to tasks, instructions, or domains. By training on narrow dataset specific to particular task or domain.
```

### 2.1 Pretraining

During pretraining, the model learns from a **large and diverse text
dataset**.

Possible data sources(Raw unlabelled text) include:

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
generating predictions. This mechnaism has become integral part of compleiing sequence modelling , allowing modelling of dependicies without regard to their distance from input or output sequence.

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