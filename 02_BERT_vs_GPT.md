## 7. BERT vs GPT

Transformers can be adapted in different ways. Key parts of transformers, allows model to weigh importance of different 
position of a single sequence in order to compute a representation of sequence.

### BERT

**BERT --- Bidirectional Encoder Representations from Transformers**

BERT is based primarily on the **encoder** side of the Transformer.

A key feature is bidirectional context: a token representation can use
information from both the left and right sides of the sequence. 
- By looking at the the entire sentence from both directions, BERT can capture nuances and relationship between words
that are important for understanding meaning and context.
- For instance , it can differentiate between "bank" as a finanicial instituion and "bank" as a river bank by considering the surrounding words.

Example:

``` text
"I deposited money in the bank."
"The boat reached the river bank."
```

The surrounding words help the model distinguish the meaning of
**bank**.

BERT was originally pretrained using objectives including **masked
language modeling**, where hidden/masked tokens are predicted.

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

##  Autoregressive Language Modeling

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
capabilities. Ability of model to perform taks that the model wasn't explicitly trained to perform is called "emenrgent behaviour".

------------------------------------------------------------------------

##  Self-Supervised Learning

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

##  Tokens

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

##  Transformers vs LLMs

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

##  Zero-Shot and Few-Shot Learning

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

##  Emergent Capabilities

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
