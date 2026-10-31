# 8. Perplexity — Measuring Language Model Performance

## 8.1 What is perplexity?

Perplexity is a metric used to evaluate language models. It measures how well a model predicts a sequence of tokens.

It is related to cross-entropy loss, but it is often easier to interpret as a measure of the model's uncertainty when predicting the next token.

- Lower perplexity generally indicates better next-token prediction on the same evaluation dataset and under the same tokenization.
- Higher perplexity indicates that the model assigns less probability to the observed sequence.

Perplexity is most useful when comparing models evaluated on the same data and with compatible tokenization and evaluation procedures.

## 8.2 Relationship between cross-entropy and perplexity

For average negative log-likelihood (cross-entropy) measured using natural logarithms:

\[
\operatorname{PPL}=\exp(L_{\text{mean}})
\]

If cross-entropy is measured using logarithms base 2:

\[
\operatorname{PPL}=2^{L_{\text{mean}}}
\]

Perplexity is therefore the exponential of the average negative log-likelihood.

## 8.3 Intuition

Imagine a model that is equally uncertain among \(k\) possible next tokens. If it assigns probability \(1/k\) to the observed token in this simplified setting, its perplexity is \(k\).

For example:

- Perplexity of 1 corresponds to perfect prediction in the idealized case.
- Perplexity of 10 can be interpreted as uncertainty equivalent to choosing uniformly among 10 alternatives.
- Lower perplexity indicates that the model places more probability on the actual observed tokens.

This interpretation is an intuition, not a claim that a model literally chooses uniformly from that many words.

## 8.4 Example calculation

Suppose the average cross-entropy loss is:

\[
L_{\text{mean}}=2
\]

Then:

\[
\operatorname{PPL}=e^2\approx7.39
\]

The perplexity is approximately 7.39.

If the average loss decreases, perplexity also decreases.

## 8.5 PyTorch calculation

```python
import math
import torch.nn.functional as F

# logits: model output before Softmax
# targets: correct next-token IDs

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1)
)

perplexity = math.exp(loss.item())

print("Cross-entropy:", loss.item())
print("Perplexity:", perplexity)
```

This example assumes the loss is the mean natural-log cross-entropy over the target positions being evaluated.

## 8.6 How should perplexity be interpreted?

- A lower score is generally better when comparing models on the same evaluation data and setup.
- Perplexity is not a direct measure of factual correctness, reasoning ability or usefulness.
- It can change with the dataset, tokenizer, vocabulary and evaluation method.
- Scores from different tokenizers or datasets should not be compared casually.

> **Key takeaway:** Perplexity is the exponential of average cross-entropy loss. It gives an intuitive view of how uncertain a language model is when predicting the next token.
