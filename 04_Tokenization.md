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

Tokenization comes under "Data Preparation and sampling".
How do you prepare input text for training text?
- Splitting text into individual word and subword token.
- Convert tokens into tokens ID's.
- Encode token ID's into vector represntation.


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

Tokenization algorithms can be:-
- Word based :- "My hobby is playing cricket" --> "My" , "hobby", "is", "playing" , "cricket"
             Problem:- WHat to do with out of vocabulary words, different meaning of smae words.

- Character based :- "M" , "Y", "h" "o".......
        Problem:- Very small vocabulary as every language has fixed no of characters.
        And no meaning associated. ALso token sequence is much larger than initial raw text.

 Sub-wrod :- Byte pair encoding is "sub- word based" tokenization. 

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
