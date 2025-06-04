# Urdu2Eng Transformer  
** CS455 NLP Project **  
**Author:** Ehtisham Khalid (2021147)

---

## Project Overview

This project implements a **Transformer-based Neural Machine Translation (NMT) model** that translates text from **Urdu to English**. The architecture leverages **Sinusoidal Positional Embeddings** and is built using **PyTorch**, without relying on high-level abstractions like `torchtext`.

The project includes:
- Custom transformer encoder-decoder architecture
- Sinusoidal positional encoding
- BLEU score evaluation
- Manual vocabulary and token handling using spaCy and NLTK
- Data batching and padding from scratch

---

## Features

- Transformer model with:
  - Multi-head self-attention
  - Position-wise feedforward networks
  - Layer normalization and dropout
- Sinusoidal positional encoding (as per Vaswani et al.)
- BLEU Score evaluation for translation quality
- Dynamic batching using `pad_sequence` and `DataLoader`
- No reliance on high-level libraries like `torchtext`

---

## Installation

Install the required packages before running the notebook:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install spacy
pip install nltk
python -m spacy download en_core_web_sm

Project Structure
TransformerEncoderLayer – Custom Transformer encoder block

TransformerDecoderLayer – Decoder block implementation (defined in notebook)

PositionalEncoding – Implements sinusoidal embedding of positions

build_vocab() – Vocabulary builder based on token frequency

collate_fn() – Custom batch collation and sequence padding

train() / evaluate() – Custom training and evaluation loops

BLEU – Translation quality measurement using BLEU score

Dataset
To run the model, you will need a parallel Urdu-English dataset with tokenized sentence pairs.

Basic preprocessing should include:

Lowercasing

Removing punctuation

Tokenization using spaCy (English) and basic Urdu rules

Evaluation
Translation quality is evaluated using BLEU Score

Sentence-level performance is calculated

Qualitative checks include output samples and manual accuracy

How to Run
Clone this repository or open the .ipynb file in Jupyter or Colab.

Install the dependencies listed above.

Load and preprocess your Urdu-English dataset.

Define source (SRC) and target (TGT) token mappings.

Train the model using the provided train() function.

Evaluate using BLEU score and test translations.

Future Improvements
Use larger and more diverse Urdu-English datasets (e.g., OPUS, TED Talks)

Add subword tokenization (Byte Pair Encoding or SentencePiece)

Add beam search for improved decoding

Implement attention visualization

Hyperparameter optimization

License
This project is intended for academic and research use. Please cite appropriately if reusing the code or methodology.
