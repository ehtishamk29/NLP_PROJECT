Urdu2Eng Transformer
CS455 NLP Project
By: Ehtisham Khalid (2021147)

Project Overview
This project implements a Transformer-based Neural Machine Translation (NMT) model that translates text from Urdu to English. The architecture leverages Sinusoidal Positional Embeddings and is built using PyTorch, without relying on high-level abstractions like torchtext.

The project showcases:

Custom Transformer encoder-decoder model implementation

Tokenization using spaCy

BLEU score-based evaluation

Training pipeline built from scratch

Dataset preprocessing pipeline

Features
Complete transformer architecture including:

Multi-head attention

Layer normalization

Position-wise feedforward networks

Sinusoidal positional encoding as described in the original Transformer paper

BLEU Score evaluation for translation performance

Preprocessing and batching using pad_sequence and DataLoader

Installation
Before running the code, install the required packages:

bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install spacy
pip install nltk
python -m spacy download en_core_web_sm
Project Structure
TransformerEncoderLayer – Custom implementation of a Transformer Encoder block

TransformerDecoderLayer – Custom implementation of Decoder block (present later in the notebook)

PositionalEncoding – Sinusoidal positional embedding module

build_vocab() – Manual vocabulary construction using frequency thresholds

collate_fn() – Batch collation and padding

train() and evaluate() – Custom training and evaluation loops

BLEU – Evaluation of translation performance using BLEU score

Dataset
This project likely uses a parallel Urdu-English corpus (not shown in the provided snippet). You will need:

A tokenized Urdu-English sentence pair dataset

Preprocessing to lowercase, remove punctuations, and clean the data

Evaluation
Model is evaluated using BLEU Score

Qualitative evaluation through sample translation outputs

Sentence-level metrics to gauge fluency and adequacy

How to Run
Clone the repository or open the notebook.

Install the required dependencies.

Load the dataset and define SRC (Urdu) and TGT (English) tokens.

Run training and monitor the BLEU scores.

Test the model with your own Urdu sentences.

Future Improvements
Integrate a larger corpus (e.g., OpenSubtitles, OPUS)

Introduce Byte-Pair Encoding (BPE) for subword tokenization

Experiment with learned positional embeddings

Hyperparameter tuning (heads, layers, d_model)

License
This project is for academic and research purposes only. Please cite appropriately if reused.
