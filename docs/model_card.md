# Model Card Draft

## Model

Native Urdu decoder-only language model trained from scratch.

Default target:

- Parameters: 1.3B
- Context length: 4096
- Tokenizer: 32k SentencePiece BPE
- Architecture: LLaMA-style decoder-only transformer
- Precision: BF16

## Intended Use

The base model is intended for Urdu language modeling, continued pretraining, and later supervised fine-tuning. It is not an instruction-tuned assistant until post-training is completed.

## Training Data

Default mixture:

- 72% Urdu web
- 20% Urdu literature / curated long-form
- 8% English replay

## Evaluation

Track validation loss separately for:

- Urdu web
- Urdu literature
- English replay
- News
- Poetry
- Wiki-like text

Urdu-native downstream evaluation should include cloze completion, reading comprehension, summarization, question answering, grammar correction, code-switching, basic math in Urdu, safety/toxicity, and long-form generation.

## Risks

Primary quality risks:

- Broken Urdu Unicode normalization
- Poor Urdu-vs-Arabic/Persian language identification
- Web boilerplate and SEO spam
- Near-duplicate contamination
- Tokenizer fertility above target
- Evaluation contamination

