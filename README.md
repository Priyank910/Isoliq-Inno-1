# Isoliq Inno 1

An evidence-first experiment in building, pretraining, fine-tuning, and diagnosing a compact language model with Google Colab and [nanoGPT](https://github.com/karpathy/nanoGPT).

Isoliq Inno 1 contains the full reproducible workspace behind a **3,000,768-parameter** model trained from scratch on a custom corpus. The project preserves the source data, tokenizer, encoded datasets, training configurations, audit reports, checkpoints, generation utility, and a step-by-step Colab user guide.

> This is an educational research project, not a production assistant. The model is intentionally small so its pipeline, data, and failure modes remain inspectable.

## Start here

Read [USER-GUIDE.md](USER-GUIDE.md) for the complete Google Colab workflow. It explains every command, expected output, generated file, recovery procedure, and common error.

Large datasets and checkpoints are stored with Git LFS. Clone the repository with Git LFS installed:

```bash
git lfs install
git clone https://github.com/Priyank910/Isoliq-Inno-1.git
cd Isoliq-Inno-1
git lfs pull
```

Then upload or copy the project into this Google Drive location before following the guide:

```text
/content/drive/MyDrive/Isoliq Inno 1
```

## What this repository demonstrates

- Training a small GPT-style model from scratch rather than downloading pretrained weights.
- Building a deterministic 1,280-token byte-level BPE tokenizer.
- Preparing train and validation binaries with recorded corpus fingerprints.
- Pinning nanoGPT to a known source revision for reproducibility.
- Running cheap smoke tests before full pretraining and fine-tuning.
- Preserving pretrained weights while resetting optimizer state for supervised fine-tuning.
- Applying loss only to assistant-answer tokens with explicit binary masks.
- Auditing duplicate and conflicting conversation groups before training.
- Evaluating multiple prompts and random seeds instead of selecting one convenient output.
- Diagnosing why a technically successful pipeline can still learn the wrong behavior from an unbalanced dataset.

## Model architecture

| Setting | Value |
| --- | ---: |
| Trainable parameters | 3,000,768 |
| Transformer layers | 6 |
| Attention heads | 6 |
| Embedding width | 192 |
| Context length | 512 tokens |
| Vocabulary | 1,280 tokens |
| Dropout during pretraining | 0.0 |
| Bias | Disabled |

The default nanoGPT parameter report is **2,902,464** because it excludes learned position embeddings. The verification script records both values and asserts the expected architecture.

## Repository contents

```text
.
|-- USER-GUIDE.md                 # Complete Colab walkthrough and command reference
|-- artifacts/
|   |-- tokenizer.json            # Trained byte-level BPE tokenizer
|   |-- pretrain_audit.json       # Corpus fingerprint, split, and token counts
|   `-- finetune_audit.json       # Conversation and supervised-token audit
|-- checkpoints/
|   |-- pretrain/                 # Main pretrained checkpoint
|   |-- pretrain-smoke/           # Pretraining smoke-test checkpoint
|   |-- finetune/                 # Main fine-tuned checkpoint
|   `-- finetune-smoke/           # Fine-tuning smoke-test checkpoint
|-- data/
|   |-- input.txt                 # Pretraining source corpus
|   |-- finetune_data.txt         # Original conversation data
|   |-- finetune_data2.txt        # Prepared conflict-filtered copy
|   `-- finetune_data_clean.txt   # Auditable cleaned ChatML output
`-- nanoGPT/
    |-- config/                    # Isoliq pretraining and fine-tuning configs
    |-- data/isoliq/               # Encoded pretraining data
    |-- data/isoliq_sft/           # Encoded SFT data and assistant masks
    |-- isoliq_scripts/            # Preparation, patching, verification, and generation
    |-- model.py
    `-- train.py
```

## Reproducibility snapshot

| Measurement | Recorded value |
| --- | ---: |
| Complete pretraining documents | 4,800 |
| Pretraining train documents | 4,560 |
| Pretraining validation documents | 240 |
| Pretraining train tokens | 38,273,145 |
| Pretraining validation tokens | 1,986,164 |
| Prepared fine-tuning conversations | 2,308 |
| Fine-tuning train tokens | 198,870 |
| Fine-tuning validation tokens | 21,295 |
| Fine-tuning supervised train tokens | 88,611 |
| Fine-tuning supervised validation tokens | 9,589 |
| Random seed | 1337 |

The audit JSON files are the source of truth for these values. Dataset SHA-256 fingerprints and validation commands are documented in [USER-GUIDE.md](USER-GUIDE.md).

## Important lesson

The pipeline can run correctly, losses can decrease, checkpoints can save, and the resulting model can still behave badly. In this experiment, the most useful result was not a polished chatbot response. It was discovering that cleaned data was still heavily unbalanced.

**Clean data is not automatically balanced data.** Deduplication and conflict removal do not replace distribution analysis, representative evaluation, or deliberate dataset design.

## Limitations

- A 3M-parameter model has very limited language capacity.
- Generated text may be incorrect, repetitive, biased, or unsafe.
- Lower loss does not prove factuality or conversational quality.
- Colab hardware and runtime availability vary between sessions.
- Results can change when the corpus, tokenizer, configuration, or random seed changes.
- The included checkpoints are experimental artifacts and should not be used for consequential decisions.

## Data responsibility

Use only data you have permission to process and redistribute. Remove private, sensitive, copyrighted, or identifying information when required. Review dataset licenses and local requirements before adapting this workflow to another corpus.

## Credits

This project builds on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and PyTorch. The vendored nanoGPT source retains its upstream `LICENSE` file. ChatGPT and Codex were used to help explain code, design checks, and interpret supplied outputs; the experiment owner ran the notebook and verified the results.
