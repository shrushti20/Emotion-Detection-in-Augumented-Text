# Emotion Detection in Argumentative Text

Repository for a master's thesis project on **emotion recognition in argumentative text under domain shift**. The work studies how emotion classifiers trained on general-domain corpora transfer to the argumentative **CONTARGA** setting, and compares those supervised models with prompt-based large language model inference. The thesis motivation and scope center on benchmarking cross-domain emotion recognition, analyzing errors, and relating predicted emotions to convincingness in arguments.

## Scope of final thesis experiments

The final experiments reported in the thesis focus on RoBERTa, DeBERTa, and prompt-based large language model approaches on the CONTARGA dataset.

The repository also contains additional exploratory experiments (e.g., XLM-R-based models), which were part of early experimentation but are **not included in the final thesis evaluation**.

## Project goals

This repository supports a thesis workflow with three main aims:

1. **Supervised transfer learning** for emotion recognition, using models trained on external datasets such as **GoEmotions** and **TweetEval Emotion**.
2. **Prompt-based inference** with instruction-tuned LLMs such as **Mistral** and **Zephyr** on argumentative text.
3. **Evaluation and analysis** on CONTARGA, including metrics, mapping between label spaces, model agreement, and convincingness correlations.

Final results in the thesis are based on RoBERTa, DeBERTa, and LLM-based approaches.

## Repository structure

```text
Emotion-Detection-in-Augumented-Text/
├── jobs/                  # SLURM batch scripts for training and inference
├── prompts/               # Prompt templates for CONTARGA / emotion prompting
├── thesis_adapt/          # Domain adaptation workspace (jobs, prompts, scripts, data)
├── thesis_scripts/        # Main evaluation, LLM, plotting, and utility scripts
├── thesis_training/       # Core transformer training scripts
├── requirements.txt       # Python dependencies
└── .gitignore
```

### `jobs/`
Cluster job files for running experiments on HPC infrastructure. These include training and evaluation jobs for:

- DeBERTa on GoEmotions / TweetEval
- RoBERTa transfer evaluation on CONTARGA
- Mistral and Zephyr prompt-based runs in zero-shot, few-shot, and chain-of-thought configurations

### `prompts/`
Prompt resources used by the LLM-based pipeline:

- `contarga_base_prompt.txt`
- `contarga_emotion_prompts.txt`

### `thesis_training/`
Core supervised training and zero-shot transfer scripts.

- `01_train_goemotions_roberta.py` — trains **RoBERTa-base** for multi-label emotion classification on **GoEmotions**
- `02_train_tweeteval_roberta.py` — trains **RoBERTa-base** on **TweetEval Emotion**
- `03_eval_goemotions_on_contarga.py` — evaluates the GoEmotions-trained RoBERTa model on **CONTARGA**, exports probabilities, computes overlap-label metrics, and estimates correlations with convincingness

### `thesis_scripts/`
Main research scripts used for downstream experimentation and thesis artifact generation.

#### LLM inference
- `contarga_llm_mistral_modes.py` — runs **Mistral** on CONTARGA in `zero`, `few`, `cot`, or `tfidf` retrieval-augmented few-shot modes
- `contarga_llm_mistral_multilabel.py` — Mistral-based LLM inference on CONTARGA using a multi-label prompting setup

#### Supervised evaluation and label augmentation
- `deberta_contarga_add_labels.py`
- `deberta_contarga_eval.py`
- `roberta_contarga_add_labels.py`
- `roberta_contarga_eval.py`
- `roberta_tweets_eval.py`
- `eval_contarga_28labels.py`
- `eval_tweeteval_to_contarga.py`

These scripts support exporting predictions, aligning label spaces, and evaluating models on the argumentative target domain.

#### Training / adaptation
- `train_deberta_contarga.py` — fine-tunes **DeBERTa-v3-base** on an 8-label CONTARGA setup
- `train_deberta_goemo.py`
- `train_deberta_tweets.py`
- `train_roberta_tweets.py`

#### Analysis, reporting, and thesis outputs
- `combine_roberta_llm.py` — merges supervised and LLM outputs for direct comparison
- `llm_contarga_metrics.py` — computes accuracy and macro-F1 for LLM predictions
- `model_agreement.py` — analyzes agreement between model families
- `rq3_convincingness_corr.py` — examines how predicted emotions relate to convincingness
- `make_metrics_table.py` — assembles metrics tables
- `make_thesis_plots.py` — generates thesis-ready plots from summary CSV files
- `make_thesis_tables.py` / `make_wide_tables.py` — creates thesis tables in export-friendly formats
- `make_contarga_fewshot_8labels.py` / `make_contarga_fewshot_unique_k8.py` — prepares few-shot examples for LLM prompting
- `download_tweeteval.py` — dataset retrieval helper

#### Supporting folders
- `mapping/` — label-space mapping resources
- `notebooks/` — exploratory notebooks and intermediate experiments
- `prompts/` — local script-level prompt assets

### `thesis_adapt/`
A separate adaptation-oriented workspace with its own:

- `jobs/`
- `logs/`
- `prompts/`
- `scripts/`
- `data_bal_n95`

This directory appears to contain intermediate or domain-adaptation experiments organized separately from the main code paths.

## Experimental pipeline

### 1. Train supervised source-domain models
The repository contains training scripts for:

- **RoBERTa-base**
- **XLM-R**
- **DeBERTa-v3-base**

Source datasets include:

- **GoEmotions** for fine-grained multi-label emotion recognition
- **TweetEval Emotion** for single-label emotion classification

### 2. Transfer models to argumentative text
Trained models are evaluated on **CONTARGA**, where the core challenge is domain shift from social / general text to argumentative discourse.

Typical transfer steps include:

- loading the best checkpoint
- running batch inference over CONTARGA text
- exporting per-emotion probabilities or class predictions
- aligning source labels with CONTARGA labels
- computing macro-F1, accuracy, precision, recall, and threshold-based analyses

### 3. Run prompt-based LLM baselines
The LLM pipeline supports multiple prompting strategies:

- **zero-shot**
- **few-shot**
- **chain-of-thought** (`cot`)
- **TF-IDF retrieved few-shot prompting**

The main script builds prompts, runs text generation with Hugging Face causal language models, parses `FINAL:` outputs, and stores one-vs-rest indicator columns for the target emotion set.

### 4. Compare systems and analyze results
The repository also supports:

- supervised vs. LLM comparison
- metric aggregation
- model agreement analysis
- convincingness correlation analysis
- plot and table generation for thesis reporting

## Datasets and label spaces

This project uses a **cross-domain emotion transfer** setup rather than a single unified dataset.

### Main datasets
- **GoEmotions** — fine-grained Reddit emotion dataset
- **TweetEval Emotion** — tweet-based single-label emotion dataset
- **CONTARGA** — argumentative text target dataset used for downstream evaluation and analysis

### Label handling
Several scripts map or restrict predictions to an **8-label CONTARGA-compatible emotion subset**:

- anger
- disgust
- fear
- joy
- pride
- relief
- sadness
- surprise

Other scripts retain broader source-domain label spaces, including the 28-label GoEmotions setup.

## Requirements

The codebase relies primarily on:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas / numpy / scipy
- matplotlib
- tqdm
- sentencepiece

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Recommended environment

- Python **3.10+**
- CUDA-enabled GPU for training and large-model inference
- Linux / HPC environment for SLURM jobs

## Reproducibility note

Parts of the model training were initially developed and executed in Google Colab for rapid prototyping.

For the final experimental pipeline, the code was consolidated into standalone Python scripts and executed on an HPC environment for large-scale training, inference, and evaluation. Pretrained checkpoints obtained during earlier runs were reused where appropriate.

The repository therefore contains the final Python-based implementation used for all reported experiments.
Pretrained model checkpoints were generated during earlier training runs and reused for downstream evaluation and analysis.

## Example usage

### Train RoBERTa on GoEmotions
```bash
python thesis_training/01_train_goemotions_roberta.py
```

### Train RoBERTa on TweetEval Emotion
```bash
python thesis_training/02_train_tweeteval_roberta.py
```

### Evaluate GoEmotions-trained RoBERTa on CONTARGA
```bash
python thesis_training/03_eval_goemotions_on_contarga.py
```

### Fine-tune DeBERTa on an 8-label CONTARGA split
```bash
python thesis_scripts/train_deberta_contarga.py \
  --train data/contarga_train.csv \
  --valid data/contarga_valid.csv \
  --test data/contarga_test.csv \
  --out_dir outputs/deberta_contarga
```

### Run Mistral in zero-shot mode on CONTARGA
```bash
python thesis_scripts/contarga_llm_mistral_modes.py \
  --mode zero \
  --data data/contarga_eval.csv \
  --out outputs/mistral_zero_contarga.csv
```

### Run Mistral with TF-IDF retrieved few-shot examples
```bash
python thesis_scripts/contarga_llm_mistral_modes.py \
  --mode tfidf \
  --data data/contarga_eval.csv \
  --train_data data/contarga_train.csv \
  --out outputs/mistral_tfidf_contarga.csv \
  --k 8 \
  --balance
```

### Generate thesis plots
```bash
python thesis_scripts/make_thesis_plots.py
```

## Outputs produced by the scripts

Depending on the pipeline stage, scripts export:

- trained model checkpoints
- tokenizer files
- per-instance emotion probabilities
- prediction CSVs for supervised and LLM models
- evaluation summaries
- convincingness correlation tables
- merged comparison CSVs
- publication/thesis-ready plots and tables

## Notes on paths and reproducibility

A number of scripts currently use **absolute HPC paths** such as:

```text
/home/hpc/v121ca/v121ca21/...
```

To run the repository on another machine, these paths usually need to be adapted to your local project structure. In practice, the easiest cleanup is to:

1. centralize paths in a config file or environment variables,
2. keep datasets under a local `data/` directory,
3. redirect outputs to a local `outputs/` or `results/` directory.

## Limitations / cleanup opportunities

The repository is research-oriented and reflects iterative thesis experimentation. Before external reuse, it would help to:

- normalize path handling across scripts,
- document expected input CSV schemas,
- pin package versions more strictly for reproducibility,
- separate final experiments from exploratory notebooks,
- add sample config files and example input data manifests.

## Citation / context

This repository accompanies a thesis on **emotion recognition in arguments**, focused on evaluating out-of-domain emotion models on argumentative text and comparing them with prompt-based LLM methods.

