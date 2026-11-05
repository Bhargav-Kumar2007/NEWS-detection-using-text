# Fakeddit fake-news detection (Embedding + BiLSTM / TextCNN)

A self-contained PyTorch project that classifies Reddit post titles from the
[Fakeddit](https://github.com/entitize/Fakeddit) dataset ([project site](https://fakeddit.netlify.app/))
as one of the two official 2-way classes, using the official `clean_title` text field and the
predefined train / validation / test splits.

> **Status:** the code has not been trained or run on the real dataset by its author - no results are
> bundled, and every number you see is computed by the scripts on *your* machine from real predictions.
> Syntax and the data / tokenizer / metric / plotting helpers were checked on synthetic data;
> the first full run is yours. If something breaks, see [Troubleshooting](#troubleshooting).

## What is inside

```
fakeddit_fake_news/
├── config.yaml          all settings (data, preprocessing, model, training)
├── download_data.py     downloads Fakeddit v2.0 text files, extracts, verifies columns/labels
├── preprocessing.py     cleaning, tokenizer, vocabulary, padding, split discovery, DataLoader
├── models.py            Embedding+BiLSTM and Embedding+TextCNN (+ feature extraction head)
├── engine.py            train / inference loops (AMP on CUDA)
├── train.py             training + validation, checkpoints, curves, final val/test evaluation
├── evaluate.py          metrics, classification report, confusion matrix, ROC curve
├── predict.py           predict_text() / extract_features() for arbitrary strings (+ CLI)
├── utils.py             config, seeding, CPU/GPU detection, metrics, plots
├── tests/test_smoke.py  quick install check (no data, no training)
├── run.sh / run.bat     Linux-macOS / Windows helper scripts
├── requirements.txt  .gitignore
```

## Quick start

### Linux / macOS
```bash
./run.sh setup            # creates .venv and installs requirements
./run.sh download         # downloads + verifies the dataset into data/raw/
./run.sh test             # optional: installation smoke test
./run.sh train-small      # quick run on a stratified subset (20k/4k/4k)
./run.sh train-full       # whole official splits
./run.sh predict --text "Scientists discover a cat that can speak fluent French"
```

### Windows
```bat
run.bat setup
run.bat download
run.bat train-small
run.bat train-full
run.bat predict --text "Scientists discover a cat that can speak fluent French"
```
Any extra arguments are forwarded to the Python script, e.g. `./run.sh train-small --model textcnn --epochs 5`.

Without the helper scripts: `pip install -r requirements.txt`, then `python download_data.py`,
`python train.py --mode small`, etc.

### GPU
Device selection is automatic (`cuda` -> `mps` -> `cpu`) and logged at start-up. Force one with
`--device cpu|cuda|mps`. **Windows note:** plain `pip install torch` installs the CPU-only build. For an
NVIDIA GPU install torch first, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu121` (pick your CUDA version at
https://pytorch.org/get-started/locally/) - or set `TORCH_INDEX_URL` before `run.bat setup` / `./run.sh setup`.
Mixed precision is used automatically on CUDA (`training.use_amp`).

### Google Colab
1. Runtime -> Change runtime type -> GPU.
2. Upload the project folder (or its zip) to Colab, then:
```python
!unzip -q fakeddit_fake_news.zip && cd fakeddit_fake_news   # skip if you uploaded the folder
%cd fakeddit_fake_news
!pip install -q gdown pyyaml scikit-learn matplotlib tqdm pandas   # Colab already ships torch
!python download_data.py
!python train.py --mode small --set training.num_workers=2
```
3. To keep results after the session, mount Drive and point outputs there:
```python
from google.colab import drive; drive.mount('/content/drive')
!python train.py --mode full --set paths.output_dir=/content/drive/MyDrive/fakeddit_outputs
```
(`--resume` continues an interrupted run from `checkpoints/last.pt`.)

## Dataset handling
* `download_data.py` fetches the **v2.0 text/metadata Google Drive folder** that the Fakeddit GitHub page
  links to (folder id in `config.yaml -> data.gdrive_folder_id`) using `gdown`, extracts any archives, finds the
  train / validate / test files and verifies they contain `clean_title` and `2_way_label`.
* `data.variant: all` (default) uses the full text splits; `multimodal_only` uses only samples that also
  have an image (the subset the paper's baselines use). Files are discovered automatically; you can pin exact
  names under `data.files`.
* Text = `clean_title`, label = `2_way_label`. The official splits are used as released - nothing is
  re-split. (The Fakeddit authors ask that the public test set is used only for final evaluation; this project
  selects the best epoch on the **validation** split.)
* If the automatic download fails (Google Drive quota is a common cause), download the TSV files manually from
  the link on the Fakeddit GitHub page into `data/raw/` and run `python download_data.py --check`.

### Label semantics (please verify)
The scripts treat the 2-way label as binary `{0, 1}` and report precision/recall/F1 for **label 1**.
Neither the repository README nor the website states in plain text which id means "fake", so class names
default to `label_0` / `label_1`. Check the dataset paper (arXiv:1911.03854) / the data itself and set
`labels.class_names` in `config.yaml` (e.g. `["fake", "real"]`) before interpreting precision/recall as
"fake-news precision/recall". Training and AUC/accuracy are unaffected.

## Small subset vs full dataset
| | `--mode small` | `--mode full` |
|---|---|---|
| train / val / test | 20,000 / 4,000 / 4,000 (stratified, seeded) | entire official splits |
| output folder | `outputs/<model>_small/` | `outputs/<model>_full/` |

Change the sizes under `data.subset_sizes` in `config.yaml`. Any setting can be overridden on the command line:
`python train.py --set training.epochs=8 model.embed_dim=300 preprocessing.max_len=64`.

## Models
* **BiLSTM** (`--model bilstm`): embedding -> dropout -> bidirectional LSTM (packed sequences) -> [final fwd/bwd
  hidden states + masked max-pool] -> dense (`fc_dim`) -> logit.
* **TextCNN** (`--model textcnn`): embedding -> parallel Conv1d (kernel sizes 2-5) -> global max-pool -> dense (`fc_dim`) -> logit.

Loss: `BCEWithLogitsLoss` (optional `use_pos_weight`). Optimiser: AdamW + ReduceLROnPlateau + gradient clipping.
Model selection / early stopping on `training.monitor` (default validation F1).

## Preprocessing
`preprocessing.py`: HTML-unescape -> Unicode NFKC -> URL removal -> lowercase -> keep letter/digit words
(apostrophes kept inside words) -> vocabulary built **from the training split only** (`max_vocab_size`,
`min_freq`, `<pad>=0`, `<unk>=1`) -> truncate / right-pad to `max_len`. Empty strings become a single `<unk>`.

## Outputs (per run: `outputs/<model>_<mode>/`)
```
checkpoints/best_model.pt     best epoch by the monitored validation metric (inference-ready)
checkpoints/last.pt           latest epoch incl. optimizer state (for --resume)
tokenizer.json                vocabulary + preprocessing settings
config_used.yaml  history.json  train.log  training_summary.json
plots/loss_curve.png  plots/accuracy_curve.png
evaluation/validation_<mode>/ and evaluation/test_<mode>/
    metrics.json              accuracy, precision, recall, F1, ROC-AUC (+ macro scores, confusion matrix)
    classification_report.txt
    confusion_matrix.png / .csv
    roc_curve.png
    predictions.csv           text, true label, P(label 1), predicted label
```
Re-evaluate any time (optionally on the full official split even if trained on a subset):
```bash
python evaluate.py --run-dir outputs/bilstm_small --split test
python evaluate.py --run-dir outputs/bilstm_small --split test --full-split
```

## Using the model from Python
```python
from predict import predict_text, extract_features, FakeNewsPredictor

predict_text("Man finds 3,000-year-old sandwich")                  # dict: label, label_name, prob_label_1, confidence
predict_text(["headline 1", "headline 2"], run_dir="outputs/bilstm_full")   # list of dicts
extract_features("Man finds 3,000-year-old sandwich")              # numpy array, shape (fc_dim,)
extract_features(["a", "b", "c"])                                   # numpy array, shape (3, fc_dim)

p = FakeNewsPredictor("outputs/textcnn_small")                      # load once, reuse
p.predict("...", threshold=0.6); p.extract_features(["...", "..."])
```
`run_dir=None` uses `inference.default_run` or the most recently trained run. CLI:
`python predict.py --text "..." --features`, `--file headlines.txt`, `--interactive`.

## Troubleshooting
* **Download fails / quota exceeded** - use the manual route described under *Dataset handling*.
* **"Could not find ... files"** - the error lists the `.tsv` files it saw; set `data.files.train/validation/test` in `config.yaml`.
* **Columns not found** - the error prints the available columns; adjust `data.text_column` / `data.label_column`.
* **CUDA out of memory** - lower `training.batch_size`; **slow on CPU** - use `--mode small`, `--model textcnn`, or Colab GPU.
* **Windows DataLoader issues** - keep `training.num_workers: 0`.

## Notes
* Reproducibility: seeded (`seed` in config), but GPU kernels may still be non-deterministic.
* Pretrained embeddings (GloVe/fastText) are not used; embeddings are learned from scratch.
* Dataset licence/citation: Nakamura, Levy, Wang - *r/Fakeddit* (LREC 2020, arXiv:1911.03854). Nothing here is pushed anywhere; the project is purely local.
