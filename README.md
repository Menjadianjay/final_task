# Javanese Automatic Speech Recognition

An experimental automatic speech recognition (ASR) pipeline for Javanese speech. The project prepares and cleans transcribed audio, creates phone or word alignments with a GMM-HMM workflow, and trains PyTorch TDNN and TDNN-F acoustic models.

This repository contains research code and experiment outputs rather than a packaged command-line application. Most scripts currently use absolute Windows paths, so update their configuration constants before running them on another machine.

## Pipeline

```text
raw WAV/FLAC audio + transcripts
              │
              ▼
      stratified sampling
              │
              ▼
      transcript/audio validation
      normalization and filtering
              │
              ▼
       MFCC + Δ + Δ² features
              │
              ▼
        MFA / GMM-HMM alignment
              │
              ▼
      LDA features and manifests
              │
       ┌──────┴──────┐
       ▼             ▼
     TDNN          TDNN-F
       └──────┬──────┘
              ▼
   phone decoding, language-model scoring,
        and CER evaluation
```

## Main components

- **Stratified sampling** — samples audio from the dataset hierarchy and joins each file with its transcript.
- **Preprocessing** — validates audio paths, normalizes transcripts, removes empty or invalid recordings, and filters entries containing English words.
- **Alignment** — builds alignment maps from MFA TextGrid files and trains or uses GMM-HMM alignment components.
- **Feature extraction** — computes 13 MFCC coefficients with first- and second-order deltas. The acoustic-model input uses the resulting features after the project’s LDA/alignment preparation.
- **TDNN** — a configurable time-delay neural network with splicing, batch normalization, dropout, cosine scheduling, SpecAugment, and early stopping.
- **TDNN-F** — a factorized TDNN with bottleneck layers, residual bypass connections, and semi-orthogonal weight updates.
- **Decoding and evaluation** — maps model outputs to phones, applies fuzzy phone decoding and an n-gram language model, and writes CER details and summaries.

## Repository layout

```text
.
├── .github/
├── data/                                      # local data; ignored by Git
├── src/codeStat/modelling/voiceModelling/
│   ├── preModelling/src/
│   │   ├── stratified_sampling.py             # create a sampled dataset
│   │   └── preprocessing.py                   # validate and clean transcripts/audio
│   ├── kaldialignment/makeAligner/src/
│   │   ├── Fase_1_CreateAligner.py            # create/validate alignment inputs
│   │   └── Fase_2v2_GMMHMMAligner.py          # GMM-HMM alignment and features
│   └── TDNN-TDNN-F/
│       ├── src/
│       │   ├── Shared_resource.py             # shared features, labels, trainer, decoder
│       │   ├── TDNN.py                         # TDNN training/evaluation
│       │   ├── TDNNF.py                        # TDNN-F training/evaluation
│       │   ├── ViterbiDec.py                   # decoding helpers
│       │   └── label_map_fixed.csv             # label mapping
│       ├── TDNN/out/                           # TDNN manifests, checkpoints, metrics
│       └── TDNN-F/out/                         # TDNN-F manifests, checkpoints, metrics
├── TODO.md
└── README.md
```

Large datasets under `data/` and generated caches are intentionally excluded by `.gitignore`. Selected model checkpoints, manifests, language-model data, and evaluation results are currently tracked under the `out/` directories.

## Requirements

Use Python 3.11. The code imports these main packages:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate          # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn soundfile \
    torch torchaudio tgt
```

CUDA, a compatible PyTorch build, and optionally CuPy can be used for GPU-assisted feature extraction and alignment. Install the PyTorch build appropriate for the target CUDA version from the official PyTorch instructions.

The MFA alignment stage additionally requires Montreal Forced Aligner, a Javanese pronunciation dictionary, and a compatible acoustic model. Those assets are external to this repository.

## Data preparation

The scripts expect a dataset organized similarly to:

```text
data/
└── Data/
    ├── 0/asr_javanese/data/<speaker-hash>/*.wav
    ├── 1/asr_javanese/data/<speaker-hash>/*.wav
    └── ...
```

Each dataset shard should also contain its `utt_spk_text.tsv` transcript file. The sampling script creates a sampled audio tree and a CSV containing `Path_Data` and `sentences` columns.

Before running anything, change the path constants at the top of these scripts:

- `preModelling/src/stratified_sampling.py`
- `preModelling/src/preprocessing.py`
- `kaldialignment/makeAligner/src/Fase_1_CreateAligner.py`
- `kaldialignment/makeAligner/src/Fase_2v2_GMMHMMAligner.py`
- `TDNN-TDNN-F/src/TDNN.py`
- `TDNN-TDNN-F/src/TDNNF.py`

The current scripts were developed with Windows path conventions. Replace the hard-coded paths with local paths or refactor them to use `pathlib.Path` before running on Linux or macOS.

## Running the workflow

Run each stage from the repository root. The scripts are separate so intermediate data can be inspected and reused.

```bash
python src/codeStat/modelling/voiceModelling/preModelling/src/stratified_sampling.py
python src/codeStat/modelling/voiceModelling/preModelling/src/preprocessing.py

# Create and validate alignment inputs, then run GMM-HMM alignment.
python src/codeStat/modelling/voiceModelling/kaldialignment/makeAligner/src/Fase_1_CreateAligner.py
python src/codeStat/modelling/voiceModelling/kaldialignment/makeAligner/src/Fase_2v2_GMMHMMAligner.py

# Train/evaluate the neural acoustic models after manifests and alignments exist.
python src/codeStat/modelling/voiceModelling/TDNN-TDNN-F/src/TDNN.py
python src/codeStat/modelling/voiceModelling/TDNN-TDNN-F/src/TDNNF.py
```

The notebooks in the corresponding `src/` directories provide an interactive alternative for preprocessing, alignment, and model experiments.

## Model configuration

Shared defaults are defined in `Shared_resource.py`:

| Setting | Default | Purpose |
|---|---:|---|
| Sample rate | 16,000 Hz | Audio resampling target |
| MFCC coefficients | 13 | Base acoustic features |
| Feature input dimension | 27 | Dimension consumed by the acoustic models after preparation |
| TDNN hidden size | 512 | Hidden width of TDNN layers |
| TDNN layers | 8 | Number of TDNN layers |
| TDNN-F bottleneck | 128 | Factorized projection width |
| Training epochs | 30 | Maximum training epochs |
| Early-stopping patience | 7 | Epochs without improvement |
| Batch size | 64 | Training batch size |

The exact output dimension, contexts, learning-rate schedule, augmentation, and decoding settings can be changed through the `Config` dataclass and model-specific configuration blocks.

## Outputs

Typical generated or tracked outputs include:

- `dataBersih.csv` — validated and normalized transcript metadata.
- `dropped.csv` and `dropenglist.csv` — records removed during preprocessing.
- Alignment maps and train/validation manifests — links between audio, frames, labels, and alignments.
- `lm.pkl` — serialized n-gram language-model data used during decoding.
- `*_best.pt` and `*_prior.pt` — TDNN/TDNN-F checkpoints and class priors.
- `*_history.json` — training history.
- `*_cer_final.csv` — utterance-level or summary CER results.

## Reproducibility notes

- Set the configured paths before running a stage; paths embedded in scripts point to the original development environment.
- Keep dataset, MFA, dictionary, and acoustic-model paths outside the repository when possible.
- The sampling workflow uses seed `42` by default.
- Check whether a cached MFCC or alignment artifact is compatible with the current feature and label configuration before reusing it.
- Training automatically uses CUDA when available and otherwise falls back to CPU.

## Project status

This is an experimental research codebase. The pipeline and model implementations are available, but the workflow still requires manual path configuration and external MFA assets. Results should be treated as experiment outputs rather than a production ASR release.

## Data and attribution

The project was developed for Javanese speech recognition using data associated with the Google–Universitas Gadjah Mada collaboration and Yogyakarta speech data. Confirm the dataset license and redistribution terms before sharing audio, transcripts, alignments, or trained models.
