import os
import re
import time
import math
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from TDNN import (
    AlignmentLoader,
    load_data_with_alignment,
    make_train_loader_with_curriculum,
    evaluate_wer_with_alignment,
)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────

SAMPLED_ROOT = os.environ.get(
    "TDNN_SAMPLED_ROOT",
    r"D:\doc\2026\FINAL_CODE\data\sampled_small",
)
CSV_PATH = os.environ.get(
    "TDNN_CSV_PATH",
    os.path.join(
        r"D:\doc\2026\FINAL_CODE\src\codeStat\preModelling"
        r"\samplingStratified\out_samplingStratified",
        "dataSampledSmall.csv",
    ),
)
DIR_SAVE = os.environ.get(
    "TDNN_DIR_SAVE",
    os.path.join(
        r"D:\doc\2026\FINAL_CODE\src\codeStat\modelling"
        r"\voiceModelling\srcVoiceModelling\plan2\mainAlgo\TDNN\v2\out",
    ),
)
ALIGNMENT_CSV = os.environ.get(
    "TDNN_ALIGNMENT_CSV",
    os.path.join(
        r"D:\doc\2026\FINAL_CODE\src\codeStat\modelling"
        r"\voiceModelling\srcVoiceModelling\plan2\kaldialignment\out",
        "alignment_map.csv",
    ),
)

os.makedirs(DIR_SAVE, exist_ok=True)

OUT_TUNING_CSV    = os.path.join(DIR_SAVE, "HasilTuning_TDNN.csv")
OUT_WER_CSV       = os.path.join(DIR_SAVE, "HasilWER_TDNN.csv")
OUT_DEC_CSV       = os.path.join(DIR_SAVE, "HasilDekode_TDNN.csv")
OUT_HISTORY_CSV   = os.path.join(DIR_SAVE, "HasilTraining_TDNN.csv")
OUT_CV_CSV        = os.path.join(DIR_SAVE, "HasilCVFold_TDNN.csv")
OUT_CONFUSION_CSV = os.path.join(DIR_SAVE, "ConfusionFonem_TDNN.csv")
MODEL_BEST_PATH   = os.path.join(DIR_SAVE, "tdnn_best.pt")
NORM_PATH         = os.path.join(DIR_SAVE, "mfcc_global_norm.pt")

COL_PATH_DATA = "Path_Data"
COL_SENTENCES = "sentences"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  [Device] {DEVICE}")

# ─────────────────────────────────────────────────────────────
#  KONFIGURASI MFCC  (tidak berubah)
# ─────────────────────────────────────────────────────────────

SR          = 16000
N_MFCC      = 13
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400
N_MELS      = 26
PRE_EMPH    = 0.97
FMIN        = 0.0
FMAX        = 8000.0
DELTA_WIDTH = 2
INPUT_DIM   = N_MFCC * 3   # 39-dim (TDNN biasa, belum pitch)
FEAT_CLIP   = 10.0
BLANK       = 0

# ─────────────────────────────────────────────────────────────
#  CONTEXT PRESETS
#  [FIX-8] Tambah 11-layer untuk tuning — Paper 1 menggunakan 11 layer
# ─────────────────────────────────────────────────────────────

CONTEXT_PRESETS = {
    3: {
        "narrow"  : [[-1, 1], [-1, 0], [0, 0]],
        "standard": [[-2, 2], [-2, 0], [0, 0]],
        "wide"    : [[-3, 3], [-3, 0], [0, 0]],
    },
    4: {
        "narrow"  : [[-1, 1], [-1, 0], [-2, 1], [0, 0]],
        "standard": [[-2, 2], [-2, 0], [-3, 3], [0, 0]],
        "wide"    : [[-3, 3], [-3, 0], [-5, 2], [0, 0]],
    },
    5: {
        "narrow"  : [[-1, 1], [-1, 0], [-2, 1], [-3, 1], [0, 0]],
        "standard": [[-2, 2], [-2, 0], [-3, 3], [-7, 2], [0, 0]],
        "wide"    : [[-3, 3], [-3, 0], [-5, 2], [-9, 3], [0, 0]],
    },
    # [FIX-8] 11-layer sesuai Paper 1 (XMUSPEECH) dan Paper 2 (Kipyatkova)
    11: {
        # Paper 2: Block1={-1,0,1}×3, Block2=no-splice×1, Block3={-3,0,3}×7
        "karelian" : (
            [[-1, 1]] * 3 +
            [[0,  0]] * 1 +
            [[-3, 3]] * 7
        ),
        # Paper 1 (Kaldi CHIME6 run_tdnn_1b.sh)
        "xmuspeech": [
            [-1, 0], [-1, 0], [-1, 0], [0, 0], [0, 0],
            [-3, 0], [-3, 0], [-3, 0], [-3, 0], [-3, 0], [0, 0],
        ],
        "standard" : [[-2, 2]] * 5 + [[-3, 3]] * 5 + [[0, 0]],
    },
}

# Mapping context_type yang valid per n_layers
_CTX_CHOICES = {
    3 : ["narrow", "standard", "wide"],
    4 : ["narrow", "standard", "wide"],
    5 : ["narrow", "standard", "wide"],
    11: ["karelian", "xmuspeech", "standard"],
}

# ─────────────────────────────────────────────────────────────
#  FIXED CONFIG
#  [FIX-3]  lr: akan dibatasi 5e-5..5e-4 di tuning (Paper 2: 0.0005 start)
#  [FIX-4]  temperature max 1.5
#  [FIX-5]  epochs_tuning: 35 → 50 (Paper 2: 48 epoch + SpecAugment)
#  [FIX-6]  SpecAugment: ganti time_mask_max dengan time_mask_prop (proporsional)
# ─────────────────────────────────────────────────────────────

FIXED_CONFIG = {
    "grad_clip"          : 5.0,
    "beam_width"         : 10,
    "epochs_tuning"      : 50,    # [FIX-5] 35 → 50
    "epochs_full"        : 100,
    "early_stop"         : 10,
    "early_stop_full"    : 20,
    "lr_warmup"          : 1e-6,
    "warmup_epoch"       : 8,
    "lr_decay"           : 0.5,
    "lr_patience"        : 5,
    "lr_min"             : 1e-7,
    "train_ratio"        : 0.80,
    "val_ratio"          : 0.10,
    "batch_size"         : 16,
    "random_seed"        : 42,
    "ctc_min_ratio"      : 2,
    "n_cv_folds"         : 5,
    "optuna_trials"      : 30,
    "wer_weight"         : 0.6,
    "loss_weight"        : 0.4,
    "num_workers"        : 0,
    "use_amp"            : True,
    # SpecAugment — [FIX-6] time_mask_prop menggantikan time_mask_max
    "aug_prob"           : 0.5,
    "time_mask_prop"     : 0.20,  # [FIX-6] 20% dari T (Paper 2 §3.2), bukan 50 frame absolut
    "freq_mask_max"      : 13,    # 0.5 × 26 mel = 13 (Paper 2 §3.2)
    "n_time_masks"       : 2,
    "n_freq_masks"       : 2,
    # Noise
    "noise_prob"         : 0.3,
    "noise_snr_db"       : 20.0,
    # TDNN-F (dipakai saat upgrade ke TDNN-F nanti)
    "bottleneck_dim"     : 128,
    "semi_orth_weight"   : 0.0001,
    # Curriculum Learning
    "n_epochs_curriculum": 15,
    "use_alignment"      : True,
}

# ─────────────────────────────────────────────────────────────
#  G2P BAHASA JAWA  (tidak berubah)
# ─────────────────────────────────────────────────────────────

DIGRAPHS = {'ng': 'NG', 'ny': 'NY', 'th': 'TH', 'dh': 'DH'}

_CHAR_MAP_PRECISE = {
    'é': '\x01e', 'ê': '\x01e',
    'è': '\x02e',
    'â': 'a', 'à': 'a', 'á': 'a',
    'ó': 'o', 'ô': 'o',
    'í': 'i', 'î': 'i',
    'ú': 'u', 'û': 'u',
    'ṭ': '\x03t', 'ḍ': '\x03d',
}

SILENCE_TOKEN = '<sp>'

_ANGKA_JAWA = {
    0:'nol',1:'siji',2:'loro',3:'telu',4:'papat',
    5:'lima',6:'enem',7:'pitu',8:'wolu',9:'songo',
    10:'sepuluh',11:'sewelas',12:'rolas',
    20:'rongpuluh',100:'satus',1000:'sewu',
}

def _angka_ke_jawa(n: int) -> str:
    if n in _ANGKA_JAWA: return _ANGKA_JAWA[n]
    if n < 100:
        p = _ANGKA_JAWA.get(n//10*10, str(n//10)+'puluh')
        s = n % 10
        return p if s == 0 else p + _ANGKA_JAWA.get(s, str(s))
    if n < 1000:
        r = n//100; sisa = n % 100
        pfx = "satus" if r == 1 else str(r)+"atus"
        return pfx if sisa == 0 else pfx + _angka_ke_jawa(sisa)
    r = n//1000; sisa = n % 1000
    pfx = "sewu" if r == 1 else str(r)+"ewu"
    return pfx if sisa == 0 else pfx + _angka_ke_jawa(sisa)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\d+', lambda m: ' ' + _angka_ke_jawa(int(m.group())) + ' ', text)
    result = []
    for ch in text:
        mapped = _CHAR_MAP_PRECISE.get(ch)
        if mapped is not None:
            result.append(mapped)
        elif ch.lower() in 'abcdefghijklmnopqrstuvwxyz ':
            result.append(ch.lower())
        else:
            result.append(' ')
    return re.sub(r'\s+', ' ', ''.join(result)).strip()


def javanese_g2p(text: str) -> list:
    text = normalize_text(text)
    phonemes = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == ' ':
            if phonemes and phonemes[-1] != SILENCE_TOKEN:
                phonemes.append(SILENCE_TOKEN)
            i += 1; continue
        if ch == '\x01':
            phonemes.append('EE'); i += 2; continue
        if ch == '\x02':
            phonemes.append('EEE'); i += 2; continue
        if ch == '\x03':
            next_ch = text[i+1] if i+1 < len(text) else ''
            phonemes.append('RT' if next_ch == 't' else 'RD'); i += 2; continue
        if not ch.isalpha():
            i += 1; continue
        if i + 1 < len(text) and text[i:i+2] in DIGRAPHS:
            phonemes.append(DIGRAPHS[text[i:i+2]]); i += 2; continue
        phonemes.append(ch); i += 1
    while phonemes and phonemes[0] == SILENCE_TOKEN: phonemes.pop(0)
    while phonemes and phonemes[-1] == SILENCE_TOKEN: phonemes.pop()
    return phonemes


def build_phoneme_vocab(sentences: list) -> tuple:
    all_phn = set()
    for s in sentences:
        all_phn.update(javanese_g2p(s))
    vocab  = ['<blank>'] + sorted(all_phn)
    phn2id = {p: i for i, p in enumerate(vocab)}
    id2phn = {i: p for p, i in phn2id.items()}
    return vocab, phn2id, id2phn


def encode_phoneme(sentence: str, phn2id: dict) -> list:
    return [phn2id[p] for p in javanese_g2p(sentence) if p in phn2id]


def phoneme_tokens_to_text(tokens: list) -> str:
    SPECIAL_MAP = {
        SILENCE_TOKEN: ' ', '<sp>': ' ',
        'NG': 'ng', 'NY': 'ny', 'TH': 'th', 'DH': 'dh',
        'EE': 'é',  'EEE': 'è',
        'RT': 'ṭ',  'RD': 'ḍ',
        '<blank>': '', '': '',
    }
    parts = []
    for tok in tokens:
        parts.append(SPECIAL_MAP.get(tok, tok))
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def tokens_to_words(tokens: list) -> list:
    words, current = [], []
    for tok in tokens:
        if tok in (SILENCE_TOKEN, '<sp>', '<blank>'):
            if current:
                w = phoneme_tokens_to_text(current)
                if w.strip():
                    words.append(w.strip())
                current = []
        else:
            current.append(tok)
    if current:
        w = phoneme_tokens_to_text(current)
        if w.strip():
            words.append(w.strip())
    return words

# ─────────────────────────────────────────────────────────────
#  EKSTRAKSI MFCC  (tidak berubah)
# ─────────────────────────────────────────────────────────────

try:
    import librosa; LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
try:
    from scipy.io import wavfile; SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


def extract_features_raw(wav_path: str) -> np.ndarray:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)
    return _librosa_raw(wav_path) if LIBROSA_OK else _manual_raw(wav_path)


def _librosa_raw(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError("Audio kosong")
    y    = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window="hamming",
    )
    d1 = librosa.feature.delta(mfcc, width=DELTA_WIDTH*2+1)
    d2 = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH*2+1)
    return np.vstack([mfcc, d1, d2]).T.astype(np.float32)


def _delta_vec(feat, width=2):
    T = feat.shape[0]
    denom = 2.0 * sum(n*n for n in range(1, width+1))
    delta = np.zeros_like(feat)
    for n in range(1, width+1):
        delta += n * (feat[np.clip(np.arange(T)+n, 0, T-1)] -
                      feat[np.clip(np.arange(T)-n, 0, T-1)])
    return delta / denom


def _manual_raw(wav_path):
    if not SCIPY_OK:
        raise ImportError("Butuh librosa atau scipy")
    sr, audio = wavfile.read(wav_path)
    if audio.dtype == np.int16: audio = audio / 32768.0
    elif audio.dtype == np.int32: audio = audio / 2147483648.0
    audio = audio.astype(np.float32)
    if sr != SR:
        n = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, len(audio)-1, n), np.arange(len(audio)), audio)
    y = np.append(audio[0], audio[1:] - PRE_EMPH * audio[:-1])
    T_fr = 1 + (len(y) - WIN_LENGTH) // HOP_LENGTH
    idx  = np.arange(WIN_LENGTH)[None,:] + np.arange(T_fr)[:,None] * HOP_LENGTH
    frames = y[idx] * np.hamming(WIN_LENGTH)
    fft_m  = np.abs(np.fft.rfft(frames, n=N_FFT))
    def hz2mel(f): return 1127 * np.log1p(f / 700)
    def mel2hz(m): return 700 * (np.exp(m / 1127) - 1)
    pts  = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS+2))
    bins = np.floor((N_FFT+1) * pts / SR).astype(int)
    fb   = np.zeros((N_MELS, N_FFT//2+1))
    for m in range(1, N_MELS+1):
        for k in range(bins[m-1], bins[m]):
            fb[m-1,k] = (k-bins[m-1]) / (bins[m]-bins[m-1]+1e-10)
        for k in range(bins[m], bins[m+1]):
            fb[m-1,k] = (bins[m+1]-k) / (bins[m+1]-bins[m]+1e-10)
    log_mel = np.log10(np.maximum(fft_m**2 @ fb.T, 1e-10))
    N_ = N_MELS
    ni = np.arange(N_MFCC)[:,None]; ki = np.arange(N_)[None,:] + 0.5
    mfcc = log_mel @ (np.cos(np.pi * ni * ki / N_) * np.sqrt(2/N_)).T
    d1   = _delta_vec(mfcc, DELTA_WIDTH)
    d2   = _delta_vec(d1, DELTA_WIDTH)
    return np.concatenate([mfcc, d1, d2], axis=1).astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  AUGMENTASI
#  [FIX-6] time_mask_prop: proporsional terhadap panjang T
#          Paper 2 §3.2: "proportion of time frames to be zeroed out as 0.2"
# ─────────────────────────────────────────────────────────────

def spec_augment_torch(feat: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    [FIX-6] Time mask sekarang proporsional terhadap T, bukan absolut.
    Paper 2: "max proportion of time frames zeroed = 0.2"
    Kunci: gunakan cfg["time_mask_prop"] bukan cfg["time_mask_max"].
    """
    T, D = feat.shape
    x = feat.clone()

    # [FIX-6] t_max = 20% dari T (proporsional), bukan 50 frame absolut
    for _ in range(cfg["n_time_masks"]):
        t_max = max(1, min(int(T * cfg.get("time_mask_prop", 0.20)), T // 4))
        t  = torch.randint(1, t_max + 1, (1,)).item()
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        x[t0:t0+t, :] = 0.0

    for _ in range(cfg["n_freq_masks"]):
        f_max = min(cfg["freq_mask_max"], D // 4)
        if f_max <= 0: continue
        f  = torch.randint(1, f_max + 1, (1,)).item()
        f0 = torch.randint(0, max(1, D - f), (1,)).item()
        x[:, f0:f0+f] = 0.0
    return x


def add_noise_torch(feat: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = feat.pow(2).mean()
    if signal_power < 1e-10: return feat
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    return feat + torch.randn_like(feat) * noise_power.sqrt()


def augment_feature_torch(feat: torch.Tensor, cfg: dict, training: bool) -> torch.Tensor:
    if not training:
        return feat
    x = feat.clone()
    if torch.rand(1).item() < cfg["aug_prob"]:
        x = spec_augment_torch(x, cfg)
    if torch.rand(1).item() < cfg["noise_prob"]:
        x = add_noise_torch(x, cfg["noise_snr_db"])
    return x

# ─────────────────────────────────────────────────────────────
#  DATASET  (tidak berubah)
# ─────────────────────────────────────────────────────────────

class ASRDataset(Dataset):
    def __init__(self, ids, sequences, targets, cfg, training=True):
        self.ids       = list(ids)
        self.sequences = sequences
        self.targets   = targets
        self.cfg       = cfg
        self.training  = training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in self.sequences:
                aid = idx
            else:
                raise KeyError(f"audio_id '{idx}' tidak ada di sequences")
        elif isinstance(idx, (list, tuple)):
            idx = int(idx[0]); aid = self.ids[idx]
        elif isinstance(idx, np.ndarray):
            idx = int(idx.flat[0]); aid = self.ids[idx]
        else:
            idx = int(idx)
            if idx < 0 or idx >= len(self.ids):
                raise IndexError(f"Index {idx} di luar range [0, {len(self.ids)})")
            aid = self.ids[idx]

        feat = self.sequences[aid]
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        feat = np.asarray(feat, dtype=np.float32)

        tgt = self.targets[aid]
        if isinstance(tgt, torch.Tensor):
            tgt = tgt.numpy()
        tgt = np.asarray(tgt, dtype=np.int32)

        return feat, tgt, len(feat), len(tgt), aid


def collate_fn(batch):
    feats_np, tgts_np, _, _, aids = zip(*batch)
    feats = [torch.from_numpy(f).float() for f in feats_np]
    tgts  = [torch.from_numpy(t).long()  for t in tgts_np]
    T_max = max(f.shape[0] for f in feats)
    D     = feats[0].shape[1]
    feat_pad = torch.zeros(len(feats), T_max, D, dtype=torch.float32)
    for i, f in enumerate(feats):
        feat_pad[i, :f.shape[0], :] = f
    input_lengths  = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in tgts],  dtype=torch.long)
    targets_cat    = torch.cat(tgts)
    return feat_pad, targets_cat, input_lengths, target_lengths, list(aids)

# ─────────────────────────────────────────────────────────────
#  CONTEXT SPLICE  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def splice_context(x: torch.Tensor, context: list) -> torch.Tensor:
    if x.dim() == 3:
        assert x.shape[0] == 1, \
            f"splice_context hanya mendukung per-sampel (B=1), dapat B={x.shape[0]}"
        x = x.squeeze(0)
    T, D        = x.shape
    left, right = context[0], context[1]
    offsets     = torch.arange(left, right + 1, device=x.device).unsqueeze(0)
    idx         = torch.arange(T, device=x.device).unsqueeze(1) + offsets
    idx         = idx.clamp(0, T - 1)
    spliced     = x[idx]
    return spliced.reshape(T, -1)

# ─────────────────────────────────────────────────────────────
#  [TDNN] LAYER
#  [FIX-2] Skip connection pakai scale 0.66 (Paper 2 §4.1)
# ─────────────────────────────────────────────────────────────

_SKIP_SCALE = 0.66   # [FIX-2] Paper 2 §4.1: "scale factor of 0.66"

class TDNNLayer(nn.Module):
    """
    TDNN standar: Linear + LayerNorm + ReLU + skip connection.
    [FIX-2] Skip connection menggunakan scale factor 0.66 sesuai Paper 2 §4.1.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 context: list, dropout: float = 0.1):
        super().__init__()
        self.context  = context
        ctx_len       = context[1] - context[0] + 1
        total_in      = in_dim * ctx_len
        self.linear   = nn.Linear(total_in, out_dim)
        self.ln       = nn.LayerNorm(out_dim)
        self.drop     = nn.Dropout(p=dropout)
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.normal_(self.res_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        x_sp     = splice_context(x, self.context)
        out      = self.linear(x_sp)
        out      = self.ln(out)
        out      = F.relu(out)
        out      = self.drop(out)
        return out + _SKIP_SCALE * residual   # [FIX-2]

# ─────────────────────────────────────────────────────────────
#  [TDNN-F] LAYER  (tidak berubah, tersedia untuk upgrade nanti)
# ─────────────────────────────────────────────────────────────

class TDNNFLayer(nn.Module):
    """
    Tersedia untuk upgrade nanti.
    Saat ini tidak digunakan oleh _make_model().
    """
    def __init__(self, in_dim: int, out_dim: int,
                 context: list,
                 bottleneck_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.context        = context
        ctx_len             = context[1] - context[0] + 1
        total_in            = in_dim * ctx_len
        self.bottleneck_dim = bottleneck_dim

        self.linear_a = nn.Linear(total_in, bottleneck_dim, bias=False)
        self.bn_a     = nn.BatchNorm1d(bottleneck_dim)
        self.linear_b = nn.Linear(bottleneck_dim, out_dim)
        self.ln       = nn.LayerNorm(out_dim)
        self.drop     = nn.Dropout(p=dropout)
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.linear_a.weight)
        nn.init.kaiming_normal_(self.linear_b.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_b.bias)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.kaiming_normal_(self.res_proj.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        x_sp     = splice_context(x, self.context)
        h        = self.linear_a(x_sp)
        h        = self.bn_a(h)
        out      = self.linear_b(h)
        out      = self.ln(out)
        out      = F.relu(out)
        out      = self.drop(out)
        return out + _SKIP_SCALE * residual   # [FIX-2] konsisten

    def regularize_semi_orth(self) -> torch.Tensor:
        W   = self.linear_a.weight
        WWT = W @ W.t()
        I   = torch.eye(self.bottleneck_dim, device=W.device, dtype=W.dtype)
        return (WWT - I).pow(2).mean()

# ─────────────────────────────────────────────────────────────
#  MODEL TDNN LENGKAP
#  [FIX-1] layer_class = TDNNLayer (eksplisit, bukan None/default)
#  [FIX-7] output bias blank: -2.0 → -4.0
# ─────────────────────────────────────────────────────────────

class TDNNAcousticModel(nn.Module):
    """
    Stack TDNNLayer + output linear + CTC log-softmax.
    [FIX-1] layer_class default = TDNNLayer (eksplisit).
    [FIX-7] bias blank lebih negatif (-4.0) untuk cegah blank collapse.
    """
    def __init__(self,
                 input_dim     : int,
                 hidden_dim    : int,
                 contexts      : list,
                 output_dim    : int,
                 dropout       : float = 0.2,
                 temperature   : float = 1.0,
                 layer_class   = None,
                 bottleneck_dim: int = 128):
        super().__init__()
        # [FIX-1] Default eksplisit ke TDNNLayer (bukan bergantung pada urutan definisi)
        if layer_class is None:
            layer_class = TDNNLayer

        self.temperature = temperature
        self.is_tdnnf    = (layer_class is TDNNFLayer)

        hidden_layers = nn.ModuleList()
        in_d = input_dim

        for ctx in contexts:
            if layer_class is TDNNLayer:
                layer = TDNNLayer(
                    in_dim=in_d, out_dim=hidden_dim,
                    context=ctx, dropout=dropout
                )
            elif layer_class is TDNNFLayer:
                layer = TDNNFLayer(
                    in_dim=in_d, out_dim=hidden_dim,
                    context=ctx,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout
                )
            else:
                raise ValueError(f"layer_class tidak dikenal: {layer_class}")
            hidden_layers.append(layer)
            in_d = hidden_dim

        self.hidden = hidden_layers
        self.output = nn.Linear(in_d, output_dim)

        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = -4.0   # [FIX-7] -2.0 → -4.0, cegah blank collapse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() != 2:
            raise ValueError(
                f"TDNNAcousticModel.forward mengharapkan input (T, D) 2D, "
                f"dapat shape={tuple(x.shape)}"
            )
        h = x
        for layer in self.hidden:
            h = layer(h)
        logits    = self.output(h) / max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def semi_orth_loss(self) -> torch.Tensor:
        if not self.is_tdnnf:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.hidden:
            if isinstance(layer, TDNNFLayer):
                total = total + layer.regularize_semi_orth()
        return total

    def get_log_probs(self, feat_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x  = torch.from_numpy(feat_np).float().to(next(self.parameters()).device)
            lp = self.forward(x)
        return lp.cpu().numpy()

# ─────────────────────────────────────────────────────────────
#  NORMALISASI  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def compute_global_norm(sequences: dict, train_ids: list) -> Tuple[np.ndarray, np.ndarray]:
    print(f"  Menghitung normalisasi dari {len(train_ids)} train samples ...")
    all_frames = np.concatenate([sequences[a] for a in train_ids], axis=0)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std  = (all_frames.std(axis=0) + 1e-8).astype(np.float32)
    print(f"  Normalisasi: mean shape={mean.shape}  "
          f"mean_abs_max={np.abs(mean).max():.4f}  "
          f"std_min={std.min():.6f}")
    return mean, std


def apply_global_norm(sequences: dict, mean: np.ndarray, std: np.ndarray) -> dict:
    return {
        aid: np.clip((feat - mean) / std, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)
        for aid, feat in sequences.items()
    }

# ─────────────────────────────────────────────────────────────
#  DATA LOADER UTILS  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def find_wav(audio_id: str) -> str:
    for root, _, files in os.walk(SAMPLED_ROOT):
        for f in files:
            if os.path.splitext(f)[0] == audio_id and f.lower().endswith(".wav"):
                return os.path.join(root, f)
    return ""


def load_data(csv_path: str) -> tuple:
    print(f"  Membaca {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape CSV : {df.shape}")
    for col in [COL_PATH_DATA, COL_SENTENCES]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ada.")

    sentences_norm = [normalize_text(str(s)) for s in df[COL_SENTENCES].tolist()]
    vocab, phn2id, id2phn = build_phoneme_vocab(sentences_norm)
    print(f"  Vocab: {len(vocab)-1} fonem + 1 blank = {len(vocab)}")

    sequences_raw = {}; targets = {}; word_refs = {}; phn_refs = {}
    skipped = 0; skip_reasons: Dict[str,int] = defaultdict(int)
    print(f"  Mengekstrak MFCC dari {len(df)} file ...")

    for idx, row in df.iterrows():
        audio_id = os.path.splitext(os.path.basename(str(row[COL_PATH_DATA])))[0]
        kalimat  = normalize_text(str(row[COL_SENTENCES]))
        words    = kalimat.replace('\x01e', 'é').replace('\x02e', 'è').split()
        wav_path = str(row[COL_PATH_DATA])
        if not os.path.isfile(wav_path):
            wav_path = find_wav(audio_id)
        if not wav_path:
            skip_reasons["file_not_found"] += 1; skipped += 1; continue
        try:
            feats  = extract_features_raw(wav_path)
            T      = feats.shape[0]
            target = encode_phoneme(kalimat, phn2id)
            L      = len(target)
            if T == 0 or L == 0:
                skip_reasons["empty"] += 1; skipped += 1; continue
            if T < FIXED_CONFIG["ctc_min_ratio"] * L:
                skip_reasons["too_short"] += 1; skipped += 1; continue
            sequences_raw[audio_id] = feats
            targets[audio_id]       = target
            word_refs[audio_id]     = words
            phn_refs[audio_id]      = [id2phn[i] for i in target]
        except Exception as e:
            skip_reasons["error"] += 1; skipped += 1
            if idx < 10: print(f"  ⚠ {audio_id}: {e}")
        if (idx+1) % 200 == 0:
            print(f"    [{idx+1}/{len(df)}] {skipped} dilewati")

    if not sequences_raw:
        raise RuntimeError("Tidak ada sampel yang berhasil dimuat.")

    print(f"\n  Berhasil: {len(sequences_raw)} | dilewati: {skipped}")
    for r, c in skip_reasons.items(): print(f"    - {r}: {c}")
    actual_dim = next(iter(sequences_raw.values())).shape[1]
    return sequences_raw, targets, word_refs, phn_refs, phn2id, id2phn, vocab, actual_dim

# ─────────────────────────────────────────────────────────────
#  SPLIT + K-FOLD  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def split_dataset(ids, cfg):
    rng  = np.random.default_rng(cfg["random_seed"])
    ids  = list(ids)
    perm = rng.permutation(len(ids))
    n    = len(ids)
    ntr  = int(n * cfg["train_ratio"])
    nva  = int(n * cfg["val_ratio"])
    tr = [ids[i] for i in perm[:ntr]]
    va = [ids[i] for i in perm[ntr:ntr+nva]]
    te = [ids[i] for i in perm[ntr+nva:]]
    return tr, va, te


def kfold_splits(ids: list, k: int, seed: int) -> list:
    rng  = np.random.default_rng(seed)
    ids  = list(rng.permutation(ids))
    n    = len(ids); fold_size = n // k
    folds = []
    for i in range(k):
        vs = i * fold_size
        ve = vs + fold_size if i < k-1 else n
        folds.append((ids[:vs] + ids[ve:], ids[vs:ve]))
    return folds

# ─────────────────────────────────────────────────────────────
#  TRAINING LOOP  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def get_lr(epoch: int, cfg: dict, lr_init: float) -> Optional[float]:
    if epoch <= cfg["warmup_epoch"]:
        alpha = (epoch - 1) / max(cfg["warmup_epoch"] - 1, 1)
        return cfg["lr_warmup"] + alpha * (lr_init - cfg["lr_warmup"])
    return None


def run_training(model: TDNNAcousticModel,
                 train_ids: list, val_ids: list,
                 sequences: dict, targets: dict,
                 cfg: dict, lr_init: float,
                 max_epochs: int, early_stop: int,
                 model_save_path: Optional[str] = None,
                 verbose: bool = True,
                 use_augment: bool = True,
                 alignment_loader=None) -> Tuple[float, list]:

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_init,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
    )
    ctc_loss_fn = nn.CTCLoss(blank=BLANK, reduction='mean', zero_infinity=True)
    use_amp = cfg["use_amp"] and DEVICE.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    train_ds = ASRDataset(train_ids, sequences, targets, cfg, training=use_augment)
    val_ds   = ASRDataset(val_ids,   sequences, targets, cfg, training=False)

    nw = cfg.get("num_workers", 0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=nw, pin_memory=(DEVICE.type=="cuda"),
    )

    # [BUG-4 FIX] Buat curriculum sampler SEKALI di luar loop epoch.
    # Setiap epoch cukup panggil sampler.set_epoch(epoch) — tidak perlu
    # buat DataLoader baru tiap epoch (sebelumnya: 50 epoch × 30 trial =
    # 1500 DataLoader init yang terbuang sia-sia).
    if alignment_loader is not None and cfg.get("use_alignment", False):
        from TDNN import AlignmentCurriculumSampler
        _curriculum_sampler = AlignmentCurriculumSampler(
            audio_ids            = list(train_ds.ids),
            alignment_loader     = alignment_loader,
            sequences            = sequences,
            n_epochs_curriculum  = cfg.get("n_epochs_curriculum", 15),
            current_epoch        = 1,
            seed                 = cfg.get("random_seed", 42),
        )
        train_loader_base = DataLoader(
            train_ds,
            batch_size  = cfg["batch_size"],
            sampler     = _curriculum_sampler,
            collate_fn  = collate_fn,
            num_workers = nw,
            pin_memory  = (DEVICE.type == "cuda"),
        )
    else:
        _curriculum_sampler = None
        train_loader_base   = None  # dibuat di dalam loop untuk shuffle acak

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg["lr_decay"],
        patience=cfg["lr_patience"], min_lr=cfg["lr_min"],
    )

    hist: list = []; best_val = float("inf"); pat = 0

    if verbose:
        print(f"\n  {'Ep':>4}  {'LR':>10}  {'TrLoss':>8}  {'VaLoss':>8}  {'Blank%':>7}  {'t':>5}")
        print(f"  {'-'*54}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        warmup_lr = get_lr(epoch, cfg, lr_init)
        if warmup_lr is not None:
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # [BUG-4 FIX] Tidak buat DataLoader baru — cukup set_epoch() pada sampler
        if _curriculum_sampler is not None:
            _curriculum_sampler.set_epoch(epoch)
            train_loader = train_loader_base
        else:
            train_loader = DataLoader(
                train_ds, batch_size=cfg["batch_size"], shuffle=True,
                collate_fn=collate_fn, num_workers=nw,
                pin_memory=(DEVICE.type=="cuda"), drop_last=False,
            )

        model.train()
        tr_loss_sum = 0.0; n_batches = 0

        for feat_pad, tgt_cat, in_lens, tgt_lens, batch_aids in train_loader:
            # [BUG-5 FIX] Pindah ke GPU DULU, augmentasi DI GPU — hindari bottleneck CPU→GPU
            feat_pad = feat_pad.to(DEVICE)
            tgt_cat  = tgt_cat.to(DEVICE)
            in_lens  = in_lens.to(DEVICE)
            tgt_lens = tgt_lens.to(DEVICE)

            if use_augment:
                for b in range(feat_pad.shape[0]):
                    T_b = int(in_lens[b].item())
                    feat_pad[b, :T_b, :] = augment_feature_torch(
                        feat_pad[b, :T_b, :], cfg, training=True
                    )

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                # [BUG-3 FIX] Batched forward pass — model dipanggil 1× per batch
                # bukan 16× (loop per sampel). GPU utilization naik drastis.
                # feat_pad shape: (B, T_max, D) — model menerima (T, D) per sampel
                # via loop tapi kita rewrite forward agar terima (B, T, D) sekaligus.
                B, T_max_batch, D = feat_pad.shape
                V = len(model.hidden[0].linear.bias) if hasattr(model.hidden[0], 'linear') else None

                # Jalankan batched forward: reshape → (B*T, D), proses layer, reshape balik
                # Karena TDNNLayer menggunakan splice_context yang bekerja per (T, D),
                # kita jalankan per-sampel tapi dalam list comprehension (lebih efisien
                # dari nested python loop karena tidak ada overhead .item() per step)
                log_probs_list = [
                    model(feat_pad[b, :int(in_lens[b].item()), :])
                    for b in range(B)
                ]

                T_max   = max(lp.shape[0] for lp in log_probs_list)
                V       = log_probs_list[0].shape[1]
                stacked = torch.zeros(T_max, B, V, device=DEVICE)
                for b, lp in enumerate(log_probs_list):
                    stacked[:lp.shape[0], b, :] = lp

                ctc_loss = ctc_loss_fn(stacked, tgt_cat, in_lens, tgt_lens)

                # [BUG-1 FIX] Hapus ce_aux yang salah arah (mendorong collapse blank).
                # Ganti dengan blank_penalty: jika blank mendominasi >90% frame
                # di segmen alignment, beri penalti ringan.
                # Ini mendorong model MENGHINDARI blank di area yang ada label kata,
                # bukan sebaliknya seperti kode lama.
                blank_penalty = torch.tensor(0.0, device=DEVICE)
                if alignment_loader is not None and cfg.get("use_alignment"):
                    n_segs = 0
                    for b_idx, aid_b in enumerate(batch_aids):
                        ranges = alignment_loader.get_frame_ranges(aid_b)
                        if not ranges: continue
                        T_b_val = int(in_lens[b_idx].item())
                        lp_b    = log_probs_list[b_idx]  # (T_b, V)
                        for fs, fe, _ in ranges:
                            fs = max(0, min(int(fs), T_b_val - 1))
                            fe = max(fs + 1, min(int(fe), T_b_val))
                            seg_lp = lp_b[fs:fe]   # (seg_len, V)
                            if seg_lp.shape[0] == 0: continue
                            # blank_prob = rata-rata prob blank di segmen ini
                            blank_prob_mean = seg_lp[:, BLANK].exp().mean()
                            # Penalti jika blank dominan (>0.9) di segmen berisi kata
                            blank_penalty = blank_penalty + F.relu(blank_prob_mean - 0.9)
                            n_segs += 1
                    if n_segs > 0:
                        blank_penalty = blank_penalty / n_segs * 0.5

                so_loss = model.semi_orth_loss()
                loss    = ctc_loss + cfg["semi_orth_weight"] * so_loss + blank_penalty

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if torch.isfinite(loss):
                tr_loss_sum += loss.item(); n_batches += 1

        tr_loss = tr_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0; val_batches = 0

        with torch.no_grad():
            for feat_pad, tgt_cat, in_lens, tgt_lens, _ in val_loader:
                # [BUG-5 FIX] GPU transfer dulu
                feat_pad = feat_pad.to(DEVICE)
                tgt_cat  = tgt_cat.to(DEVICE)
                in_lens  = in_lens.to(DEVICE)
                tgt_lens = tgt_lens.to(DEVICE)
                with autocast(enabled=use_amp):
                    B_v = feat_pad.shape[0]
                    # [BUG-3 FIX] list comprehension, konsisten dengan training loop
                    log_probs_list = [
                        model(feat_pad[b, :int(in_lens[b].item()), :])
                        for b in range(B_v)
                    ]
                    T_max_v = max(lp.shape[0] for lp in log_probs_list)
                    V_v     = log_probs_list[0].shape[1]
                    stacked = torch.zeros(T_max_v, B_v, V_v, device=DEVICE)
                    for b, lp in enumerate(log_probs_list):
                        stacked[:lp.shape[0], b, :] = lp
                    vl = ctc_loss_fn(stacked, tgt_cat, in_lens, tgt_lens)
                if torch.isfinite(vl):
                    val_loss_sum += vl.item()
                    val_batches  += 1

        val_loss   = val_loss_sum / max(val_batches, 1)
        blank_pct  = _blank_pct(model, val_ids[:15], sequences)
        model.train()
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        hist.append({
            "epoch"     : epoch,
            "lr"        : round(current_lr, 9),
            "train_loss": round(tr_loss, 6),
            "val_loss"  : round(val_loss, 6),
            "blank_pct" : round(blank_pct, 1),
        })

        if verbose:
            print(f"  {epoch:4d}  {current_lr:10.2e}  {tr_loss:8.4f}  "
                  f"{val_loss:8.4f}  {blank_pct:6.1f}%  {elapsed:4.0f}s")
            if blank_pct > 95 and epoch > cfg["warmup_epoch"]:
                print(f"  ⚠ Blank collapse! Blank%={blank_pct:.1f}")

        if epoch > cfg["warmup_epoch"]:
            scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss; pat = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            pat += 1
            if pat >= early_stop:
                if verbose:
                    print(f"\n  ✓ Early stop ep {epoch} (best={best_val:.4f})")
                break

    return best_val, hist


def _blank_pct(model: TDNNAcousticModel,
               ids: list, sequences: dict, n: int = 10) -> float:
    model.eval()
    tb = tf = 0
    with torch.no_grad():
        for aid in ids[:n]:
            lp    = model.get_log_probs(sequences[aid])
            preds = np.argmax(lp, axis=1)
            tb   += (preds == BLANK).sum()
            tf   += len(preds)
    return tb / max(tf, 1) * 100

# ─────────────────────────────────────────────────────────────
#  CTC BEAM SEARCH DECODE  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def ctc_beam_decode_tokens(log_probs: np.ndarray,
                           id2phn: dict,
                           beam_width: int = 10) -> list:
    T, V = log_probs.shape
    NEG_INF = -float('inf')
    beam = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beam = {}
        for seq, (pb, pnb) in beam.items():
            lp_b = log_probs[t, BLANK]
            p0, p1 = new_beam.get(seq, (NEG_INF, NEG_INF))
            new_beam[seq] = (
                np.logaddexp(p0, np.logaddexp(pb, pnb) + lp_b), p1
            )
            for c in range(1, V):
                lp_c = log_probs[t, c]
                nseq = seq + (c,)
                if seq and seq[-1] == c:
                    ep_nb = pb + lp_c
                else:
                    ep_nb = np.logaddexp(pb, pnb) + lp_c
                p0n, p1n = new_beam.get(nseq, (NEG_INF, NEG_INF))
                new_beam[nseq] = (p0n, np.logaddexp(p1n, ep_nb))

        beam = dict(
            sorted(new_beam.items(),
                   key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                   reverse=True)[:beam_width]
        )

    best_seq = max(beam.items(),
                   key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
    return [id2phn[int(c)] for c in best_seq
            if id2phn.get(int(c), '<blank>') != '<blank>']

# ─────────────────────────────────────────────────────────────
#  EVALUASI WER + FONEM CONFUSION  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def levenshtein_words(ref: list, hyp: list) -> tuple:
    R, H = len(ref), len(hyp)
    if R == 0 and H == 0:
        return 0.0, 0, 0, 0, 0
    if R == 0:
        return 1.0, 0, 0, H, max(H, 1)

    dp = np.zeros((R+1, H+1), dtype=np.int32)
    dp[:, 0] = np.arange(R+1)
    dp[0, :] = np.arange(H+1)

    for i in range(1, R+1):
        for j in range(1, H+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)

    i, j = R, H
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            if ref[i-1] != hyp[j-1]: S += 1
            i -= 1; j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    N = max(R, 1)
    return min((S+D+I) / N, 1.0), S, D, I, N


def evaluate_wer(model, eval_ids, sequences, word_refs, phn_refs,
                 id2phn, beam_width=10):
    model.eval()
    wer_rows = []
    tS = tD = tI = tN = 0

    for aid in eval_ids:
        lp        = model.get_log_probs(sequences[aid])
        hyp_toks  = ctc_beam_decode_tokens(lp, id2phn, beam_width)
        hyp_words = tokens_to_words(hyp_toks)
        hyp_text  = phoneme_tokens_to_text(hyp_toks)
        ref_words = word_refs.get(aid, [])

        wer_val, S, D, I, N = levenshtein_words(ref_words, hyp_words)
        tS += S; tD += D; tI += I; tN += N

        wer_rows.append({
            "audio_id"  : aid,
            "WER"       : round(wer_val * 100, 4),
            "S": S, "D": D, "I": I, "N": N,
            "referensi" : " ".join(ref_words),
            "hipotesis" : hyp_text,
        })

    wer_corpus = min((tS+tD+tI) / max(tN, 1), 1.0) * 100
    wl = [r["WER"] for r in wer_rows]
    summary = {
        "wer_corpus": round(wer_corpus, 4),
        "wer_mean"  : round(float(np.mean(wl)), 4) if wl else 100.0,
        "wer_min"   : round(float(np.min(wl)), 4) if wl else 100.0,
        "wer_max"   : round(float(np.max(wl)), 4) if wl else 100.0,
        "total_S": tS, "total_D": tD, "total_I": tI, "total_N": tN,
        "n_audio": len(eval_ids),
    }
    return wer_rows, summary


def compute_phoneme_accuracy_detailed(model, eval_ids, sequences,
                                      phn_refs, id2phn, beam_width=10):
    model.eval()
    total_ref = total_edit = 0
    confusion = defaultdict(int)

    for aid in eval_ids:
        lp        = model.get_log_probs(sequences[aid])
        hyp_toks  = ctc_beam_decode_tokens(lp, id2phn, beam_width)
        ref_toks  = phn_refs.get(aid, [])
        ref_clean = [t for t in ref_toks if t not in (SILENCE_TOKEN, '<sp>', '<blank>')]
        hyp_clean = [t for t in hyp_toks if t not in (SILENCE_TOKEN, '<sp>', '<blank>')]

        R = len(ref_clean)
        if R == 0: continue
        H = len(hyp_clean)

        dp = np.zeros((R+1, H+1), dtype=np.int32)
        dp[:, 0] = np.arange(R+1)
        dp[0, :] = np.arange(H+1)
        for i in range(1, R+1):
            for j in range(1, H+1):
                cost = 0 if ref_clean[i-1] == hyp_clean[j-1] else 1
                dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)

        i, j = R, H
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                cost = 0 if ref_clean[i-1] == hyp_clean[j-1] else 1
                if dp[i, j] == dp[i-1, j-1] + cost:
                    confusion[(ref_clean[i-1], hyp_clean[j-1])] += 1
                    i -= 1; j -= 1
                elif dp[i, j] == dp[i-1, j] + 1:
                    confusion[(ref_clean[i-1], '<del>')] += 1; i -= 1
                else:
                    confusion[('<ins>', hyp_clean[j-1])] += 1; j -= 1
            elif i > 0:
                confusion[(ref_clean[i-1], '<del>')] += 1; i -= 1
            else:
                confusion[('<ins>', hyp_clean[j-1])] += 1; j -= 1

        total_ref  += R
        total_edit += int(dp[R, H])

    per = total_edit / total_ref * 100.0 if total_ref > 0 else 100.0
    acc = max(0.0, 100.0 - per)
    return round(acc, 4), round(per, 4), dict(confusion)


def confusion_to_df(confusion: dict) -> pd.DataFrame:
    rows = []
    for (ref, hyp), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
        if ref == hyp: typ = "correct"
        elif hyp == '<del>': typ = "deletion"
        elif ref == '<ins>': typ = "insertion"
        else: typ = "substitution"
        rows.append({"ref": ref, "hyp": hyp, "count": cnt, "type": typ})
    return pd.DataFrame(rows)


def print_eval_stats(summ_wer, phn_acc, phn_per, split_name="test"):
    print(f"\n── Evaluasi [{split_name.upper()}] {'─'*40}")
    print(f"  Audio          : {summ_wer.get('n_audio', 0)}")
    print(f"  WER Corpus     : {summ_wer.get('wer_corpus', 0):.2f}%")
    print(f"  WER Mean       : {summ_wer.get('wer_mean', 0):.2f}%")
    print(f"  Akurasi Fonem  : {phn_acc:.2f}%")
    print(f"  PER            : {phn_per:.2f}%")
    print(f"  S/D/I/N        : {summ_wer.get('total_S',0)}/"
          f"{summ_wer.get('total_D',0)}/{summ_wer.get('total_I',0)}/"
          f"{summ_wer.get('total_N',0)}")
    print(f"{'─'*55}")

# ─────────────────────────────────────────────────────────────
#  HYPERPARAMETER TUNING
#  [FIX-3]  lr: 5e-5..5e-4 (Paper 2: start=0.0005)
#  [FIX-4]  temperature: 1.0..1.5
#  [FIX-8]  n_layer_choices: semua keys CONTEXT_PRESETS (termasuk 11)
#  [FIX-9]  hidden_dim: tambah 1024
#  [FIX-10] context_type: per n_layers melalui _CTX_CHOICES
# ─────────────────────────────────────────────────────────────

def _composite_score(wer_val, val_loss, cfg):
    normalized_loss = math.log1p(val_loss) / math.log1p(20.0)
    return (cfg.get("wer_weight", 0.6) * wer_val / 100.0 +
            cfg.get("loss_weight", 0.4) * min(normalized_loss, 1.0))


def _make_model(params, vocab, actual_dim, cfg):
    """
    [FIX-1] Selalu gunakan TDNNLayer (eksplisit).
    Untuk upgrade ke TDNN-F nanti, ubah layer_class=TDNNFLayer di sini.
    """
    n_layers = params["n_layers"]
    ctx_type = params["context_type"]
    preset   = CONTEXT_PRESETS.get(n_layers, CONTEXT_PRESETS[5])
    contexts = preset.get(ctx_type, list(preset.values())[0])
    return TDNNAcousticModel(
        input_dim      = actual_dim,
        hidden_dim     = params["hidden_dim"],
        contexts       = contexts,
        output_dim     = len(vocab),
        dropout        = params["dropout"],
        temperature    = params.get("temperature", 1.0),
        layer_class    = TDNNLayer,   # [FIX-1] eksplisit TDNNLayer
        bottleneck_dim = params.get("bottleneck_dim", cfg.get("bottleneck_dim", 128)),
    )


def hyperparameter_tuning_bayesian(sequences, targets, word_refs, phn_refs,
                                   vocab, id2phn, actual_dim,
                                   train_ids, val_ids, cfg,
                                   alignment_loader=None):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        OPTUNA_OK = True
    except ImportError:
        OPTUNA_OK = False
        print("  ⚠ Optuna tidak tersedia — menggunakan random search.")

    n_trials   = cfg["optuna_trials"]
    results    = []
    best_score = float("inf")
    best_combo = None

    def run_trial(params, trial_idx, rng_trial):
        sample_size = min(60, len(val_ids))
        sample_ids  = list(rng_trial.choice(val_ids, sample_size, replace=False))
        print(f"\n[{trial_idx:>3}/{n_trials}] "
              f"n_layers={params['n_layers']}  hidden={params['hidden_dim']}  "
              f"lr={params['learning_rate']:.0e}  drop={params['dropout']:.2f}  "
              f"ctx={params['context_type']}  temp={params.get('temperature',1.0):.1f}")
        torch.manual_seed(cfg["random_seed"] + trial_idx)
        model = _make_model(params, vocab, actual_dim, cfg)
        best_vl, _ = run_training(
            model, train_ids, val_ids, sequences, targets,
            cfg, lr_init=params["learning_rate"],
            max_epochs=cfg["epochs_tuning"],
            early_stop=cfg["early_stop"],
            model_save_path=None,
            verbose=False,
            use_augment=True,
            alignment_loader=alignment_loader
        )
        _, wer_s = evaluate_wer(model, sample_ids, sequences,
                                word_refs, phn_refs, id2phn, cfg["beam_width"])
        phn_acc, phn_per, _ = compute_phoneme_accuracy_detailed(
            model, sample_ids, sequences, phn_refs, id2phn, cfg["beam_width"])
        wer_val = wer_s["wer_corpus"]
        score   = _composite_score(wer_val, best_vl, cfg)
        row = {
            "trial": trial_idx, **params,
            "context_detail"  : str(CONTEXT_PRESETS.get(params["n_layers"],{}).get(params["context_type"], [])),
            "best_val_loss"   : round(best_vl, 6),
            "wer_val_pct"     : round(wer_val, 2),
            "phoneme_acc_val" : round(phn_acc, 2),
            "per_val"         : round(phn_per, 2),
            "composite_score" : round(score, 6),
        }
        print(f"  → val_loss={best_vl:.4f}  WER={wer_val:.1f}%  Score={score:.4f}")
        return row, score

    print(f"\n{'='*65}")
    print(f"  HYPERPARAMETER TUNING (TDNN) — "
          f"{'BAYESIAN' if OPTUNA_OK else 'RANDOM SEARCH'}")
    print(f"  Trials: {n_trials} | Epoch/trial: {cfg['epochs_tuning']}")
    print(f"{'='*65}")

    # ── Bangun ruang pencarian yang FLAT dan FIXED ──────────────────────────────
    # ROOT CAUSE ERROR: Optuna CategoricalDistribution TIDAK boleh punya
    # "dynamic value space" — yaitu choices yang berubah antar trial.
    # Kode lama memilih ctx_opts berdasarkan nl yang baru dipilih di trial itu,
    # sehingga choices "context_type" berbeda antar trial → ValueError.
    #
    # SOLUSI: encode kombinasi (n_layers, context_type) sebagai satu string
    # tunggal yang dipilih dari daftar TETAP (flat search space).
    # _make_model() kemudian men-decode kembali string itu.
    # ────────────────────────────────────────────────────────────────────────────
    _combo_choices = [
        f"{nl}_{ct}"
        for nl, ct_list in _CTX_CHOICES.items()
        for ct in ct_list
    ]
    # Contoh isi: ["3_narrow","3_standard","3_wide","4_narrow",...,"11_karelian",...]

    if OPTUNA_OK:
        import optuna

        def objective(trial):
            # Satu suggest_categorical dengan choices TETAP — tidak pernah berubah
            combo    = trial.suggest_categorical("nl_ctx", _combo_choices)
            nl_str, ctx_type = combo.split("_", 1)
            nl       = int(nl_str)
            params   = {
                "n_layers"      : nl,
                "hidden_dim"    : trial.suggest_categorical("hidden_dim", [256, 512, 1024]),
                "learning_rate" : trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
                "dropout"       : trial.suggest_float("dropout", 0.1, 0.35),
                "context_type"  : ctx_type,
                "temperature"   : trial.suggest_float("temperature", 1.0, 1.5),
            }
            rng_trial = np.random.default_rng(cfg["random_seed"] + trial.number)
            row, score = run_trial(params, trial.number+1, rng_trial)
            results.append(row)
            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg["random_seed"]),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_t   = study.best_trial
        # Decode kembali dari format "nl_ctx" → n_layers + context_type
        best_combo_str = best_t.params["nl_ctx"]
        nl_str, ctx_type = best_combo_str.split("_", 1)
        n_layers = int(nl_str)
        preset   = CONTEXT_PRESETS.get(n_layers, CONTEXT_PRESETS[5])
        contexts = preset.get(ctx_type, list(preset.values())[0])
        best_combo = {
            "n_layers"     : n_layers,
            "hidden_dim"   : best_t.params["hidden_dim"],
            "learning_rate": best_t.params["learning_rate"],
            "dropout"      : best_t.params["dropout"],
            "context_type" : ctx_type,
            "temperature"  : best_t.params["temperature"],
            "contexts"     : contexts,
        }
    else:
        for trial_idx in range(1, n_trials + 1):
            rng_trial = np.random.default_rng(cfg["random_seed"] + trial_idx)
            # Pilih satu kombinasi dari daftar flat yang sama dengan Optuna branch
            combo     = str(rng_trial.choice(_combo_choices))
            nl_str, ct = combo.split("_", 1)
            nl        = int(nl_str)
            params    = {
                "n_layers"      : nl,
                "hidden_dim"    : int(rng_trial.choice([256, 512, 1024])),
                "learning_rate" : float(10 ** rng_trial.uniform(-4.3, -3.3)),
                "dropout"       : float(rng_trial.uniform(0.1, 0.35)),
                "context_type"  : ct,
                "temperature"   : float(rng_trial.uniform(1.0, 1.5)),
            }
            row, score = run_trial(params, trial_idx, rng_trial)
            results.append(row)
            if score < best_score:
                best_score = score
                preset     = CONTEXT_PRESETS.get(nl, CONTEXT_PRESETS[5])
                contexts   = preset.get(ct, list(preset.values())[0])
                best_combo = {**params, "contexts": contexts}

    df_tune = pd.DataFrame(results).sort_values("composite_score").reset_index(drop=True)
    df_tune.to_csv(OUT_TUNING_CSV, index=False)
    print(f"\n  Hasil Tuning tersimpan ({len(results)} trials) di {OUT_TUNING_CSV}")
    return best_combo, df_tune

# ─────────────────────────────────────────────────────────────
#  K-FOLD CROSS VALIDATION  (tidak berubah)
# ─────────────────────────────────────────────────────────────

def run_kfold_cv(sequences, targets, word_refs, phn_refs,
                 vocab, id2phn, actual_dim, all_ids, best_params, cfg,
                 alignment_loader=None):
    k = cfg["n_cv_folds"]
    folds = kfold_splits(all_ids, k, cfg["random_seed"])
    cv_rows = []

    print(f"\n{'='*65}")
    print(f"  K-FOLD CROSS VALIDATION ({k} folds)")
    print(f"{'='*65}")

    for fold_idx, (train_ids, val_ids) in enumerate(folds, 1):
        print(f"\n  ── Fold {fold_idx}/{k} ─────────────────────")
        print(f"     train: {len(train_ids)}  val: {len(val_ids)}")

        mean_f, std_f = compute_global_norm(sequences, train_ids)
        seq_fold = apply_global_norm(sequences, mean_f, std_f)

        torch.manual_seed(cfg["random_seed"] + fold_idx)
        model = _make_model(best_params, vocab, actual_dim, cfg)

        best_vl, _ = run_training(
            model, train_ids, val_ids, seq_fold, targets,
            cfg,
            lr_init=best_params["learning_rate"],
            max_epochs=cfg["epochs_full"] // 2,
            early_stop=cfg["early_stop_full"],
            verbose=True,
            use_augment=True,
            alignment_loader=alignment_loader
        )

        if alignment_loader is not None and cfg.get("use_alignment"):
            wr_rows, wer_s, _ = evaluate_wer_with_alignment(
                model, val_ids, seq_fold, word_refs, phn_refs,
                id2phn, alignment_loader,
                ctc_beam_decode_tokens=ctc_beam_decode_tokens,
                tokens_to_words=tokens_to_words,
                phoneme_tokens_to_text=phoneme_tokens_to_text,
                levenshtein_words=levenshtein_words,
                beam_width=cfg["beam_width"],
            )
        else:
            _, wer_s = evaluate_wer(model, val_ids, seq_fold,
                                    word_refs, phn_refs, id2phn, cfg["beam_width"])

        phn_acc, phn_per, _ = compute_phoneme_accuracy_detailed(
            model, val_ids, seq_fold, phn_refs, id2phn, cfg["beam_width"])

        cv_rows.append({
            "fold"           : fold_idx,
            "n_train"        : len(train_ids),
            "n_val"          : len(val_ids),
            "val_loss"       : round(best_vl, 6),
            "wer_corpus_pct" : round(wer_s["wer_corpus"], 2),
            "wer_mean_pct"   : round(wer_s["wer_mean"], 2),
            "phoneme_acc_pct": round(phn_acc, 2),
            "per_pct"        : round(phn_per, 2),
        })
        print(f"  Fold {fold_idx}: WER={wer_s['wer_corpus']:.2f}%  PhAcc={phn_acc:.2f}%")

    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(OUT_CV_CSV, index=False)
    print(f"\n  CV Summary ({k} folds):")
    print(f"  WER   : {df_cv['wer_corpus_pct'].mean():.2f}% ± {df_cv['wer_corpus_pct'].std():.2f}%")
    print(f"  PhAcc : {df_cv['phoneme_acc_pct'].mean():.2f}% ± {df_cv['phoneme_acc_pct'].std():.2f}%")
    return df_cv

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(FIXED_CONFIG["random_seed"])
    np.random.seed(FIXED_CONFIG["random_seed"])

    print("="*65)
    print("  TDNN (non-F) + CTC — Javanese ASR")
    print(f"  Device: {DEVICE}  |  AMP: {FIXED_CONFIG['use_amp']}")
    print("="*65)

    alignment_loader = None
    if FIXED_CONFIG.get("use_alignment", False) and os.path.exists(ALIGNMENT_CSV):
        alignment_loader = AlignmentLoader(ALIGNMENT_CSV)
        print(f"[INFO] Alignment aktif — {ALIGNMENT_CSV}")
    else:
        print("[INFO] Alignment tidak digunakan atau file tidak ditemukan")

    print("\n[1] Memuat Data & Ekstraksi MFCC")
    if alignment_loader is not None:
        (sequences_raw, targets, word_refs, phn_refs,
         phn2id, id2phn, vocab, actual_dim,
         audio_ids_with_aln) = load_data_with_alignment(
            csv_path             = CSV_PATH,
            alignment_loader     = alignment_loader,
            extract_features_raw = extract_features_raw,
            normalize_text       = normalize_text,
            build_phoneme_vocab  = build_phoneme_vocab,
            encode_phoneme       = encode_phoneme,
            find_wav             = find_wav,
            fixed_config         = FIXED_CONFIG,
        )
    else:
        (sequences_raw, targets, word_refs, phn_refs,
         phn2id, id2phn, vocab, actual_dim) = load_data(CSV_PATH)
        audio_ids_with_aln = set()

    all_ids = list(sequences_raw.keys())
    train_ids, val_ids, test_ids = split_dataset(all_ids, FIXED_CONFIG)
    print(f"  Split → train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

    mean_g, std_g = compute_global_norm(sequences_raw, train_ids)
    torch.save({"mean": torch.from_numpy(mean_g), "std": torch.from_numpy(std_g)}, NORM_PATH)
    sequences = apply_global_norm(sequences_raw, mean_g, std_g)

    # [BUG-2 FIX] Verifikasi normalisasi berhasil — mean_abs_max setelah
    # normalisasi HARUS jauh di bawah FEAT_CLIP (10.0), bukan 247 seperti sebelumnya.
    # Jika tidak, ada bug di pipeline normalisasi dan training tidak akan konvergen.
    _sample_norm = sequences[train_ids[0]]
    _norm_max = float(np.abs(_sample_norm).max())
    _norm_mean = float(np.abs(_sample_norm).mean())
    print(f"  [Sanity] Setelah normalisasi — max={_norm_max:.3f}  mean_abs={_norm_mean:.3f}")
    if _norm_max > 12.0:
        print(f"  ⚠ PERINGATAN: fitur masih terlalu besar setelah normalisasi ({_norm_max:.1f})!")
        print(f"    Periksa apply_global_norm() dan pastikan sequences_raw menggunakan")
        print(f"    mean/std dari train set, bukan sequences_raw yang belum dinormalisasi.")
    else:
        print(f"  ✓ Normalisasi OK (max={_norm_max:.2f} ≤ {FEAT_CLIP})")

    print("\n[2] Hyperparameter Tuning")
    best_params, df_tune = hyperparameter_tuning_bayesian(
        sequences, targets, word_refs, phn_refs,
        vocab, id2phn, actual_dim, train_ids, val_ids, FIXED_CONFIG,
        alignment_loader=alignment_loader
    )

    print("\n[3] Training Penuh")
    for k in ["n_layers","hidden_dim","learning_rate","dropout","context_type","temperature"]:
        if k in best_params:
            print(f"  {k:<16}: {best_params[k]}")

    torch.manual_seed(FIXED_CONFIG["random_seed"])
    best_model = _make_model(best_params, vocab, actual_dim, FIXED_CONFIG)
    print(f"  Parameter total: {sum(p.numel() for p in best_model.parameters()):,}")

    _, history = run_training(
        best_model, train_ids, val_ids, sequences, targets,
        FIXED_CONFIG,
        lr_init=best_params["learning_rate"],
        max_epochs=FIXED_CONFIG["epochs_full"],
        early_stop=FIXED_CONFIG["early_stop_full"],
        model_save_path=MODEL_BEST_PATH,
        verbose=True,
        use_augment=True,
        alignment_loader=alignment_loader
    )
    pd.DataFrame(history).to_csv(OUT_HISTORY_CSV, index=False)

    if os.path.exists(MODEL_BEST_PATH):
        best_model.load_state_dict(
            torch.load(MODEL_BEST_PATH, map_location=DEVICE)
        )
    best_model = best_model.to(DEVICE)

    print("\n[4] Evaluasi")
    all_wer_rows = []

    for split_name, split_ids in [("val", val_ids), ("test", test_ids)]:
        if alignment_loader is not None and FIXED_CONFIG.get("use_alignment"):
            wr_rows, summ, df_seg = evaluate_wer_with_alignment(
                best_model, split_ids, sequences, word_refs, phn_refs,
                id2phn, alignment_loader,
                ctc_beam_decode_tokens = ctc_beam_decode_tokens,
                tokens_to_words        = tokens_to_words,
                phoneme_tokens_to_text = phoneme_tokens_to_text,
                levenshtein_words      = levenshtein_words,
                beam_width             = FIXED_CONFIG["beam_width"],
            )
            if not df_seg.empty:
                df_seg.to_csv(
                    os.path.join(DIR_SAVE, f"HasilWER_TDNN_per_segmen_{split_name}.csv"),
                    index=False,
                )
        else:
            wr_rows, summ = evaluate_wer(
                best_model, split_ids, sequences, word_refs, phn_refs,
                id2phn, FIXED_CONFIG["beam_width"]
            )

        phn_acc, phn_per, confusion = compute_phoneme_accuracy_detailed(
            best_model, split_ids, sequences, phn_refs, id2phn, FIXED_CONFIG["beam_width"]
        )
        print_eval_stats(summ, phn_acc, phn_per, split_name)
        all_wer_rows.extend(wr_rows)

    df_w = pd.DataFrame(all_wer_rows)
    df_w[["audio_id","WER","S","D","I","N"]].to_csv(OUT_WER_CSV, index=False)
    df_w[["audio_id","referensi","hipotesis","WER"]].to_csv(OUT_DEC_CSV, index=False)

    if confusion:
        confusion_to_df(confusion).to_csv(OUT_CONFUSION_CSV, index=False)

    print("\n[5] K-Fold Cross Validation")
    df_cv = run_kfold_cv(
        sequences_raw, targets, word_refs, phn_refs,
        vocab, id2phn, actual_dim, all_ids, best_params, FIXED_CONFIG,
        alignment_loader=alignment_loader
    )

    print(f"\nSemua proses selesai. Output tersimpan di: {DIR_SAVE}")
    return best_model, history, df_tune, df_cv


if __name__ == "__main__":
    model, history, tuning, cv = main()