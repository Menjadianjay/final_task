import math, os, re, json, time, pickle, logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as TAT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

try:
    import tgt
    _TGT_OK = True
except ImportError:
    _TGT_OK = False
    logging.getLogger(__name__).warning(
        "Paket 'tgt' tidak ditemukan. Install dengan: pip install tgt\n"
        "TextGrid parsing tidak akan berfungsi."
    )

logging.basicConfig(
    format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# KONSTANTA GLOBAL
# ─────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM   = 27
SR, N_MFCC, HOP, WIN, N_MELS = 16_000, 13, 160, 400, 26
FRAME_SHIFT = HOP / SR


# ─────────────────────────────────────────────────────────────────
# MAPPING FONEM & KARAKTER
# ─────────────────────────────────────────────────────────────────

_PH = {
    "ng":"ng","ny":"ny","th":"th","dh":"dh",
    "a":"a","b":"b","c":"ch","d":"d","e":"e","f":"f","g":"g","h":"h",
    "i":"i","j":"j","k":"k","l":"l","m":"m","n":"n","o":"o","p":"p",
    "r":"r","s":"s","t":"t","u":"u","v":"v","w":"w","y":"y","z":"z",
    "x":"k","q":"k",
}
_MJ = {
    "Ã©":"é","Ã¨":"è","Ã ":"à","Ã¢":"â","Ã®":"î","Ã´":"ô",
    "Ã»":"û","Ã«":"ë","Ã¯":"ï","Ã¼":"ü","Ã±":"ñ","Ã§":"ç",
}
_AC  = str.maketrans("éèêëàáâäîïíìôóòöûúùüñç", "eeeeaaaaiiiioooouuuunc")
_SIL = {"silence", "sil", "sp", "spn", "<eps>", ""}


# ─────────────────────────────────────────────────────────────────
# NORMALISASI & G2P
# ─────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    for b, g in _MJ.items():
        text = text.replace(b, g)
    text = text.lower().strip()
    text = re.sub(r'["\'`\u201c\u201d\u2018\u2019()\[\]{}]', " ", text)
    text = re.sub(r"[+\-*/=<>%^&|~\\@#$.,;:!?]", " ", text)
    text = text.translate(_AC)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

clean = normalise


def word_to_phones(word: str) -> List[str]:
    word = normalise(word)
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in _PH:
            out.append(_PH[word[i:i+2]]); i += 2
        elif word[i] in _PH:
            out.append(_PH[word[i]]); i += 1
        else:
            i += 1
    return out or list(word)

word_to_phonemes = word_to_phones  # alias


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    num_pdfs           : int   = 29
    epochs             : int   = 30
    patience           : int   = 7
    batch_size         : int   = 64
    lr_start           : float = 5e-4
    lr_end             : float = 1e-6
    weight_decay       : float = 5e-4
    dropout            : float = 0.2
    scheduler          : str   = "cosine"
    clip_grad          : float = 5.0
    use_specaug        : bool  = True
    tdnn_layers        : int   = 8
    tdnn_hidden        : int   = 512
    tdnn_contexts      : List  = field(default_factory=lambda: [
        [-2,-1,0,1,2], [-1,0,1], [-3,0,3], [-7,0,2],
        [-3,0,3], [0], [-6,0,6], [-14,0,14],
    ])
    tdnnf_hidden       : int   = 512
    tdnnf_bottleneck   : int   = 128
    tdnnf_prefinal_dim : int   = 192
    tdnnf_bypass_scale : float = 0.66
    ortho_period       : int   = 4
    prefinal_dim       : int   = 192


# ─────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

_dev   = torch.device(DEVICE)
_mfcc  = TAT.MFCC(SR, N_MFCC, melkwargs={
    "n_fft": 512, "n_mels": N_MELS, "win_length": WIN,
    "hop_length": HOP, "f_min": 0, "f_max": 4000,
}).to(_dev)
_delta = TAT.ComputeDeltas().to(_dev)


def extract_mfcc(wav_path: str) -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0, keepdim=True).to(_dev)
    pre = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], dim=1)
    m   = _mfcc(pre)
    d1  = _delta(m)
    d2  = _delta(d1)
    return torch.cat([m, d1, d2], dim=1).squeeze(0).T.cpu().numpy().astype(np.float32)


def spec_augment(
    x           : torch.Tensor,
    n_freq_mask : int = 2,
    f           : int = 8,
    n_time_mask : int = 2,
    t           : int = 40,
) -> torch.Tensor:
    T, D = x.shape
    out  = x.clone()
    for _ in range(n_freq_mask):
        if D > f:
            f0 = torch.randint(0, D - f, (1,), device=x.device).item()
            out[:, f0 : f0 + f] = 0.0
    for _ in range(n_time_mask):
        if T > t:
            t0 = torch.randint(0, T - t, (1,), device=x.device).item()
            out[t0 : t0 + t, :] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────
# TEXTGRID & LABEL MAP
# ─────────────────────────────────────────────────────────────────

def _phone_from_mfa(raw: str) -> str:
    ph = raw.strip().lower()
    ph = re.sub(r"\d+$", "", ph)
    return ph if ph not in _SIL else ""


def parse_textgrid(
    tg_path    : str,
    tier_name  : str   = "phones",
    frame_shift: float = FRAME_SHIFT,
) -> List[str]:
    if not _TGT_OK:
        raise ImportError("Install 'tgt' terlebih dahulu: pip install tgt")
    tg   = tgt.io.read_textgrid(tg_path)
    tier = None
    for name in tg.get_tier_names():
        if tier_name.lower() in name.lower():
            tier = tg.get_tier_by_name(name)
            break
    if tier is None:
        raise ValueError(
            f"Tier '{tier_name}' tidak ditemukan di {tg_path}. "
            f"Tier tersedia: {tg.get_tier_names()}"
        )
    total_dur = tier.end_time
    n_frames  = max(1, int(round(total_dur / frame_shift)))
    labels    = [""] * n_frames
    for interval in tier:
        ph = _phone_from_mfa(interval.text)
        if not ph:
            continue
        f_start = int(round(interval.start_time / frame_shift))
        f_end   = int(round(interval.end_time   / frame_shift))
        for f in range(f_start, min(f_end, n_frames)):
            labels[f] = ph
    return labels


def build_label_map_from_textgrids(
    tg_dir   : str,
    tier_name: str = "phones",
) -> Dict[str, int]:
    if not _TGT_OK:
        raise ImportError("Install 'tgt' terlebih dahulu: pip install tgt")
    phones: set = set()
    for tg_path in Path(tg_dir).rglob("*.TextGrid"):
        tg = tgt.io.read_textgrid(str(tg_path))
        for name in tg.get_tier_names():
            if tier_name.lower() not in name.lower():
                continue
            tier = tg.get_tier_by_name(name)
            for interval in tier:
                ph = _phone_from_mfa(interval.text)
                if ph:
                    phones.add(ph)
    phonetic_order = [
        "sil","sp","a","e","i","o","u",
        "b","c","ch","d","dh","f","g","h","j",
        "k","ks","l","m","n","ng","ny","p",
        "r","s","t","th","v","w","y","z",
    ]
    phone_list = sorted(
        phones,
        key=lambda p: (phonetic_order.index(p) if p in phonetic_order else 999, p),
    )
    label2id = {ph: i for i, ph in enumerate(phone_list)}
    log.info(f"Label map dari TextGrid: {len(label2id)} fonem unik")
    log.info(f"Urutan fonem: {list(label2id.keys())}")
    return label2id


def save_label_map(label2id: Dict[str, int], csv_path: str) -> None:
    df = pd.DataFrame([
        {"label_tdnn": ph, "label_id": idx}
        for ph, idx in sorted(label2id.items(), key=lambda x: x[1])
    ])
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Label map disimpan: {csv_path}  ({len(df)} fonem)")


# ─────────────────────────────────────────────────────────────────
# ALIGNMENT BRIDGE
# ─────────────────────────────────────────────────────────────────

class AlignmentBridge:
    def __init__(
        self,
        label_map_csv: str,
        tg_dir       : str,
        lda_pkl      : str   = "",
        tier_name    : str   = "phones",
        frame_shift  : float = FRAME_SHIFT,
    ):
        ldf           = pd.read_csv(label_map_csv)
        self.label2id = dict(zip(ldf["label_tdnn"], ldf["label_id"].astype(int)))
        self.num_pdfs = int(ldf["label_id"].max()) + 1
        log.info(f"AlignmentBridge (MFA): {self.num_pdfs} kelas fonem")

        self.tg_dir      = Path(tg_dir)
        self.tier_name   = tier_name
        self.frame_shift = frame_shift
        self._tg_index: Dict[str, Path] = {}
        for tg_path in self.tg_dir.rglob("*.TextGrid"):
            self._tg_index[tg_path.stem] = tg_path
        log.info(f"TextGrid ditemukan: {len(self._tg_index):,} file di '{tg_dir}'")

        self.lda     = None
        self.lda_dim = INPUT_DIM
        if lda_pkl and os.path.exists(lda_pkl):
            with open(lda_pkl, "rb") as f:
                self.lda = pickle.load(f)
            self.lda_dim = self.lda.n_components
            log.info(f"LDA: {self.lda_dim} dims")

    @property
    def _index(self) -> Dict[str, Any]:
        return self._tg_index

    def transform(self, feat: np.ndarray) -> np.ndarray:
        if self.lda:
            feat = self.lda.transform(feat).astype(np.float32)
        return feat.astype(np.float32)

    def get(self, audio_id: str, wav_path: str) -> Tuple[np.ndarray, np.ndarray]:
        feat   = self.transform(extract_mfcc(wav_path))
        T      = feat.shape[0]
        labels = np.full(T, -1, dtype=np.int64)
        tg_path = self._tg_index.get(audio_id)
        if tg_path is None:
            log.debug(f"TextGrid tidak ditemukan untuk audio_id='{audio_id}'")
            return feat, labels
        try:
            ph_per_frame = parse_textgrid(str(tg_path), self.tier_name, self.frame_shift)
        except Exception as e:
            log.warning(f"Gagal parse TextGrid '{tg_path}': {e}")
            return feat, labels
        for f, ph in enumerate(ph_per_frame[:T]):
            lid = self.label2id.get(ph, -1)
            if lid >= 0:
                labels[f] = lid
        return feat, labels


def load_bridge(
    label_map_csv : str,
    tg_dir        : str,
    lda_pkl       : str   = "",
    norm_pkl      : str   = "",
    tier_name     : str   = "phones",
    frame_shift   : float = FRAME_SHIFT,
    alignment_csv : str   = "",
    alignment_test: str   = "",
) -> Tuple[Optional[AlignmentBridge], Dict[int, str], int]:
    if not os.path.exists(label_map_csv):
        log.error(f"File tidak ditemukan: {label_map_csv}")
        return None, {}, 0
    if not os.path.isdir(tg_dir):
        log.error(f"Folder TextGrid tidak ditemukan: {tg_dir}")
        return None, {}, 0
    ldf      = pd.read_csv(label_map_csv)
    id2label = {int(r["label_id"]): str(r["label_tdnn"]) for _, r in ldf.iterrows()}
    bridge   = AlignmentBridge(
        label_map_csv=label_map_csv, tg_dir=tg_dir,
        lda_pkl=lda_pkl, tier_name=tier_name, frame_shift=frame_shift,
    )
    return bridge, id2label, bridge.num_pdfs


def prepare_mfa_label_map(
    tg_dir       : str,
    label_map_out: str,
    tier_name    : str = "phones",
) -> Dict[str, int]:
    label2id = build_label_map_from_textgrids(tg_dir, tier_name)
    save_label_map(label2id, label_map_out)
    return label2id


# ─────────────────────────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────

class JavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: str,
        bridge  : AlignmentBridge,
        augment : bool = False,
    ):
        self.bridge  = bridge
        self.augment = augment
        with open(manifest, encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]
        self.items = [it for it in items if it["audio_id"] in bridge._tg_index]
        log.info(
            f"Dataset {Path(manifest).name}: {len(self.items):,} utts "
            f"(dari {len(items):,} total, "
            f"{len(items)-len(self.items):,} tanpa TextGrid)"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it           = self.items[idx]
        feat, labels = self.bridge.get(
            it["audio_id"], it.get("wav_path", it.get("wav", ""))
        )
        t = torch.from_numpy(feat)
        if self.augment:
            t = spec_augment(t)
        return {
            "feats"   : t,
            "ali"     : torch.from_numpy(labels),
            "lengths" : feat.shape[0],
            "audio_id": it["audio_id"],
            "text"    : it.get("text", ""),
        }


def _collate(batch: List[Dict]) -> Dict:
    B     = len(batch)
    max_T = max(b["feats"].shape[0] for b in batch)
    D     = batch[0]["feats"].shape[1]
    feats = torch.zeros(B, max_T, D, dtype=torch.float32)
    ali   = torch.full((B, max_T), -1, dtype=torch.long)
    lens  = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        T = b["feats"].shape[0]
        feats[i, :T] = b["feats"]
        ali[i, :T]   = b["ali"]
        lens[i]      = T
    return {
        "feats"    : feats,
        "ali"      : ali,
        "lengths"  : lens,
        "audio_ids": [b["audio_id"] for b in batch],
        "texts"    : [b["text"]     for b in batch],
    }


def make_loaders(
    cfg           : Config,
    train_manifest: str,
    val_manifest  : str,
    bridge        : AlignmentBridge,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    nw = min(4, os.cpu_count() or 1)
    kw = dict(
        collate_fn=_collate, num_workers=nw,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(nw > 0),
    )
    return (
        torch.utils.data.DataLoader(
            JavDataset(train_manifest, bridge, augment=True),
            cfg.batch_size, shuffle=True, drop_last=True, **kw,
        ),
        torch.utils.data.DataLoader(
            JavDataset(val_manifest, bridge), cfg.batch_size, **kw,
        ),
    )


def build_manifests(
    split_csv : str,
    meta_csv  : str,
    out_dir   : str,
    audio_col : str = "Path_Data",
    text_col  : str = "sentences",
) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    split_df = pd.read_csv(split_csv)
    meta_df  = pd.read_csv(meta_csv)
    meta_df["audio_id"] = meta_df[audio_col].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    meta   = {r["audio_id"]: r for _, r in meta_df.iterrows()}
    counts = {}
    for name in ["train", "val"]:
        col  = "test" if (name == "val" and "val" not in split_df["split"].values) else name
        rows = [
            {
                "audio_id": aid,
                "wav_path": str(meta[aid][audio_col]),
                "text"    : normalise(str(meta[aid][text_col])),
            }
            for aid in split_df[split_df["split"] == col]["audio_id"]
            if aid in meta
        ]
        out = os.path.join(out_dir, f"manifest_{name}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
        log.info(f"manifest_{name}.jsonl: {len(rows):,} utts")
    return counts



# ─────────────────────────────────────────────────────────────────
# PETA KONFUSI FONEM (dari alignment data TDNN)
# ref_fonem -> [fonem_TDNN_yang_sering_muncul], urutan: paling sering dulu
# ─────────────────────────────────────────────────────────────────

CONFUSE_MAP: Dict[str, List[str]] = {
    "t" : ["t", "th", "d",  "j"],
    "d" : ["d", "dh"],
    "k" : ["k", "g",  "ch"],
    "j" : ["j", "ch", "dh", "t"],
    "p" : ["p", "b"],
    "b" : ["b", "p"],
    "n" : ["n", "ng", "ny", "m"],
    "ng": ["ng","n"],
    "ny": ["ny","n"],
    "r" : ["r", "l"],
    "l" : ["l", "r"],
    "s" : ["s", "ks", "z"],
    "h" : ["h", "ng"],
    "i" : ["i", "e",  "y"],
    "e" : ["e", "i",  "a", "o", "y"],
    "a" : ["a", "o",  "e", "u"],
    "o" : ["o", "u",  "a"],
    "u" : ["u", "o",  "w"],
}

_NOISE: frozenset = frozenset({"z", "v", "nan", "f", "c"})


def collapse_phones(phones: List[str]) -> List[str]:
    """Hapus fonem berulang berurutan (j j -> j)."""
    out: List[str] = []
    for p in phones:
        if not out or p != out[-1]:
            out.append(p)
    return out


def clean_phones(phones: List[str]) -> List[str]:
    """Buang noise lalu collapse duplikat."""
    return collapse_phones([p for p in phones if p not in _NOISE])


# ─────────────────────────────────────────────────────────────────
# FUZZY TRIE
# ─────────────────────────────────────────────────────────────────

def _trie_insert(trie: Dict, phones: Tuple[str, ...], word: str) -> None:
    node = trie
    for ph in phones:
        node = node.setdefault(ph, {})
    node["__w__"] = word


def build_fuzzy_trie(
    sentences      : List[str],
    confuse_map    : Dict[str, List[str]] = None,
    min_phones     : int = 2,
    long_threshold : int = 7,
    max_pos_expand : int = 5,
    max_alt_per_pos: int = 3,
) -> Dict:
    """
    Bangun trie fonem dengan variasi konfusabel untuk kata panjang.

    Memungkinkan kata panjang (sulistyowati, universitas) ditemukan
    meski TDNN menghasilkan fonem yang sedikit berbeda (j vs t, e vs i).
    """
    if confuse_map is None:
        confuse_map = CONFUSE_MAP
    lexicon: Dict[str, Tuple[str, ...]] = {}
    for sent in sentences:
        for word in normalise(sent).split():
            if word not in lexicon:
                ph = tuple(word_to_phones(word))
                if len(ph) >= min_phones:
                    lexicon[word] = ph
    trie: Dict = {}
    n_base = n_var = 0
    for word, phones in lexicon.items():
        _trie_insert(trie, phones, word)
        n_base += 1
        if len(phones) < long_threshold:
            continue
        confusable_positions = [
            (i, ph) for i, ph in enumerate(phones)
            if ph in confuse_map and len(confuse_map[ph]) > 1
        ]
        for pos, ref_ph in confusable_positions[:max_pos_expand]:
            for alt in confuse_map[ref_ph][1 : 1 + max_alt_per_pos]:
                variant = phones[:pos] + (alt,) + phones[pos+1:]
                _trie_insert(trie, variant, word)
                n_var += 1
    log.info(
        f"build_fuzzy_trie: {n_base} kata, {n_var} variasi, "
        f"total {n_base+n_var} entri trie"
    )
    return trie


def _trie_matches_from(
    trie   : Dict,
    phones : List[str],
    start  : int,
    max_len: int,
) -> List[Tuple[int, str]]:
    """Cari semua kata di trie yang cocok mulai posisi start."""
    results: List[Tuple[int, str]] = []
    node, n = trie, len(phones)
    for k in range(start, min(start + max_len, n)):
        ch = phones[k]
        if ch not in node:
            break
        node = node[ch]
        if "__w__" in node:
            results.append((k + 1, node["__w__"]))
    return results


# ─────────────────────────────────────────────────────────────────
# N-GRAM LANGUAGE MODEL
# ─────────────────────────────────────────────────────────────────

class NgramLM:
    """Kneser-Ney smoothed n-gram LM."""

    def __init__(self, order: int = 3, discount: float = 0.75):
        self.order    = order
        self.discount = discount
        self._counts  : List[Dict] = []
        self._vocab   : set        = set()
        self._N       : int        = 0
        self._trained : bool       = False
        self._continuation   : Dict[str, int] = {}
        self._n_bigram_types : int            = 0

    def build(self, sentences: List[str]) -> "NgramLM":
        self._counts = [defaultdict(int) for _ in range(self.order)]
        bigram_set: set = set()
        for sent in sentences:
            toks = ["<s>"] * (self.order - 1) + normalise(sent).split() + ["</s>"]
            self._vocab.update(toks)
            self._N += len(toks)
            for n in range(1, self.order + 1):
                for i in range(len(toks) - n + 1):
                    ng = tuple(toks[i : i + n])
                    self._counts[n - 1][ng] += 1
                    if n == 2:
                        w = toks[i + 1]
                        if (toks[i], w) not in bigram_set:
                            bigram_set.add((toks[i], w))
                            self._continuation[w] = self._continuation.get(w, 0) + 1
        self._vocab.add("<unk>")
        self._n_bigram_types = len(bigram_set)
        self._n_following: Dict = {}
        for oi in range(2, self.order + 1):
            nf: Dict = defaultdict(int)
            for ng in self._counts[oi - 1]:
                nf[ng[:-1]] += 1
            self._n_following[oi] = nf
        self._trained = True
        log.info(f"NgramLM({self.order}): vocab={len(self._vocab):,}, N={self._N:,}")
        return self

    def _pkn_unigram(self, w: str) -> float:
        if self._n_bigram_types == 0:
            return 1 / max(len(self._vocab), 1)
        return self._continuation.get(w, 0) / self._n_bigram_types

    def score(self, words: List[str]) -> float:
        if not self._trained or not words:
            return 0.0
        tok = (
            ["<s>"] * (self.order - 1)
            + [w if w in self._vocab else "<unk>" for w in words]
            + ["</s>"]
        )
        lp = 0.0
        for i in range(self.order - 1, len(tok)):
            prob = self._pkn_unigram(tok[i])
            for n in range(2, self.order + 1):
                ng    = tuple(tok[max(0, i - n + 1) : i + 1])
                ctx   = ng[:-1]
                c_ng  = self._counts[n - 1].get(ng, 0)
                c_ctx = self._counts[n - 2].get(ctx, 0)
                if c_ctx > 0:
                    nf   = self._n_following[n].get(ctx, 0)
                    lam  = self.discount * nf / c_ctx
                    prob = max(c_ng - self.discount, 0) / c_ctx + lam * prob
            lp += math.log10(max(prob, 1e-12))
        return lp

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        log.info(f"NgramLM disimpan: {path}")

    @staticmethod
    def load(path: str) -> "NgramLM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"NgramLM dimuat: {path}")
        return obj


# ─────────────────────────────────────────────────────────────────
# IMPROVED DECODER  (menggantikan decode_phones_to_sentence)
# ─────────────────────────────────────────────────────────────────

def decode_phones_improved(
    phones_raw  : List[str],
    lm          : "NgramLM",
    fuzzy_trie  : Dict,
    beam        : int   = 20,
    max_ph_word : int   = 16,
    skip_penalty: float = 2.0,
    len_bonus   : float = 1.0,
    long_bonus  : float = 1.5,
    lm_weight   : float = 0.3,
    min_word_ph : int   = 2,
) -> Tuple[List[str], float]:
    phones = clean_phones(phones_raw)
    if not phones:
        return [], 0.0

    lexicon_vocab: set = set()
    def _collect(node: Dict) -> None:
        if "__w__" in node: lexicon_vocab.add(node["__w__"])
        for k, v in node.items():
            if k != "__w__": _collect(v)
    _collect(fuzzy_trie)

    n  = len(phones)
    dp: Dict[int, List[Tuple[float, List[str]]]] = {0: [(0.0, [])]}

    for i in range(n):
        if i not in dp:
            continue
        dp[i].sort(key=lambda x: -x[0])
        dp[i] = dp[i][:beam]
        states = dp[i]

        for end_pos, word in _trie_matches_from(fuzzy_trie, phones, i, max_ph_word):
            ph_len = end_pos - i
            if ph_len < min_word_ph:
                continue
            if lexicon_vocab and word not in lexicon_vocab:
                continue
            w_score = (
                ph_len * len_bonus
                + (long_bonus if len(word) >= 6  else 0.0)
                + (long_bonus if len(word) >= 10 else 0.0)
            )
            for score, words in states:
                lm_bonus = 0.0
                if lm and words:
                    lm_bonus = lm_weight * (
                        lm.score([words[-1], word]) - lm.score([words[-1]])
                    )
                if end_pos not in dp:
                    dp[end_pos] = []
                dp[end_pos].append((score + w_score + lm_bonus, words + [word]))

        nxt = i + 1
        if nxt not in dp:
            dp[nxt] = []
        for score, words in states:
            dp[nxt].append((score - skip_penalty, words))

    best_words: List[str] = []
    best_score: float     = -1e18
    for end in range(n, -1, -1):
        if end in dp and dp[end]:
            dp[end].sort(key=lambda x: -x[0])
            s, w = dp[end][0]
            if w:
                best_score, best_words = s, w
                break

    if lm and best_words:
        best_score = lm.score(best_words)

    return best_words, best_score

def _phones_from_labels(labels: np.ndarray, id2label: Dict) -> List[str]:
    """Label TDNN per-frame → list fonem (collapse duplikat, skip silence)."""
    out, prev = [], None
    for lid in labels:
        if lid < 0:
            continue
        lbl = id2label.get(int(lid), "")
        # Tangani format "fonem_suffix" dari TDNN-F
        ph  = lbl.rsplit("_", 1)[0] if "_" in lbl else lbl
        ph  = ph.strip().lower()
        if not ph or ph in _SIL:
            prev = None
            continue
        if ph == prev:
            continue
        out.append(ph)
        prev = ph
    return out


def _fonem_to_kata(
    phones : List[str],
    lexicon,
    p2w    : Dict,
) -> List[str]:
    """
    Fonem → kata via lexicon trie atau p2w dict.
    Dipertahankan sebagai fallback — untuk decode utama
    gunakan decode_phones_improved (improved_lm_decoder.py).
    """
    if lexicon and getattr(lexicon, "_trie", None):
        words = lexicon.decode(phones)
        if words:
            return words
    if p2w:
        words, i = [], 0
        while i < len(phones):
            matched = False
            for span in range(min(len(phones) - i, 12), 0, -1):
                key = tuple(phones[i : i + span])
                if key in p2w:
                    words.append(p2w[key])
                    i += span
                    matched = True
                    break
            if not matched:
                i += 1
        if words:
            return words
    return []


def _lm_reorder(words: List[str], lm, **kwargs) -> List[str]:
    """Stub — reordering dinonaktifkan."""
    return words


def _argmax_decode(log_probs: torch.Tensor) -> torch.Tensor:
    """Argmax per frame dari log-posterior."""
    return log_probs.argmax(dim=-1).int()


def viterbi_decode_labels(log_probs: np.ndarray) -> np.ndarray:
    """Alias argmax untuk kompatibilitas kode lama."""
    return np.argmax(log_probs, axis=-1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────
# CER — CHARACTER ERROR RATE
# ─────────────────────────────────────────────────────────────────

def compute_cer(hyp: List[str], ref: List[str]) -> float:
    """
    Character Error Rate — edit distance Levenshtein pada level karakter.

    Kata-kata digabung TANPA SPASI sebelum dihitung, sehingga perbedaan
    segmentasi kata tidak dihitung sebagai error. Cocok untuk bahasa Jawa
    di mana batas kata sering ambigu.

    Returns float [0, 1]. Kalikan 100 untuk persen.

    Contoh
    ------
    >>> compute_cer(["di","wujud","ke"], ["diwujudke"])
    0.0                     # segmentasi beda, karakter sama
    >>> compute_cer(["karo","maneh"], ["karo","maneh","ing"])
    0.25                    # 3 karakter hilang dari total 12
    """
    hyp_str = "".join(hyp)
    ref_str = "".join(ref)
    if not ref_str:
        return 0.0
    if not hyp_str:
        return 1.0
    m, n  = len(hyp_str), len(ref_str)
    dp    = list(range(n + 1))
    for i in range(1, m + 1):
        prev  = dp[:]
        dp[0] = i
        h_ch  = hyp_str[i - 1]
        for j in range(1, n + 1):
            dp[j] = (prev[j - 1] if h_ch == ref_str[j - 1]
                     else 1 + min(prev[j], dp[j - 1], prev[j - 1]))
    return min(dp[n] / n, 1.0)


# ─────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model   ,
        cfg     : Config,
        train_dl,
        val_dl  ,
        device  : str = "cpu",
        name    : str = "model",
        out     : str = ".",
    ):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.tdl    = train_dl
        self.vdl    = val_dl
        self.device = device
        self.name   = name
        self.out    = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay
        )
        steps = cfg.epochs * max(len(train_dl), 1)
        self.sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=steps, eta_min=cfg.lr_end
            )
            if cfg.scheduler == "cosine"
            else torch.optim.lr_scheduler.LinearLR(
                self.opt,
                end_factor=cfg.lr_end / cfg.lr_start,
                total_iters=steps,
            )
        )
        self.best_val_loss: float      = float("inf")
        self.best_epoch   : int        = 0
        self.history      : List[Dict] = []
        self._batch_count : int        = 0
        self._no_improve  : int        = 0

    def _epoch(self, dl, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in dl:
                feats   = batch["feats"].to(self.device)
                ali     = batch["ali"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                logits  = self.model(feats)
                B, T, C = logits.shape
                mask = (
                    torch.arange(T, device=self.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                ) & (ali >= 0)
                if mask.sum() == 0:
                    continue
                loss = F.cross_entropy(logits[mask], ali[mask], ignore_index=-1)
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
                    self.opt.step()
                    self.sched.step()
                    self._batch_count += 1
                    if (
                        hasattr(self.model, "ortho_step")
                        and self._batch_count % self.cfg.ortho_period == 0
                    ):
                        self.model.ortho_step()
                total += loss.item()
                n     += 1
        return total / max(n, 1)

    def train(self) -> List[Dict]:
        log.info(
            f"[{self.name}] Training {self.cfg.epochs} epoch — {self.device}  "
            f"(early stopping patience={self.cfg.patience})"
        )
        for ep in range(1, self.cfg.epochs + 1):
            t0       = time.time()
            tr_loss  = self._epoch(self.tdl, True)
            val_loss = self._epoch(self.vdl, False)
            lr       = self.opt.param_groups[0]["lr"]
            elapsed  = time.time() - t0
            self.history.append({
                "epoch"     : ep,
                "train_loss": round(tr_loss,  5),
                "val_loss"  : round(val_loss, 5),
                "lr"        : round(lr,       8),
                "time_s"    : round(elapsed,  1),
            })
            log.info(
                f"  [{self.name}] {ep:3d}/{self.cfg.epochs}  "
                f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                f"lr={lr:.2e}  {elapsed:.0f}s"
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = ep
                self._no_improve   = 0
                torch.save(
                    {
                        "epoch"     : ep,
                        "state_dict": self.model.state_dict(),
                        "val_loss"  : val_loss,
                        "cfg"       : self.cfg.__dict__,
                    },
                    self.out / f"{self.name}_best.pt",
                )
                log.info(f"best — epoch={ep}  val_loss={val_loss:.4f}")
            else:
                self._no_improve += 1
                if self._no_improve >= self.cfg.patience:
                    log.info(
                        f"[{self.name}] Early stopping — "
                        f"tidak ada perbaikan selama {self.cfg.patience} epoch"
                    )
                    break
        with open(self.out / f"{self.name}_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(
            f"[{self.name}] Selesai — best val_loss={self.best_val_loss:.4f}  "
            f"epoch={self.best_epoch}"
        )
        return self.history

    @property
    def best_ckpt_path(self) -> Path:
        return self.out / f"{self.name}_best.pt"

    def load_best(self) -> None:
        ckpt = torch.load(
            self.best_ckpt_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        log.info(
            f"[{self.name}] Checkpoint dimuat — epoch={ckpt['epoch']}  "
            f"val_loss={ckpt['val_loss']:.4f}"
        )

    def compute_prior(self) -> torch.Tensor:
        num_classes = self.cfg.num_pdfs
        counts      = torch.zeros(num_classes, dtype=torch.float32)
        for batch in self.tdl:
            ali   = batch["ali"]
            valid = ali[ali >= 0]
            for lid in valid:
                idx = int(lid)
                if idx < num_classes:
                    counts[idx] += 1
        prior      = counts / counts.sum().clamp(min=1.0)
        prior_path = self.out / f"{self.name}_prior.pt"
        torch.save(prior, prior_path)
        log.info(
            f"[{self.name}] Prior disimpan: {prior_path}  "
            f"(min={prior.min():.4f}, max={prior.max():.4f})"
        )
        return prior
