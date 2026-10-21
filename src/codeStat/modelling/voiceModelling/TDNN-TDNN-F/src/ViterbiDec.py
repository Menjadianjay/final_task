from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import tgt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Shared_resource import (
    NgramLM,
    normalise,
    decode_phones_improved,   # ← updated
    build_fuzzy_trie,          # ← needed for improved decoder
)

log = logging.getLogger(__name__)

_SIL_SET = {"silence", "sil", "sp", "spn", "<eps>", ""}


def build_transition_matrix(
    tg_dir: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    self_loop_weight: float = 0.5,
    floor: float = 1e-8,
    tier_name: str = "phones",
) -> np.ndarray:
    """Build log transition matrix from TextGrid alignments"""
    N = len(id2label)
    counts = np.zeros((N, N), dtype=np.float64)
    tg_paths = list(Path(tg_dir).rglob("*.TextGrid"))
    
    log.info(f"build_transition_matrix: memproses {len(tg_paths):,} TextGrid ...")
    n_bigrams = 0

    for tg_path in tg_paths:
        try:
            tg = tgt.io.read_textgrid(str(tg_path))
        except Exception:
            continue

        tier = None
        for name in tg.get_tier_names():
            if tier_name.lower() in name.lower():
                tier = tg.get_tier_by_name(name)
                break
        if tier is None:
            continue

        seq = []
        for interval in tier:
            raw = interval.text.strip().lower()
            ph = raw.rsplit("_", 1)[0] if "_" in raw else raw
            ph = ph.strip()
            if ph and ph not in _SIL_SET and ph in label2id:
                seq.append(label2id[ph])

        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1
            n_bigrams += 1

    log.info(f"build_transition_matrix: {n_bigrams:,} bigram fonem ditemukan")

    for i in range(N):
        counts[i, i] = max(counts[i, i], self_loop_weight * counts[i].sum() + floor)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans = counts / row_sums
    trans = np.clip(trans, floor, 1.0)

    log_A = np.log(trans).astype(np.float32)

    log.info(
        f"build_transition_matrix: selesai shape={log_A.shape} "
        f"min={log_A.min():.2f} max={log_A.max():.2f}"
    )
    return log_A


def _viterbi(log_emit: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    """Viterbi decoding"""
    T, N = log_emit.shape
    delta = log_emit[0] - np.log(N)
    psi = np.zeros((T, N), dtype=np.int32)

    for t in range(1, T):
        scores = delta[:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_emit[t]

    path = np.empty(T, dtype=np.int32)
    path[T - 1] = delta.argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _collapse_and_get_phones(
    path: np.ndarray, id2label: Dict[int, str]
) -> List[str]:
    """Collapse repeated labels → phone sequence"""
    phones, prev = [], None
    for lid in path:
        if lid == prev:
            continue
        prev = lid
        lbl = id2label.get(int(lid), "")
        ph = lbl.rsplit("_", 1)[0] if "_" in lbl else lbl
        ph = ph.strip().lower()
        if ph and ph not in _SIL_SET:
            phones.append(ph)
    return phones


def _phones_to_words(
    phones: List[str],
    lm: Optional[NgramLM],
    fuzzy_trie: Optional[Dict] = None,   # ← new
    lm_weight: float = 0.3,
    beam: int = 20,
) -> List[str]:
    """Phone sequence → Words using improved decoder"""
    if not phones:
        return []
    
    words, _ = decode_phones_improved(
        phones_raw=phones,
        lm=lm,
        fuzzy_trie=fuzzy_trie,
        beam=beam,
        lm_weight=lm_weight,
    )
    return [w for w in words if len(w) >= 3]


def compute_cer(hyp: List[str], ref: List[str]) -> float:
    """Character Error Rate"""
    hyp_str = "".join(hyp)
    ref_str = "".join(ref)
    if not ref_str:
        return 0.0
    if not hyp_str:
        return 1.0

    m, n = len(hyp_str), len(ref_str)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        h_ch = hyp_str[i - 1]
        for j in range(1, n + 1):
            cost = 0 if h_ch == ref_str[j - 1] else 1
            dp[j] = min(
                prev[j - 1] + cost,
                dp[j - 1] + 1,
                prev[j] + 1
            )
    return dp[n] / n


def evaluate_viterbi(
    model,
    dl,
    id2label: Dict[int, str],
    lm: Optional[NgramLM],
    fuzzy_trie: Optional[Dict],          # ← new parameter
    device: str,
    name: str,
    out_dir: str,
    prior: Optional[torch.Tensor] = None,
    log_A: Optional[np.ndarray] = None,
    lm_weight: float = 0.3,
    beam: int = 20,
    log_every: int = 200,
) -> float:
    """
    Viterbi + CER evaluation using improved decoder
    """
    N = len(id2label)
    if log_A is None:
        log.warning(f"[{name}] log_A tidak diberikan, menggunakan uniform.")
        log_A = np.full((N, N), -np.log(N), dtype=np.float32)

    assert log_A.shape == (N, N)

    log_prior_np = None
    if prior is not None:
        lp = torch.log(prior.to("cpu") + 1e-8).numpy()
        log_prior_np = lp.astype(np.float32)

    log.info(f"[{name}] evaluate_viterbi (Viterbi + CER) dimulai")

    t0 = time.time()
    model.eval()
    cer_list: List[float] = []
    details: List[Dict] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            log_post = F.log_softmax(
                model(batch["feats"].to(device)), dim=-1
            ).cpu().numpy().astype(np.float32)

            B = log_post.shape[0]
            for b in range(B):
                T = int(batch["lengths"][b])
                ref = normalise(batch["texts"][b]).split()
                if not ref:
                    continue

                log_emit = log_post[b, :T]
                if log_prior_np is not None:
                    log_emit = log_emit - log_prior_np

                path = _viterbi(log_emit, log_A)
                phones = _collapse_and_get_phones(path, id2label)
                words = _phones_to_words(
                    phones, lm, fuzzy_trie, lm_weight=lm_weight, beam=beam
                )

                cer = compute_cer(words, ref)
                cer_list.append(cer)

                details.append({
                    "utt_id"     : batch["audio_ids"][b],
                    "referensi"  : " ".join(ref),
                    "phones_raw" : " ".join(phones),
                    "hipotesis"  : " ".join(words),
                    "ref_len"    : len(ref),
                    "hyp_len"    : len(words),
                    "cer_utt"    : round(cer * 100, 2),
                })

            if (batch_idx + 1) % max(1, log_every // B) == 0:
                done = len(details)
                cur_cer = sum(cer_list) / len(cer_list) * 100 if cer_list else 0
                log.info(f"[{name}] {done} utt — CER = {cur_cer:.2f}%")

    cer_mean = (sum(cer_list) / len(cer_list) * 100) if cer_list else 100.0

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_dir) / f"{name}_cer_viterbi.csv")

    df = pd.DataFrame(details)
    summary = {
        "utt_id"    : "TOTAL",
        "referensi" : "",
        "phones_raw": "",
        "hipotesis" : "",
        "ref_len"   : int(df["ref_len"].sum()),
        "hyp_len"   : int(df["hyp_len"].sum()),
        "cer_utt"   : round(cer_mean, 2),
    }

    pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(
        csv_path, index=False, encoding="utf-8"
    )

    log.info(
        f"[{name}] CER (Viterbi) = {cer_mean:.2f}% | "
        f"{len(cer_list):,} utt | {time.time()-t0:.1f}s → {csv_path}"
    )
    return cer_mean