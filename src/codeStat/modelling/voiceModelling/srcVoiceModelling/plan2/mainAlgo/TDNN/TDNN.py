import os
import re
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set   # [FIX-8] tambah Set


# [FIX-10] Path alignment dapat di-override via environment variable
# sehingga kode tidak harus di-edit setiap pindah mesin.
ALIGNMENT_CSV = os.environ.get(
    "TDNN_ALIGNMENT_CSV",
    os.path.join(
        r"D:\doc\2026\FINAL_CODE\src\codeStat\modelling"
        r"\voiceModelling\srcVoiceModelling\plan2\kaldialignment\out",
        "alignment_map.csv",
    ),
)

# Kolom yang dibutuhkan di alignment_map.csv
COL_AUDIO_ID    = "audio_id"
COL_KATA        = "kata"
COL_FRAME_START = "frame_start"
COL_FRAME_END   = "frame_end"


class AlignmentLoader:
    """
    Memuat alignment_map.csv dari MFA dan menyediakan:
      - get_total_frames(audio_id)          → int
      - get_kata_list(audio_id)             → List[str]
      - get_frame_ranges(audio_id)          → List[(frame_start, frame_end, kata)]
      - is_valid_for_ctc(audio_id, T, L, min_ratio) → (bool, str)
      - get_word_lengths(audio_id)          → Dict[str, List[int]]
      - get_utterance_duration_frames(id)   → int

    Referensi arsitektur: Paper 2 (Kipyatkova & Kagirov, 2023) — TDNN-F/LF-MMI
    untuk low-resource Livvi-Karelian. Validasi CTC mengikuti constraint
    T >= 2L+1 (blank di antara setiap label).
    """

    def __init__(self, alignment_csv: str):
        if not os.path.exists(alignment_csv):
            raise FileNotFoundError(
                f"alignment_map.csv tidak ditemukan: {alignment_csv}\n"
                f"Jalankan FASE 1 (MFA) terlebih dahulu."
            )
        self.df = pd.read_csv(alignment_csv)

        # Validasi kolom
        required = [COL_AUDIO_ID, COL_KATA, COL_FRAME_START, COL_FRAME_END]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan di alignment_map.csv: {missing}")

        # [FIX-4] Pastikan tipe numerik sebelum indexing
        self.df[COL_FRAME_START] = pd.to_numeric(self.df[COL_FRAME_START], errors="coerce").fillna(0).astype(int)
        self.df[COL_FRAME_END]   = pd.to_numeric(self.df[COL_FRAME_END],   errors="coerce").fillna(0).astype(int)

        # Index per audio_id untuk lookup O(1)
        self._index: Dict[str, pd.DataFrame] = {}
        for audio_id, grp in self.df.groupby(COL_AUDIO_ID):
            self._index[str(audio_id)] = grp.reset_index(drop=True)

        print(f"  [Alignment] {len(self.df)} interval, "
              f"{len(self._index)} audio, "
              f"{self.df[COL_KATA].nunique()} kata unik")

    def has(self, audio_id: str) -> bool:
        """True jika audio_id ada dalam indeks alignment."""
        return str(audio_id) in self._index

    def get_kata_list(self, audio_id: str) -> List[str]:
        """Urutan kata dari alignment (ground truth MFA)."""
        if not self.has(audio_id):
            return []
        return list(self._index[str(audio_id)][COL_KATA])

    def get_frame_ranges(self, audio_id: str) -> List[Tuple[int, int, str]]:
        """List (frame_start, frame_end, kata) per interval alignment."""
        if not self.has(audio_id):
            return []
        grp = self._index[str(audio_id)]
        return [
            (int(row[COL_FRAME_START]), int(row[COL_FRAME_END]), str(row[COL_KATA]))
            for _, row in grp.iterrows()
        ]

    def get_aligned_frame_count(self, audio_id: str) -> int:
        """
        Total frame yang terkover alignment untuk audio ini.
        Berbeda dari T total — hanya frame yang punya label kata.
        """
        if not self.has(audio_id):
            return 0
        grp = self._index[str(audio_id)]
        return int(
            (grp[COL_FRAME_END] - grp[COL_FRAME_START]).sum()
        )

    def is_valid_for_ctc(
        self,
        audio_id  : str,
        T_total   : int,
        n_phonemes: int,
        min_ratio : int = 2,    # [FIX-1] default 4→2 sesuai formula CTC: T >= 2L+1
    ) -> Tuple[bool, str]:
        """
        Validasi apakah sampel layak masuk CTC training.

        Constraint CTC (Graves et al.): T >= 2L + 1
        dimana T = jumlah frame, L = jumlah label (fonem).

        Paper 2 (Kipyatkova 2023) menggunakan LF-MMI yang sedikit
        lebih fleksibel dari CTC murni, namun constraint minimum tetap berlaku.

        Returns:
          (True, "ok")      jika valid
          (False, alasan)   jika tidak valid
        """
        # [FIX-4] Guard n_phonemes nol
        if n_phonemes <= 0:
            return False, "zero_phonemes"

        if not self.has(audio_id):
            # Tidak ada alignment → fallback ke estimasi
            if T_total < min_ratio * n_phonemes:
                return False, f"no_alignment_too_short(T={T_total},L={n_phonemes})"
            return True, "ok_no_alignment"

        aligned_frames = self.get_aligned_frame_count(audio_id)
        if aligned_frames == 0:
            return False, "alignment_empty"

        # CTC butuh setidaknya 1 frame per token output
        # (setiap fonem + blank di antaranya = 2*L+1 minimal)
        min_frames_needed = 2 * n_phonemes + 1
        if T_total < min_frames_needed:
            return False, (
                f"T_too_short(T={T_total}, "
                f"need={min_frames_needed}, L={n_phonemes})"
            )

        # [FIX-4] denominator aman — n_phonemes sudah dicek > 0
        frames_per_phoneme = aligned_frames / n_phonemes
        if frames_per_phoneme < 2.0:
            return False, (
                f"too_dense(aligned={aligned_frames}, "
                f"L={n_phonemes}, ratio={frames_per_phoneme:.1f})"
            )

        return True, "ok"

    def get_word_lengths(self, audio_id: str) -> Dict[str, List[int]]:
        """
        [FIX-5] Panjang (dalam frame) tiap kemunculan kata untuk audio ini.
        Sebelumnya menyimpan satu nilai per kata (overwrite jika duplikat).
        Kini menyimpan List[int] agar semua kemunculan kata terwakili.

        Dipakai oleh curriculum sampler untuk menghitung total durasi.
        """
        if not self.has(audio_id):
            return {}
        grp = self._index[str(audio_id)]
        result: Dict[str, List[int]] = defaultdict(list)
        for _, row in grp.iterrows():
            kata = str(row[COL_KATA])
            dur  = int(row[COL_FRAME_END]) - int(row[COL_FRAME_START])
            result[kata].append(dur)
        return dict(result)

    def get_utterance_duration_frames(self, audio_id: str) -> int:
        """Durasi total utterance dalam frame (frame_end terakhir)."""
        if not self.has(audio_id):
            return 0
        grp = self._index[str(audio_id)]
        return int(grp[COL_FRAME_END].max())


# ─────────────────────────────────────────────────────────────
#  load_data_with_alignment()
# ─────────────────────────────────────────────────────────────

def load_data_with_alignment(
    csv_path         : str,
    alignment_loader : AlignmentLoader,
    extract_features_raw,
    normalize_text,
    build_phoneme_vocab,
    encode_phoneme,
    find_wav,
    fixed_config     : dict,
    col_path_data    : str = "Path_Data",
    col_sentences    : str = "sentences",
) -> tuple:
    """
    Versi load_data() yang menggunakan AlignmentLoader untuk:
      1. Filter sampel dengan validasi frame aktual dari MFA (bukan estimasi)
      2. Log audio tanpa alignment (debugging coverage MFA)

    [FIX-7] Tambah guard untuk sequences_raw kosong sebelum akses
            next(iter(...)) yang akan raise StopIteration.

    Returns:
      sequences_raw, targets, word_refs, phn_refs,
      phn2id, id2phn, vocab, actual_dim,
      audio_ids_with_alignment  ← set audio_id yang punya alignment
    """
    print(f"  Membaca {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape CSV : {df.shape}")

    for col in [col_path_data, col_sentences]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    sentences_norm = [normalize_text(str(s)) for s in df[col_sentences].tolist()]
    vocab, phn2id, id2phn = build_phoneme_vocab(sentences_norm)
    print(f"  Vocab: {len(vocab)-1} fonem + 1 blank = {len(vocab)}")

    sequences_raw  : Dict[str, np.ndarray] = {}
    targets        : Dict[str, list]        = {}
    word_refs      : Dict[str, list]        = {}
    phn_refs       : Dict[str, list]        = {}
    audio_with_aln : Set[str]               = set()   # [FIX-8]

    skipped      = 0
    skip_reasons : Dict[str, int] = defaultdict(int)
    n_no_aln     = 0

    print(f"  Mengekstrak MFCC + validasi alignment ({len(df)} file) ...")

    for idx, row in df.iterrows():
        audio_id = os.path.splitext(
            os.path.basename(str(row[col_path_data])))[0]
        kalimat  = normalize_text(str(row[col_sentences]))
        # Kembalikan karakter khusus sebelum split kata
        words    = kalimat.replace('\x01e', 'é').replace('\x02e', 'è').split()

        wav_path = str(row[col_path_data])
        if not os.path.isfile(wav_path):
            wav_path = find_wav(audio_id)
        if not wav_path:
            skip_reasons["file_not_found"] += 1
            skipped += 1
            continue

        try:
            feats  = extract_features_raw(wav_path)
            T      = feats.shape[0]
            target = encode_phoneme(kalimat, phn2id)
            L      = len(target)

            if T == 0 or L == 0:
                skip_reasons["empty"] += 1
                skipped += 1
                continue

            valid, reason = alignment_loader.is_valid_for_ctc(
                audio_id, T, L,
                min_ratio=fixed_config.get("ctc_min_ratio", 2)   # [FIX-1]
            )
            if not valid:
                skip_reasons[f"aln_{reason}"] += 1
                skipped += 1
                continue

            has_aln = alignment_loader.has(audio_id)
            if not has_aln:
                n_no_aln += 1
            else:
                audio_with_aln.add(audio_id)

            sequences_raw[audio_id] = feats
            targets[audio_id]       = target
            word_refs[audio_id]     = words
            phn_refs[audio_id]      = [id2phn[i] for i in target]

        except Exception as e:
            skip_reasons["error"] += 1
            skipped += 1
            if idx < 10:
                print(f"  ⚠ {audio_id}: {e}")

        if (idx + 1) % 200 == 0:
            print(f"    [{idx+1}/{len(df)}] ok={len(sequences_raw)} skip={skipped}")

    # [FIX-7] Guard sequences_raw kosong
    if not sequences_raw:
        raise RuntimeError(
            "Tidak ada sampel yang berhasil dimuat. "
            "Periksa path CSV, file WAV, dan alignment."
        )

    actual_dim = next(iter(sequences_raw.values())).shape[1]

    print(f"\n  Berhasil     : {len(sequences_raw)}")
    print(f"  Dilewati     : {skipped}")
    print(f"  Punya aln    : {len(audio_with_aln)}")
    print(f"  Tanpa aln    : {n_no_aln} (fallback ke estimasi T>=2L+1)")
    for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {cnt}")

    return (
        sequences_raw, targets, word_refs, phn_refs,
        phn2id, id2phn, vocab, actual_dim,
        audio_with_aln,
    )


# ─────────────────────────────────────────────────────────────
#  AlignmentCurriculumSampler
#
#  Curriculum learning berbasis paper Livvi-Karelian (Paper 2):
#  epoch awal → utterance pendek dulu (model belajar fonem
#  sederhana lebih stabil), epoch akhir → shuffle acak penuh.
#
#  Referensi: Kipyatkova & Kagirov (2023) Section 4.1 —
#  "The number of training epochs was increased when applying
#  SpecAugment... to account for the augmented data."
#  Curriculum adalah komplementasi augmentasi untuk stabilitas awal.
# ─────────────────────────────────────────────────────────────

class AlignmentCurriculumSampler(Sampler):
    """
    Curriculum learning sampler berbasis panjang utterance dari alignment.

    Epoch 1..n_epochs_curriculum:
      - Urutkan utterance dari pendek ke panjang (dalam frame)
      - Shuffle dalam bucket ±noise_frac durasi untuk mencegah overfitting urutan

    Epoch > n_epochs_curriculum:
      - Shuffle acak penuh (identik dengan shuffle=True default)

    [FIX-2] Seed per epoch dibuat dari seed_base + epoch*31 (prime multiplier)
    untuk menghindari korelasi antar-epoch yang muncul jika hanya seed + epoch.
    """

    def __init__(
        self,
        audio_ids            : List[str],
        alignment_loader     : AlignmentLoader,
        sequences            : Dict[str, np.ndarray],
        n_epochs_curriculum  : int = 15,
        current_epoch        : int = 1,
        bucket_noise_frac    : float = 0.20,
        seed                 : int = 42,
    ):
        self.ids               = audio_ids
        self.n_curriculum      = n_epochs_curriculum
        self.current_epoch     = current_epoch
        self.bucket_noise_frac = bucket_noise_frac
        self._seed_base        = seed
        # [FIX-2] Gunakan prime multiplier agar epoch seed tidak berkorelasi
        self.rng = np.random.default_rng(seed + current_epoch * 31)

        # Hitung panjang tiap utterance; prioritaskan alignment
        self._lengths: Dict[str, int] = {}
        for aid in audio_ids:
            aln_dur = alignment_loader.get_utterance_duration_frames(aid)
            if aln_dur > 0:
                self._lengths[aid] = aln_dur
            elif aid in sequences:
                self._lengths[aid] = sequences[aid].shape[0]
            else:
                self._lengths[aid] = 0

    def set_epoch(self, epoch: int) -> None:
        """Panggil di awal tiap epoch di run_training()."""
        self.current_epoch = epoch
        self.rng = np.random.default_rng(self._seed_base + epoch * 31)   # [FIX-2]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        if self.current_epoch > self.n_curriculum:
            # Fase normal: shuffle acak penuh
            perm = self.rng.permutation(len(self.ids))
            return iter(perm.tolist())

        # Fase curriculum: urutkan pendek → panjang dengan sedikit noise
        progress    = self.current_epoch / max(self.n_curriculum, 1)
        noise_scale = self.bucket_noise_frac * progress

        lengths = np.array(
            [self._lengths.get(aid, 0) for aid in self.ids],
            dtype=np.float32
        )
        noise = self.rng.uniform(
            1.0 - noise_scale, 1.0 + noise_scale, size=len(lengths)
        )
        noisy_lengths = lengths * noise
        order = np.argsort(noisy_lengths)
        return iter(order.tolist())


# ─────────────────────────────────────────────────────────────
#  make_train_loader_with_curriculum()
# ─────────────────────────────────────────────────────────────

def make_train_loader_with_curriculum(
    train_ds,
    train_ids        : List[str],
    alignment_loader : AlignmentLoader,
    sequences        : Dict[str, np.ndarray],
    cfg              : dict,
    collate_fn,
    current_epoch    : int,
) -> "torch.utils.data.DataLoader":
    """
    Buat DataLoader training dengan AlignmentCurriculumSampler.

    [FIX-3] num_workers diteruskan dari cfg (sebelumnya di-hardcode 0).
    Pada Windows, num_workers=0 masih disarankan untuk menghindari
    multiprocessing error; konfigurasi melalui cfg["num_workers"].

    Panggil di dalam loop epoch di run_training():

        for epoch in range(1, max_epochs + 1):
            if alignment_loader is not None:
                train_loader = make_train_loader_with_curriculum(
                    train_ds, train_ids, alignment_loader,
                    sequences, cfg, collate_fn, epoch
                )
            ...
    """
    if not isinstance(train_ds.ids, list):
        train_ds.ids = list(train_ids)

    sampler = AlignmentCurriculumSampler(
        audio_ids           = train_ds.ids,
        alignment_loader    = alignment_loader,
        sequences           = sequences,
        n_epochs_curriculum = cfg.get("n_epochs_curriculum", 15),
        current_epoch       = current_epoch,
        seed                = cfg.get("random_seed", 42),
    )
    # [FIX-3] Ambil num_workers dari cfg, bukan hardcode 0
    nw = cfg.get("num_workers", 0)
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        sampler     = sampler,
        collate_fn  = collate_fn,
        num_workers = nw,
        pin_memory  = (torch.cuda.is_available() and nw == 0),
    )


# ─────────────────────────────────────────────────────────────
#  evaluate_wer_with_alignment()
# ─────────────────────────────────────────────────────────────

def evaluate_wer_with_alignment(
    model,
    eval_ids         : List[str],
    sequences        : Dict[str, np.ndarray],
    word_refs        : Dict[str, list],
    phn_refs         : Dict[str, list],
    id2phn           : dict,
    alignment_loader : AlignmentLoader,
    ctc_beam_decode_tokens,
    tokens_to_words,
    phoneme_tokens_to_text,
    levenshtein_words,
    beam_width       : int = 10,
    silence_token    : str = '<sp>',
) -> Tuple[List[dict], dict, "pd.DataFrame"]:
    """
    Evaluasi WER dengan breakdown per-segmen kata dari alignment.

    [FIX-6] model.eval() dipastikan dipanggil. get_log_probs() menggunakan
    next(model.parameters()).device secara internal sehingga inference
    selalu pada device yang benar tanpa perlu memindah tensor secara manual.

    Menghasilkan df_segmen dengan kolom:
      audio_id, kata_ref, frame_start, frame_end, dur_frames,
      hipotesis, WER_segmen, S, D, I, benar

    Referensi evaluasi: Paper 1 (Tong et al. 2022) menggunakan WER per
    aksen; Paper 2 (Kipyatkova 2023) menggunakan WER dan RTF.
    Fungsi ini menyediakan WER per segmen kata menggunakan timestamp MFA.

    Returns:
      wer_rows    : List[dict] — per audio
      summary     : dict       — ringkasan WER corpus
      df_segmen   : DataFrame  — WER per segmen kata dengan timestamp
    """
    model.eval()   # [FIX-6]
    wer_rows    = []
    segmen_rows = []
    tS = tD = tI = tN = 0

    for aid in eval_ids:
        feat      = sequences[aid]
        T_total   = feat.shape[0]

        # [FIX-6] get_log_probs sudah handle device secara internal
        lp        = model.get_log_probs(feat)    # (T, V)
        hyp_toks  = ctc_beam_decode_tokens(lp, id2phn, beam_width)
        hyp_words = tokens_to_words(hyp_toks)
        hyp_text  = phoneme_tokens_to_text(hyp_toks)
        ref_words = word_refs.get(aid, [])

        wer_val, S, D, I, N = levenshtein_words(ref_words, hyp_words)
        tS += S; tD += D; tI += I; tN += N

        wer_rows.append({
            "audio_id"      : aid,
            "WER"           : round(wer_val * 100, 4),
            "S": S, "D": D, "I": I, "N": N,
            "referensi"     : " ".join(ref_words),
            "hipotesis"     : hyp_text,
            "has_alignment" : alignment_loader.has(aid),
        })

        if not alignment_loader.has(aid):
            continue

        frame_ranges = alignment_loader.get_frame_ranges(aid)

        for frame_start, frame_end, kata_ref in frame_ranges:
            fs = max(0, min(frame_start, T_total - 1))
            fe = max(fs + 1, min(frame_end, T_total))

            feat_seg = feat[fs:fe]
            if feat_seg.shape[0] == 0:
                continue

            lp_seg      = model.get_log_probs(feat_seg)
            hyp_seg_tok = ctc_beam_decode_tokens(
                lp_seg, id2phn, beam_width=min(beam_width, 5)
            )
            hyp_seg_clean = [
                t for t in hyp_seg_tok
                if t not in (silence_token, '<sp>', '<blank>')
            ]
            hyp_seg_text = phoneme_tokens_to_text(hyp_seg_clean)

            ref_seg_words = [kata_ref]
            hyp_seg_words = tokens_to_words(hyp_seg_tok)
            wer_seg, sS, sD, sI, sN = levenshtein_words(
                ref_seg_words, hyp_seg_words
            )

            segmen_rows.append({
                "audio_id"   : aid,
                "kata_ref"   : kata_ref,
                "frame_start": frame_start,
                "frame_end"  : frame_end,
                "dur_frames" : frame_end - frame_start,
                "hipotesis"  : hyp_seg_text,
                "WER_segmen" : round(wer_seg * 100, 2),
                "S": sS, "D": sD, "I": sI,
                "benar"      : (wer_seg == 0.0),
            })

    # Ringkasan utama
    wer_corpus = min((tS + tD + tI) / max(tN, 1), 1.0) * 100
    wl = [r["WER"] for r in wer_rows]
    summary = {
        "wer_corpus" : round(wer_corpus, 4),
        "wer_mean"   : round(float(np.mean(wl)), 4) if wl else 100.0,
        "wer_min"    : round(float(np.min(wl)),  4) if wl else 100.0,
        "wer_max"    : round(float(np.max(wl)),  4) if wl else 100.0,
        "total_S"    : tS, "total_D": tD, "total_I": tI, "total_N": tN,
        "n_audio"    : len(eval_ids),
        "n_with_aln" : sum(1 for r in wer_rows if r["has_alignment"]),
    }

    df_segmen = pd.DataFrame(segmen_rows) if segmen_rows else pd.DataFrame()

    if not df_segmen.empty:
        acc_segmen = df_segmen["benar"].mean() * 100
        print(f"  Akurasi per-kata (segmen): {acc_segmen:.2f}%")
        print(f"  Total segmen dievaluasi  : {len(df_segmen)}")
        salah = (
            df_segmen[~df_segmen["benar"]]
            .groupby("kata_ref")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        if not salah.empty:
            print("  Top kata paling sering salah:")
            for kata, cnt in salah.items():
                print(f"    '{kata}': {int(cnt)}x")

    return wer_rows, summary, df_segmen


# ─────────────────────────────────────────────────────────────
#  PANDUAN INTEGRASI KE TDNN_torch.py
# ─────────────────────────────────────────────────────────────
#
#  from TDNN_fixed import (
#      AlignmentLoader,
#      load_data_with_alignment,
#      make_train_loader_with_curriculum,
#      evaluate_wer_with_alignment,
#      ALIGNMENT_CSV,
#  )
#
#  Perubahan FIXED_CONFIG yang disarankan:
#    "ctc_min_ratio"      : 2,   # sesuai CTC constraint T >= 2L+1
#    "n_epochs_curriculum": 15,
#    "use_alignment"      : True,
#
#  Lihat TDNN_torch_fixed.py untuk contoh integrasi lengkap.

if __name__ == "__main__":
    print("Mengecek alignment_map.csv ...")
    try:
        al = AlignmentLoader(ALIGNMENT_CSV)
        sample_ids = list(al._index.keys())[:3]
        for aid in sample_ids:
            ranges = al.get_frame_ranges(aid)
            dur    = al.get_utterance_duration_frames(aid)
            wl     = al.get_word_lengths(aid)
            print(f"  {aid}: {len(ranges)} kata, dur={dur} frame, "
                  f"word_groups={len(wl)}")
        print("OK — AlignmentLoader siap.")
    except FileNotFoundError as e:
        print(f"  {e}")