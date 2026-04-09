import sys
import os
sys.path.insert(0, r"D:\doc\2026\FINAL_CODE\src")

import os
import glob
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import codeStat.modelimports.modelFunction as func
from codeStat.modelimports.modelFunction import *
import codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram as lexicon_ngram
from codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram import *

def baca_transkrip_csv(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        audio_id = os.path.splitext(os.path.basename(str(row["Path_Data"])))[0]
        kalimat  = str(row["sentences"]).lower().strip()
        result[audio_id] = kalimat
    return result

# BAGIAN 2 — CACHE MFCC 

def build_mfcc_cache(
    sampled_root   : str,
    dict_transkrip : Dict[str, str],
    verbose        : bool = True,
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    wav_files  = glob.glob(os.path.join(sampled_root, "**/*.wav"), recursive=True)
    cache_mfcc : Dict[str, np.ndarray] = {}
    meta_list  : List[Dict]            = []
    n_total    = len(wav_files)
    n_ok       = 0

    if verbose:
        print(f" Mengekstrak MFCC dari {n_total} file audio ...")

    t0 = time.time()
    for idx, wav_path in enumerate(wav_files):
        try:
            audio_id = os.path.splitext(os.path.basename(wav_path))[0]
            label    = dict_transkrip.get(audio_id, "")
            if label == "":
                continue  # skip audio tanpa transkrip

            fitur = func.extract_features(wav_path)   # (T, 39)
            cache_mfcc[audio_id] = fitur
            meta_list.append({
                "Id"           : audio_id,
                "Label"        : label,
                "Jumlah Frame" : fitur.shape[0],
                "Dimensi Fitur": fitur.shape[1],
            })
            n_ok += 1

            if verbose and (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (idx + 1) * (n_total - idx - 1)
                print(f"    [{idx+1}/{n_total}] OK={n_ok}  "
                      f"Elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

        except Exception as e:
            if verbose:
                print(f"    [SKIP] {wav_path} : {e}")

    if verbose:
        print(f"  Selesai: {n_ok} audio di-cache dalam {time.time()-t0:.1f}s")

    return cache_mfcc, meta_list
# BAGIAN 3 — KUMPULKAN FRAME PER KATA 

def pecah_frame_per_kata(
    fitur    : np.ndarray,
    kata_list: List[str],
) -> Dict[str, np.ndarray]:
    T, n_kata = fitur.shape[0], len(kata_list)
    if n_kata == 0 or T == 0:
        return {}

    # Bobot = panjang fonem
    bobot = np.array([max(len(func.kata_ke_fonem(k)), 1) for k in kata_list], dtype=float)
    bobot /= bobot.sum()

    n_frame = (bobot * T).astype(int)
    sisa    = T - n_frame.sum()
    for i in range(sisa):
        n_frame[i % n_kata] += 1

    hasil: Dict[str, np.ndarray] = {}
    idx = 0
    for i, kata in enumerate(kata_list):
        n = n_frame[i]
        if n <= 0:
            continue
        seg = fitur[idx: idx + n]
        if kata in hasil:
            hasil[kata] = np.vstack([hasil[kata], seg])
        else:
            hasil[kata] = seg
        idx += n

    return hasil


def kumpulkan_frame_per_kata_dari_cache(
    cache_mfcc   : Dict[str, np.ndarray],
    dict_transkrip: Dict[str, str],
    verbose       : bool = True,
) -> Dict[str, List[np.ndarray]]:
    frame_per_kata: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_audio = len(cache_mfcc)

    if verbose:
        print(f"  Mengumpulkan frame per kata dari {n_audio}")

    for audio_id, fitur in cache_mfcc.items():
        label     = dict_transkrip.get(audio_id, "")
        if label == "":
            continue
        kata_list = lexicon_ngram.normalisasi(label).split()
        if not kata_list:
            continue

        pecahan = pecah_frame_per_kata(fitur, kata_list)
        for kata, frame in pecahan.items():
            frame_per_kata[kata].append(frame)

    if verbose:
        print(f"  Kata unik dengan frame: {len(frame_per_kata)}")

    return frame_per_kata
# BAGIAN 4 — LATIH GMM PER KATA
def latih_gmm_per_kata(
    frame_per_kata: Dict[str, List[np.ndarray]],
    n_components  : int   = 4,
    n_iter        : int   = 100,
    tol           : float = 1e-3,
    min_frame     : int   = 10,
    verbose       : bool  = True,
) -> Dict[str, func.GaussianMixtureModel]:
    gmm_per_kata: Dict[str, GaussianMixtureModel] = {}
    kata_list = sorted(frame_per_kata.keys())
    n_kata    = len(kata_list)

    if verbose:
        print(f"  Melatih GMM untuk {n_kata} kata unik "
              f"(K={n_components}, iter={n_iter}) ...")

    t0 = time.time()
    for i, kata in enumerate(kata_list):
        frames_list = frame_per_kata[kata]
        X = np.vstack(frames_list) 

        if X.shape[0] < min_frame:
            continue  
        K_eff = min(n_components, X.shape[0])  
        gmm   = GaussianMixtureModel(
            n_components = K_eff,
            n_iter       = n_iter,
            tol          = tol,
            reg_covar    = 1e-6,
            random_state = hash(kata) % 100_000,
        )
        gmm.fit(X)
        gmm_per_kata[kata] = gmm

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n_kata}] elapsed={time.time()-t0:.1f}s")

    if verbose:
        print(f"  GMM terlatih: {len(gmm_per_kata)} kata dalam {time.time()-t0:.1f}s")

    return gmm_per_kata
# BAGIAN 5 — DEKODE AUDIO

def dekode_audio(
    fitur        : np.ndarray,
    kata_ref_list: List[str],
    gmm_per_kata : Dict[str, GaussianMixtureModel],
    ngram_model  : NGramModel,
    top_k        : int = 10,
) -> List[str]:
    if not gmm_per_kata or not kata_ref_list:
        return []

    pecahan = pecah_frame_per_kata(fitur, kata_ref_list)

    vocab_gmm = list(gmm_per_kata.keys())

    hasil_kata: List[str] = []

    for kata_seg in kata_ref_list:
        seg_fitur = pecahan.get(kata_seg, fitur[:max(5, fitur.shape[0] // len(kata_ref_list))])

        if seg_fitur.shape[0] == 0:
            hasil_kata.append(vocab_gmm[0] if vocab_gmm else kata_seg)
            continue

        skor = {
            kata: gmm_per_kata[kata].score(seg_fitur)
            for kata in vocab_gmm
        }
        sorted_kata = sorted(skor, key=skor.get, reverse=True)[:top_k]
        best_gmm = sorted_kata[0]
        hasil_kata.append(best_gmm)
        
    kandidat = [hasil_kata]
    return ngram_model.rescore(kandidat)


def dekode_semua_audio(
    cache_mfcc   : Dict[str, np.ndarray],
    dict_transkrip: Dict[str, str],
    gmm_per_kata : Dict[str, GaussianMixtureModel],
    ngram_model  : NGramModel,
    top_k        : int = 10,
    verbose       : bool = True,
) -> List[Dict]:
    hasil_wer = []
    n_total   = len(cache_mfcc)
    n_ok      = 0
    t0        = time.time()

    if verbose:
        print(f"  Dekode {n_total} audio ...")

    for idx, (audio_id, fitur) in enumerate(cache_mfcc.items()):
        try:
            ref_kalimat = dict_transkrip.get(audio_id, "")
            if ref_kalimat == "":
                continue

            ref_kata = normalisasi(ref_kalimat).split()
            if not ref_kata:
                continue

            hyp_kata  = dekode_audio(fitur, ref_kata, gmm_per_kata, ngram_model, top_k)
            wer_hasil = compute_wer(ref_kata, hyp_kata)

            hasil_wer.append({
                "Id"        : audio_id,
                "Referensi" : " ".join(ref_kata),
                "Hipotesis" : " ".join(hyp_kata),
                "WER (%)"   : wer_hasil["wer_persen"],
                "S"         : wer_hasil["S"],
                "D"         : wer_hasil["D"],
                "I"         : wer_hasil["I"],
                "N"         : wer_hasil["N"],
            })
            n_ok += 1

            if verbose and (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{idx+1}/{n_total}] decoded={n_ok} "
                      f"elapsed={elapsed:.1f}s")

        except Exception as e:
            if verbose:
                print(f"    [ERROR] {audio_id}: {e}")

    if verbose:
        print(f"  Selesai dekode: {n_ok} audio dalam {time.time()-t0:.1f}s")

    return hasil_wer

# BAGIAN 6 — RINGKASAN WER

def ringkasan_wer(df_wer: pd.DataFrame, label: str = "") -> Dict:
    if len(df_wer) == 0:
        print("  [WARNING] Tidak ada data WER")
        return {}

    total_S    = int(df_wer["S"].sum())
    total_D    = int(df_wer["D"].sum())
    total_I    = int(df_wer["I"].sum())
    total_N    = int(df_wer["N"].sum())
    wer_corpus = (total_S + total_D + total_I) / total_N * 100 if total_N > 0 else 100.0

    print(f"\n── Ringkasan WER {label} {'─'*(30-len(label))}")
    print(f"  Jumlah audio  : {len(df_wer)}")
    print(f"  WER Corpus    : {wer_corpus:.2f}%")
    print(f"  WER Mean      : {df_wer['WER (%)'].mean():.2f}%")
    print(f"  WER Median    : {df_wer['WER (%)'].median():.2f}%")
    print(f"  WER Std       : {df_wer['WER (%)'].std():.2f}%")
    print(f"  WER Min       : {df_wer['WER (%)'].min():.2f}%")
    print(f"  WER Max       : {df_wer['WER (%)'].max():.2f}%")
    print(f"  Total S       : {total_S}")
    print(f"  Total D       : {total_D}")
    print(f"  Total I       : {total_I}")
    print(f"  Total N (kata): {total_N}")
    print(f"{'─'*40}")

    return {
        "n_audio"    : len(df_wer),
        "wer_corpus" : round(wer_corpus, 2),
        "wer_mean"   : round(df_wer["WER (%)"].mean(), 2),
        "wer_median" : round(df_wer["WER (%)"].median(), 2),
        "wer_std"    : round(df_wer["WER (%)"].std(), 2),
        "wer_min"    : round(df_wer["WER (%)"].min(), 2),
        "wer_max"    : round(df_wer["WER (%)"].max(), 2),
        "total_S"    : total_S,
        "total_D"    : total_D,
        "total_I"    : total_I,
        "total_N"    : total_N,
    }


# BAGIAN 7 — HYPERPARAMETER TUNING (EFISIEN)

def hyperparameter_tuning(
    frame_per_kata : Dict[str, List[np.ndarray]],
    cache_mfcc     : Dict[str, np.ndarray],
    dict_transkrip : Dict[str, str],
    ngram_model    : NGramModel,
    dir_save       : str = ".",
    n_gram         : int = 3,
    verbose        : bool = True,
) -> pd.DataFrame:
    grid = [
        (K, tol, max_iter)
        for K        in [4, 8, 16, 32]
        for tol      in [1e-3, 1e-4]
        for max_iter in [50, 100, 200]
    ]

    n_eksp = len(grid)
    print(f"\n[Hyperparameter Tuning] {n_eksp} eksperimen")
    print(f"  (MFCC cache: {len(cache_mfcc)} audio — tidak di-extract ulang)")
    print(f"  (frame_per_kata: {len(frame_per_kata)} kata — tidak dikumpulkan ulang)")

    hasil_tuning = []
    t0_total     = time.time()

    for idx, (K, tol, max_iter) in enumerate(grid, 1):
        t0 = time.time()
        print(f"\n  [{idx}/{n_eksp}] K={K}, tol={tol:.0e}, max_iter={max_iter}")

        gmm_per_kata = latih_gmm_per_kata(
            frame_per_kata = frame_per_kata,
            n_components   = K,
            n_iter         = max_iter,
            tol            = tol,
            verbose        = False,  
        )
        hasil_wer = dekode_semua_audio(
            cache_mfcc    = cache_mfcc,
            dict_transkrip= dict_transkrip,
            gmm_per_kata  = gmm_per_kata,
            ngram_model   = ngram_model,
            verbose       = False,
        )

        if hasil_wer:
            df_tmp     = pd.DataFrame(hasil_wer)
            total_errs = df_tmp["S"].sum() + df_tmp["D"].sum() + df_tmp["I"].sum()
            total_N    = df_tmp["N"].sum()
            wer_corpus = total_errs / total_N * 100 if total_N > 0 else 100.0
        else:
            wer_corpus = 100.0

        elapsed = time.time() - t0
        print(f"         WER: {wer_corpus:.2f}%  (selesai dalam {elapsed:.1f}s)")

        hasil_tuning.append({
            "Eksperimen"     : idx,
            "K Komponen"     : K,
            "Toleransi"      : tol,
            "Maks Iterasi"   : max_iter,
            "WER Corpus (%)": round(wer_corpus, 2),
            "Waktu (s)"     : round(elapsed, 1),
        })

    df_tuning = pd.DataFrame(hasil_tuning).sort_values("WER Corpus (%)")
    df_tuning.to_csv(f"{dir_save}/HasilHyperparameterTuning_GMM.csv", index=False)

    total_elapsed = time.time() - t0_total
    print(f"\n Hasil Tuning (diurutkan WER)")
    print(df_tuning.to_string(index=False))

    best = df_tuning.iloc[0]
    print(f"\nParameter Terbaik: K={int(best['K Komponen'])}, "
          f"tol={best['Toleransi']:.0e}, iter={int(best['Maks Iterasi'])}, "
          f"WER={best['WER Corpus (%)']:.2f}%")
    print(f"  Total waktu tuning: {total_elapsed/60:.1f} menit")

    return df_tuning