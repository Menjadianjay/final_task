from collections import defaultdict
from typing import Dict, List
import numpy as np
import time
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.model.gmmModel import GaussianMixtureModel
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.normalisasiLexicon import normalisasi, kata_ke_fonem 

# BAGIAN 7 — FRAME PER KATA & LATIH GMM PER KATA

def pecah_frame_per_kata(
    fitur    : np.ndarray,
    kata_list: List[str],cmcz
) -> Dict[str, np.ndarray]:
    """Bagi frame MFCC secara proporsional ke setiap kata."""
    T, n = fitur.shape[0], len(kata_list)
    if n == 0 or T == 0:
        return {}

    bobot = np.array([max(len(kata_ke_fonem(k)), 1) for k in kata_list], dtype=float)
    bobot /= bobot.sum()
    n_frame = (bobot * T).astype(int)
    sisa    = T - n_frame.sum()
    for i in range(sisa):
        n_frame[i % n] += 1

    hasil: Dict[str, np.ndarray] = {}
    idx = 0
    for i, kata in enumerate(kata_list):
        nf  = n_frame[i]
        seg = fitur[idx: idx + nf]
        if nf > 0:
            hasil[kata] = np.vstack([hasil[kata], seg]) if kata in hasil else seg
        idx += nf
    return hasil


def kumpulkan_frame_per_kata(
    cache_mfcc    : Dict[str, np.ndarray],
    dict_transkrip: Dict[str, str],
) -> Dict[str, List[np.ndarray]]:
    """Kumpulkan frame per kata dari cache (tidak baca file)."""
    frame_per_kata: Dict[str, List[np.ndarray]] = defaultdict(list)

    print(f"  Mengumpulkan frame per kata dari {len(cache_mfcc)} audio ...")
    for audio_id, fitur in cache_mfcc.items():
        label = dict_transkrip.get(audio_id, "")
        if not label:
            continue
        kata_list = normalisasi(label).split()
        for kata, frame in pecah_frame_per_kata(fitur, kata_list).items():
            frame_per_kata[kata].append(frame)

    print(f"  Kata unik dengan frame: {len(frame_per_kata)}")
    return frame_per_kata


def latih_gmm_per_kata(
    frame_per_kata: Dict[str, List[np.ndarray]],
    n_components  : int   = 4,
    n_iter        : int   = 100,
    tol           : float = 1e-3,
    min_frame     : int   = 10,
    verbose       : bool  = True,
) -> Dict[str, GaussianMixtureModel]:
    """Latih satu GMM per kata."""
    gmm_per_kata: Dict[str, GaussianMixtureModel] = {}
    kata_list    = sorted(frame_per_kata.keys())
    n_kata       = len(kata_list)

    if verbose:
        print(f"  Melatih GMM untuk {n_kata} kata (K={n_components}) ...")
    t0 = time.time()

    for i, kata in enumerate(kata_list):
        X = np.vstack(frame_per_kata[kata])
        if X.shape[0] < min_frame:
            continue

        gmm = GaussianMixtureModel(
            n_components = min(n_components, X.shape[0]),
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