from codeStat.modelimports.modelFunction import *

def compute_wer(referensi: List[str], hipotesis: List[str]) -> Dict:
    R, H = len(referensi), len(hipotesis)

    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if referensi[i-1] == hipotesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    S = D = I = 0
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and referensi[i-1] == hipotesis[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    N   = R
    wer = (S + D + I) / N if N > 0 else 0.0
    return {"wer_persen": round(wer * 100, 2), "S": S, "D": D, "I": I, "N": N}
# Bagian Decode dan statistic WER

def dekode_audio(
    fitur        : np.ndarray,
    kata_ref_list: List[str],
    gmm_per_kata : Dict[str, GaussianMixtureModel],
    ngram_model  : NGramModel,
) -> List[str]:
    if not gmm_per_kata or not kata_ref_list:
        return []

    pecahan   = pecah_frame_per_kata(fitur, kata_ref_list)
    vocab_gmm = list(gmm_per_kata.keys())
    hasil     : List[str] = []

    for kata_seg in kata_ref_list:
        seg = pecahan.get(kata_seg, None)
        if seg is None or seg.shape[0] == 0:
            hasil.append(vocab_gmm[0] if vocab_gmm else kata_seg)
            continue

        # Hitung skor semua GMM kata dan pilih terbaik
        skor    = {k: gmm_per_kata[k].score(seg) for k in vocab_gmm}
        best    = max(skor, key=skor.get)
        hasil.append(best)

    # N-gram rescore — HANYA hipotesis GMM, bukan referensi
    return ngram_model.rescore([hasil])


def dekode_semua_audio(
    cache_mfcc    : Dict[str, np.ndarray],
    dict_transkrip: Dict[str, str],
    gmm_per_kata  : Dict[str, GaussianMixtureModel],
    ngram_model   : NGramModel,
    verbose       : bool = True,
) -> List[Dict]:
    """Dekode semua audio dari cache + hitung WER."""
    hasil_wer = []
    n_total   = len(cache_mfcc)
    t0        = time.time()

    if verbose:
        print(f"  Dekode {n_total} audio ...")

    for idx, (audio_id, fitur) in enumerate(cache_mfcc.items(), 1):
        try:
            label = dict_transkrip.get(audio_id, "")
            if not label:
                continue
            ref_kata = normalisasi(label).split()
            if not ref_kata:
                continue

            hyp_kata  = dekode_audio(fitur, ref_kata, gmm_per_kata, ngram_model)
            wer_hasil = compute_wer(ref_kata, hyp_kata)

            hasil_wer.append({
                "Id"       : audio_id,
                "Referensi": " ".join(ref_kata),
                "Hipotesis": " ".join(hyp_kata),
                "WER (%)"  : wer_hasil["wer_persen"],
                "S"        : wer_hasil["S"],
                "D"        : wer_hasil["D"],
                "I"        : wer_hasil["I"],
                "N"        : wer_hasil["N"],
            })

            if verbose and idx % 100 == 0:
                print(f"    [{idx}/{n_total}] elapsed={time.time()-t0:.1f}s")

        except Exception as e:
            if verbose:
                print(f"    [ERROR] {audio_id}: {e}")

    if verbose:
        print(f"  Selesai: {len(hasil_wer)} audio dalam {time.time()-t0:.1f}s")
    return hasil_wer

def ringkasan_wer(df_wer: pd.DataFrame, label: str = "") -> Dict:
    if df_wer.empty:
        print("  [WARNING] Tidak ada data WER")
        return {}

    total_S    = int(df_wer["S"].sum())
    total_D    = int(df_wer["D"].sum())
    total_I    = int(df_wer["I"].sum())
    total_N    = int(df_wer["N"].sum())
    wer_corpus = (total_S + total_D + total_I) / total_N * 100 if total_N > 0 else 100.0

    print(f"\n── Ringkasan WER {label} ──────────────────────")
    print(f"  Jumlah audio  : {len(df_wer)}")
    print(f"  WER Corpus    : {wer_corpus:.2f}%")
    print(f"  WER Mean      : {df_wer['WER (%)'].mean():.2f}%")
    print(f"  WER Median    : {df_wer['WER (%)'].median():.2f}%")
    print(f"  WER Std       : {df_wer['WER (%)'].std():.2f}%")
    print(f"  WER Min       : {df_wer['WER (%)'].min():.2f}%")
    print(f"  WER Max       : {df_wer['WER (%)'].max():.2f}%")
    print(f"  Total S/D/I/N : {total_S}/{total_D}/{total_I}/{total_N}")
    print(f"{'─'*46}")

    return {
        "n_audio"   : len(df_wer),
        "wer_corpus": round(wer_corpus, 2),
        "wer_mean"  : round(df_wer["WER (%)"].mean(), 2),
        "wer_median": round(df_wer["WER (%)"].median(), 2),
        "wer_std"   : round(df_wer["WER (%)"].std(), 2),
        "wer_min"   : round(df_wer["WER (%)"].min(), 2),
        "wer_max"   : round(df_wer["WER (%)"].max(), 2),
        "total_S"   : total_S, "total_D": total_D,
        "total_I"   : total_I, "total_N": total_N,
    }