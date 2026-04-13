# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 9 — HYPERPARAMETER TUNING (EFISIEN)
# ══════════════════════════════════════════════════════════════════════════════

def hyperparameter_tuning(
    frame_per_kata : Dict[str, List[np.ndarray]],
    cache_mfcc     : Dict[str, np.ndarray],
    dict_transkrip : Dict[str, str],
    ngram_model    : NGramModel,s
    dir_save       : str = ".",
) -> pd.DataFrame:
    grid = [
        (K, tol, mi)
        for K  in [4, 8, 16, 32]
        for tol in [1e-3, 1e-4]
        for mi in [50, 100, 200]
    ]
    n_eksp = len(grid)
    print(f"\n  Grid: {n_eksp} eksperimen (tanpa re-extract MFCC)")

    hasil_tuning = []
    t0_total     = time.time()

    for idx, (K, tol, mi) in enumerate(grid, 1):
        t0 = time.time()
        print(f"\n  [{idx}/{n_eksp}] K={K}, tol={tol:.0e}, max_iter={mi}")

        # Hanya re-train GMM
        gmm_pk = latih_gmm_per_kata(
            frame_per_kata, n_components=K, n_iter=mi, tol=tol, verbose=False
        )

        # Hanya re-decode dari cache
        hw = dekode_semua_audio(
            cache_mfcc, dict_transkrip, gmm_pk, ngram_model, verbose=False
        )

        if hw:
            df_tmp     = pd.DataFrame(hw)
            errs       = df_tmp["S"].sum() + df_tmp["D"].sum() + df_tmp["I"].sum()
            total_N    = df_tmp["N"].sum()
            wer_corpus = errs / total_N * 100 if total_N > 0 else 100.0
        else:
            wer_corpus = 100.0

        ela = time.time() - t0
        print(f"         WER={wer_corpus:.2f}%  ({ela:.1f}s)")

        hasil_tuning.append({
            "Eksperimen"    : idx,
            "K Komponen"    : K,
            "Toleransi"     : tol,
            "Maks Iterasi"  : mi,
            "WER Corpus (%)": round(wer_corpus, 2),
            "Waktu (s)"     : round(ela, 1),
        })

    df_tuning = pd.DataFrame(hasil_tuning).sort_values("WER Corpus (%)")
    df_tuning.to_csv(f"{dir_save}/HasilHyperparameterTuning_GMM.csv", index=False)

    best = df_tuning.iloc[0]
    print(f"\n  ✓ Parameter Terbaik:")
    print(f"    K={int(best['K Komponen'])}, "
          f"tol={best['Toleransi']:.0e}, "
          f"iter={int(best['Maks Iterasi'])}, "
          f"WER={best['WER Corpus (%)']:.2f}%")
    print(f"  Total waktu tuning: {(time.time()-t0_total)/60:.1f} menit")
    return df_tuning