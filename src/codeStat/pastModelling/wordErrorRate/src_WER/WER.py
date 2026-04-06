from typing import Dict, List, Optional, Tuple


def compute_wer(reference: List[str], hypothesis: List[str]) -> Dict:
    """
    Word Error Rate sesuai formula (2.24):
        WER = (S + D + I) / N

    Args:
        reference : daftar kata referensi (ground truth)
        hypothesis: daftar kata hasil sistem ASR

    Returns:
        dict: {wer, wer_persen, S, D, I, N}
    """
    R, H = len(reference), len(hypothesis)

    # Matriks edit distance
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1],   # Substitusi (S)
                                    dp[i-1][j],      # Penghapusan (D)
                                    dp[i][j-1])      # Penyisipan  (I)

    # Backtrack hitung S, D, I
    S = D = I = 0
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    N   = R
    wer = (S + D + I) / N if N > 0 else 0.0
    return {"wer": wer, "wer_persen": round(wer * 100, 2),
            "S": S, "D": D, "I": I, "N": N}

   
# ── Evaluasi WER ──────────────────────────────────────────────────────
sep = "=" * 60
if __name__ == "__main__":
    contoh_uji = [
        ("driji kang nutupi tabung banjur diculake",
         "driji kang nutupi tabung banjur dicukake"),
        ("chow yun fat lunga dhateng aberdeen",
         "chow yun fat lunga dhateng aberdeen"),
        ("aaron kwok lunga dhateng guangzhou",
         "aron kwok lungo dhateng guangzu"),
    ]
    total_S = total_D = total_I = total_N = 0
    for ref_str, hyp_str in contoh_uji:
        ref = ref_str.split()
        hyp = hyp_str.split()
        r   = compute_wer(ref, hyp)
        total_S += r["S"]; total_D += r["D"]
        total_I += r["I"]; total_N += r["N"]
        print(f"  Ref : {ref_str}")
        print(f"  Hyp : {hyp_str}")
        print(f"  WER : {r['wer_persen']}%  "
              f"(S={r['S']} D={r['D']} I={r['I']} N={r['N']})")
        print()

    wer_total = (total_S + total_D + total_I) / total_N * 100
    print(f"  WER Keseluruhan: {wer_total:.2f}%")

    print(f"\n{sep}")
    print("  Selesai.")
    print(sep)
