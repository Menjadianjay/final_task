def compute_wer(reference: List[str], hypothesis: List[str]):
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
                dp[i][j] = 1 + min(dp[i-1][j-1],     # Substitusi (S)
                                    dp[i-1][j],      # deletion (D)
                                    dp[i][j-1])      # insertion  (I)

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
