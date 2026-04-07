"""
lexicon_ngram.py
================
Lexicon, N-Gram Language Model (Kneser-Ney), dan Word Error Rate.

Sesuai sub-bab skripsi:
  3.3.6   Lexicon   — pemetaan huruf ke fonem, digraf diprioritaskan
  3.3.7   Language Model — Trigram + Kneser-Ney smoothing (formula 2.20)
  3.3.9   WER — formula 3.3.9.1: WER = (S+D+I)/N
"""

import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — NORMALISASI & LEXICON
# Sub-bab 3.3.6 — pemetaan huruf ke fonem Bahasa Jawa
# ══════════════════════════════════════════════════════════════════════════════

# Digraf harus diletakkan sebelum monograf agar diprioritaskan
PETA_FONEM: Dict[str, List[str]] = {
    # Digraf khas Bahasa Jawa
    "ng": ["ng"],
    "ny": ["ny"],
    "th": ["th"],
    "dh": ["dh"],
    # Monograf
    "a": ["a"], "b": ["b"], "c": ["ch"], "d": ["d"],
    "e": ["e"], "f": ["f"], "g": ["g"],  "h": ["h"],
    "i": ["i"], "j": ["j"], "k": ["k"],  "l": ["l"],
    "m": ["m"], "n": ["n"], "o": ["o"],  "p": ["p"],
    "r": ["r"], "s": ["s"], "t": ["t"],  "u": ["u"],
    "v": ["v"], "w": ["w"], "y": ["y"],  "z": ["z"],
}


def normalisasi(teks: str) -> str:
    """Lowercase, hapus karakter non-huruf, normalisasi spasi."""
    teks = teks.lower().strip()
    teks = re.sub(r"[^a-z\s]", "", teks)
    return re.sub(r"\s+", " ", teks).strip()


def kata_ke_fonem(kata: str) -> List[str]:
    """
    Konversi kata ke urutan fonem. Digraf (ng, ny, th, dh) diprioritaskan.

    Contoh: "ngomong" → ["ng", "o", "m", "o", "ng"]
    """
    fonem, i = [], 0
    while i < len(kata):
        if i + 1 < len(kata) and kata[i: i + 2] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i: i + 2]])
            i += 2
        elif kata[i] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i]])
            i += 1
        else:
            i += 1   # skip karakter tidak dikenal
    return fonem if fonem else [kata]


def buat_lexicon(transkrip_list: List[str]) -> Dict[str, List[str]]:
    """
    Bangun kamus fonem {kata: [fonem]} dari semua transkrip.

    Args:
        transkrip_list : list kalimat transkrip

    Returns:
        {kata: [fonem]}
    """
    lexicon: Dict[str, List[str]] = {}
    for kalimat in transkrip_list:
        for kata in normalisasi(kalimat).split():
            if kata not in lexicon:
                lexicon[kata] = kata_ke_fonem(kata)
    return lexicon


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — N-GRAM LANGUAGE MODEL
# Sub-bab 3.3.7 — Trigram dengan Kneser-Ney Smoothing (formula 2.20)
# ══════════════════════════════════════════════════════════════════════════════

class NGramModel:
    """
    N-Gram Language Model dengan Kneser-Ney Smoothing.

    Parameters
    ----------
    n : int
        Orde N-gram (default 3 = trigram, sesuai sub-bab 3.3.7)
    discount : float
        Discount Kneser-Ney (0 < d < 1, default 0.75)
    """

    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n        = n
        self.discount = discount

        self.ngram_count  : Dict[tuple, int] = defaultdict(int)
        self.context_count: Dict[tuple, int] = defaultdict(int)
        self.vocab        : set               = set()
        self.N_total      : int               = 0

        # Cache agar tidak dihitung ulang tiap rescore
        self._context_diversity: Dict[tuple, int] = {}  # |{w: C(ctx,w)>0}|

    def _tokenize(self, kalimat: str) -> List[str]:
        """Tambah token <s> dan </s> untuk konteks awal/akhir."""
        tokens = normalisasi(kalimat).split()
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def fit(self, korpus: List[str]) -> "NGramModel":
        """
        Latih N-gram dari daftar kalimat.

        Args:
            korpus : list kalimat (teks sudah di-normalisasi atau belum)

        Returns:
            self
        """
        for kalimat in korpus:
            tokens = self._tokenize(kalimat)
            self.vocab.update(tokens)
            self.N_total += len(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram   = tuple(tokens[i: i + self.n])
                context = ngram[:-1]
                self.ngram_count[ngram]     += 1
                self.context_count[context] += 1

        # Hitung context diversity: |{w : C(ctx, w) > 0}| untuk tiap context
        ctx_words: Dict[tuple, set] = defaultdict(set)
        for ngram in self.ngram_count:
            ctx_words[ngram[:-1]].add(ngram[-1])
        self._context_diversity = {
            ctx: len(words) for ctx, words in ctx_words.items()
        }

        return self

    def log_prob(self, kata: str, context: tuple) -> float:
        """
        Hitung log P(kata | context) dengan Kneser-Ney smoothing.

        Args:
            kata    : kata yang ingin dihitung probabilitasnya
            context : tuple konteks (panjang n-1)

        Returns:
            float — log-probabilitas
        """
        ngram   = context + (kata,)
        C_ngram = self.ngram_count.get(ngram, 0)
        C_ctx   = self.context_count.get(context, 0)

        # Jika konteks belum pernah muncul → fallback ke unigram
        if C_ctx == 0:
            C_uni = self.ngram_count.get((kata,), 0)
            prob  = (C_uni + 1) / (self.N_total + len(self.vocab))
            return math.log(prob + 1e-300)

        # Kneser-Ney: P_KN = max(C-d, 0)/C(ctx) + λ(ctx) × P_back
        diversity = self._context_diversity.get(context, 0)
        lam       = (self.discount * diversity) / C_ctx

        # Back-off ke unigram
        C_uni  = self.ngram_count.get((kata,), 0)
        p_back = (C_uni + 1) / (self.N_total + len(self.vocab))

        prob = max(C_ngram - self.discount, 0) / C_ctx + lam * p_back
        return math.log(prob + 1e-300)

    def score_sequence(self, kata_list: List[str]) -> float:
        """
        Hitung total log-probability urutan kata.

        Args:
            kata_list : list kata hipotesis

        Returns:
            float — log-probability keseluruhan urutan
        """
        tokens = ["<s>"] * (self.n - 1) + kata_list + ["</s>"]
        log_p  = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1: i])
            log_p  += self.log_prob(tokens[i], context)
        return log_p

    def rescore(self, kandidat: List[List[str]]) -> List[str]:
        """
        Pilih urutan kata terbaik dari beberapa kandidat berdasarkan N-gram.

        Args:
            kandidat : list dari list kata

        Returns:
            list kata dengan skor N-gram tertinggi
        """
        if not kandidat:
            return []
        return max(kandidat, key=self.score_sequence)


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 3 — WORD ERROR RATE
# Formula 3.3.9.1: WER = (S + D + I) / N
# ══════════════════════════════════════════════════════════════════════════════

def compute_wer(referensi: List[str], hipotesis: List[str]) -> Dict:
    """
    Hitung Word Error Rate menggunakan dynamic programming (Levenshtein).

    Formula 3.3.9.1: WER = (S + D + I) / N
      S = Substitution (salah kata)
      D = Deletion     (kata hilang)
      I = Insertion    (kata tambahan)
      N = Jumlah kata referensi

    Args:
        referensi : list kata ground-truth
        hipotesis : list kata hasil dekoding

    Returns:
        dict dengan kunci: wer_persen, S, D, I, N
    """
    R, H = len(referensi), len(hipotesis)

    # ── Dynamic programming ────────────────────────────────────────────────
    # Gunakan 1D array rolling untuk hemat memori
    prev = list(range(H + 1))
    for i in range(1, R + 1):
        curr    = [i] + [0] * H
        for j in range(1, H + 1):
            if referensi[i - 1] == hipotesis[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr

    # ── Traceback untuk hitung S, D, I ────────────────────────────────────
    # Buat kembali matriks penuh untuk traceback (lebih mudah)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1): dp[i][0] = i
    for j in range(H + 1): dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if referensi[i - 1] == hipotesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    S = D = I = 0
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and referensi[i - 1] == hipotesis[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    N   = R
    wer = (S + D + I) / N if N > 0 else 0.0
    return {"wer_persen": round(wer * 100, 2), "S": S, "D": D, "I": I, "N": N}