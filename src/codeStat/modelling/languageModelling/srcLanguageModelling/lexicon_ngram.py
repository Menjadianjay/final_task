import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple

#  pemetaan huruf ke fonem Bahasa Jawa
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
    teks = teks.lower().strip()
    teks = re.sub(r"[^a-z\s]", "", teks)
    return re.sub(r"\s+", " ", teks).strip()


def kata_ke_fonem(kata: str) -> List[str]:
    fonem, i = [], 0
    while i < len(kata):
        if i + 1 < len(kata) and kata[i: i + 2] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i: i + 2]])
            i += 2
        elif kata[i] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i]])
            i += 1
        else:
            i += 1  
    return fonem if fonem else [kata]


def buat_lexicon(transkrip_list: List[str]) -> Dict[str, List[str]]:
    lexicon: Dict[str, List[str]] = {}
    for kalimat in transkrip_list:
        for kata in normalisasi(kalimat).split():
            if kata not in lexicon:
                lexicon[kata] = kata_ke_fonem(kata)
    return lexicon



#Trigram dengan Kneser-Ney Smoothing 

class NGramModel:
    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n        = n
        self.discount = discount

        self.ngram_count  : Dict[tuple, int] = defaultdict(int)
        self.context_count: Dict[tuple, int] = defaultdict(int)
        self.vocab        : set               = set()
        self.N_total      : int               = 0
        
        self._context_diversity: Dict[tuple, int] = {}

    def _tokenize(self, kalimat: str) -> List[str]:
        tokens = normalisasi(kalimat).split()
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def fit(self, korpus: List[str]) -> "NGramModel":
        for kalimat in korpus:
            tokens = self._tokenize(kalimat)
            self.vocab.update(tokens)
            self.N_total += len(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram   = tuple(tokens[i: i + self.n])
                context = ngram[:-1]
                self.ngram_count[ngram]     += 1
                self.context_count[context] += 1

        ctx_words: Dict[tuple, set] = defaultdict(set)
        for ngram in self.ngram_count:
            ctx_words[ngram[:-1]].add(ngram[-1])
        self._context_diversity = {
            ctx: len(words) for ctx, words in ctx_words.items()
        }

        return self

    def log_prob(self, kata: str, context: tuple) -> float:
        ngram   = context + (kata,)
        C_ngram = self.ngram_count.get(ngram, 0)
        C_ctx   = self.context_count.get(context, 0)

        # Jika konteks belum pernah muncul embali ke unigram
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
        tokens = ["<s>"] * (self.n - 1) + kata_list + ["</s>"]
        log_p  = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1: i])
            log_p  += self.log_prob(tokens[i], context)
        return log_p

    def rescore(self, kandidat: List[List[str]]) -> List[str]:
        if not kandidat:
            return []
        return max(kandidat, key=self.score_sequence)


#  WER = (S + D + I) / N

def compute_wer(referensi: List[str], hipotesis: List[str]) -> Dict:
    R, H = len(referensi), len(hipotesis)

    prev = list(range(H + 1))
    for i in range(1, R + 1):
        curr    = [i] + [0] * H
        for j in range(1, H + 1):
            if referensi[i - 1] == hipotesis[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr

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