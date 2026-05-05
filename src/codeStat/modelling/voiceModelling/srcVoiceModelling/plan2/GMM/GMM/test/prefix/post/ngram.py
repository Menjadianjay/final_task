from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math
# Normalisasi Lexicon
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.normalisasiLexicon as normLexicon
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.normalisasiLexicon import *

class NGramModel:
    # trigram Language Model dengan Kneser-Ney smoothing.

    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n        = n
        self.discount = discount
        self.ngram_count  : Dict[tuple, int] = defaultdict(int)
        self.context_count: Dict[tuple, int] = defaultdict(int)
        self.vocab        : set              = set()
        self.N_total      : int              = 0
        self._ctx_diversity: Dict[tuple, int] = {}

    def _tokenize(self, kalimat: str) -> List[str]:
        tokens = normalisasi(kalimat).split()
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def fit(self, korpus: List[str]) -> "NGramModel":
        ctx_words: Dict[tuple, set] = defaultdict(set)
        for kalimat in korpus:
            tokens = self._tokenize(kalimat)
            self.vocab.update(tokens)
            self.N_total += len(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram   = tuple(tokens[i: i + self.n])
                ctx     = ngram[:-1]
                self.ngram_count[ngram] += 1
                self.context_count[ctx] += 1
                ctx_words[ctx].add(ngram[-1])
        self._ctx_diversity = {c: len(w) for c, w in ctx_words.items()}
        return self

    def log_prob(self, kata: str, context: tuple) -> float:
        ngram   = context + (kata,)
        C_ng    = self.ngram_count.get(ngram, 0)
        C_ctx   = self.context_count.get(context, 0)

        if C_ctx == 0:
            C_uni = self.ngram_count.get((kata,), 0)
            return math.log((C_uni + 1) / (self.N_total + len(self.vocab)) + 1e-300)

        lam    = self.discount * self._ctx_diversity.get(context, 0) / C_ctx
        C_uni  = self.ngram_count.get((kata,), 0)
        p_back = (C_uni + 1) / (self.N_total + len(self.vocab))
        prob   = max(C_ng - self.discount, 0) / C_ctx + lam * p_back
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
