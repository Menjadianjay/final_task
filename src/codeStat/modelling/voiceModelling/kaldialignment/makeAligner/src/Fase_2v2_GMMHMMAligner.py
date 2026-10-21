import os, re, glob, math, time, pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as TAT
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

try:
    import cupy as cp
    _CP = cp.cuda.is_available()
except ImportError:
    cp, _CP = None, False

xp      = cp if _CP else np
TD      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PT_GPU = TD.type == "cuda"


# ── Paths & Config ────────────────────────────────────────────────────────────

AUDIO_DIR     = r"D:\118_Rio\118\datasampled"
CSV_PATH      = r"D:\118_Rio\118\src\codeStat\preModelling\out\databersih.csv"
ALIGNMENT_CSV = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\makeAligner\sampleddata\alignment_map.csv"
OUT_DIR       = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\makeAligner\sampleddata"

N_STATES, N_MIX, N_ITER_GMM = 3, 32, 100
N_ITER_VIT, N_LDA_DIM       = 10, 29
TOL, MIN_FRAMES              = 1e-4, 3
TEST_SIZE, SEED, P_SELF      = 0.2, 42, 0.7
EM_BATCH, FREE_EVERY         = 50_000, 100
SR, N_MFCC, HOP, WIN, N_MELS, PRE_EMPH = 16_000, 13, 160, 400, 26, 0.97

K_TABLE = [(50,2),(500,4),(2000,8),(10000,16),(float("inf"),32)]

PHONEME_MAP: Dict[str,str] = {
    "ng":"ng","ny":"ny","th":"th","dh":"dh",
    "a":"a","b":"b","c":"ch","d":"d","e":"e","f":"f","g":"g","h":"h",
    "i":"i","j":"j","k":"k","l":"l","m":"m","n":"n","o":"o","p":"p",
    "r":"r","s":"s","t":"t","u":"u","v":"v","w":"w","y":"y","z":"z",
    "x":"k","q":"k",
}

KAMUS_RUSAK = {
    "Ã©":"é","Ã¨":"è","Ã ":"à","Ã¢":"â","Ã®":"î","Ã´":"ô",
    "Ã»":"û","Ã«":"ë","Ã¯":"ï","Ã¼":"ü","Ã±":"ñ","Ã§":"ç",
}
_ACCENT = str.maketrans({
    "é":"e","è":"e","ê":"e","ë":"e","à":"a","á":"a","â":"a","ä":"a",
    "î":"i","ï":"i","í":"i","ì":"i","ô":"o","ó":"o","ò":"o","ö":"o",
    "û":"u","ú":"u","ù":"u","ü":"u","ñ":"n","ç":"c",
})


# ── GPU Helpers ───────────────────────────────────────────────────────────────

def to_gpu(a):
    return cp.asarray(a) if _CP else np.asarray(a)

def to_cpu(a) -> np.ndarray:
    return cp.asnumpy(a) if (_CP and isinstance(a, cp.ndarray)) else np.asarray(a)

def free_gpu():
    if _CP: cp.get_default_memory_pool().free_all_blocks()
    if _PT_GPU: torch.cuda.empty_cache()


# ── Teks ──────────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    for bad, good in KAMUS_RUSAK.items(): text = text.replace(bad, good)
    text = text.lower().strip()
    text = re.sub(r'["\'`\u201c\u201d\u2018\u2019()\[\]{}]', " ", text)
    text = re.sub(r"[+\-*/=<>%^&|~\\@#$.,;:!?]", " ", text)
    text = text.translate(_ACCENT)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def word_to_phonemes(word: str) -> List[str]:
    word = clean(word); result, i = [], 0
    while i < len(word):
        if i+1 < len(word) and word[i:i+2] in PHONEME_MAP:
            result.append(PHONEME_MAP[word[i:i+2]]); i += 2
        elif word[i] in PHONEME_MAP:
            result.append(PHONEME_MAP[word[i]]); i += 1
        else: i += 1
    return result or list(word)

def sentence_to_phonemes(sentence: str) -> List[str]:
    out = []
    for word in clean(sentence).split(): out.extend(word_to_phonemes(word))
    return out


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_transcripts(csv_path: str) -> Dict[str,str]:
    df = pd.read_csv(csv_path)
    return {os.path.splitext(os.path.basename(str(r["Path_Data"])))[0]:
            str(r["sentences"]).lower().strip() for _, r in df.iterrows()}


# ── Ekstraksi MFCC ────────────────────────────────────────────────────────────

_mfcc_tf = TAT.MFCC(SR, N_MFCC, melkwargs={
    "n_fft":512,"n_mels":N_MELS,"win_length":WIN,
    "hop_length":HOP,"f_min":0.0,"f_max":4000.0}).to(TD)
_delta_tf = TAT.ComputeDeltas().to(TD)

def _pre_emph(wav: torch.Tensor) -> torch.Tensor:
    return torch.cat([wav[:,:1], wav[:,1:] - PRE_EMPH*wav[:,:-1]], dim=1)

def extract_mfcc(wav_path: str) -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[1] == 0: raise ValueError(f"Audio kosong: {wav_path}")
    if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
    wav    = _pre_emph(wav.mean(0, keepdim=True).to(TD))
    mfcc   = _mfcc_tf(wav)
    d1 = _delta_tf(mfcc)
    d2 = _delta_tf(d1)
    return torch.cat([mfcc,d1,d2],dim=1).squeeze(0).T.cpu().numpy().astype(np.float32)

def build_cache(audio_dir: str, transcripts: Dict[str,str],
                cache_path: str = "") -> Tuple[Dict[str,np.ndarray], List[Dict]]:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path,"rb") as f: cache, meta = pickle.load(f)
        print(f"  Cache dimuat: {len(cache):,} audio"); return cache, meta

    wav_files  = glob.glob(os.path.join(audio_dir,"**/*.wav"),  recursive=True)
    wav_files += glob.glob(os.path.join(audio_dir,"**/*.flac"), recursive=True)
    cache: Dict[str,np.ndarray] = {}; meta: List[Dict] = []
    n_skip = 0; t0 = time.time()
    print(f"  Ekstraksi MFCC ({'GPU' if _PT_GPU else 'CPU'}) dari {len(wav_files):,} file ...")

    for i, path in enumerate(wav_files, 1):
        aid = os.path.splitext(os.path.basename(path))[0]
        if aid not in transcripts: continue
        try:
            feat = extract_mfcc(path)
            cache[aid] = feat
            meta.append({"Id":aid,"Label":transcripts[aid],
                         "Frames":feat.shape[0],"Dims":feat.shape[1]})
        except Exception as e:
            n_skip += 1; print(f"  [skip] {os.path.basename(path)}: {e}")
        if i % 500 == 0:
            ela = time.time()-t0; eta = ela/i*(len(wav_files)-i)
            print(f"  {i:,}/{len(wav_files):,}  ok={len(cache):,}  skip={n_skip}  ela={ela:.0f}s  ETA={eta:.0f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path,"wb") as f: pickle.dump((cache,meta),f,protocol=4)
        print(f"  Cache disimpan ({os.path.getsize(cache_path)/1024**2:.0f} MB)")
    print(f"  Selesai: {len(cache):,} ok, {n_skip} skip, {time.time()-t0:.0f}s")
    return cache, meta


# ── LDA ───────────────────────────────────────────────────────────────────────

_lda_model: Optional[LDA] = None

def _segment_frames(cache, transcripts, alignment, ids):
    for aid in ids:
        feat = cache.get(aid); label = transcripts.get(aid,"")
        if feat is None or not label: continue
        if alignment and aid in alignment._index:
            for word, seg, _, _ in alignment.get_segments(aid, feat):
                phones = word_to_phonemes(word)
                if not phones or seg.shape[0] < MIN_FRAMES: continue
                step = max(1, seg.shape[0]//len(phones))
                for fi, ph in enumerate(phones):
                    end   = fi*step+step if fi < len(phones)-1 else seg.shape[0]
                    chunk = seg[fi*step:end]
                    if chunk.shape[0] >= MIN_FRAMES: yield ph, chunk
        else:
            phones = sentence_to_phonemes(label)
            T, F   = feat.shape[0], len(phones)
            step   = max(1, T//F)
            for fi, ph in enumerate(phones):
                end   = fi*step+step if fi < F-1 else T
                chunk = feat[fi*step:end]
                if chunk.shape[0] >= MIN_FRAMES: yield ph, chunk

def train_lda(cache, transcripts, alignment, train_ids, lda_path="") -> LDA:
    global _lda_model
    if lda_path and os.path.exists(lda_path):
        with open(lda_path,"rb") as f: _lda_model = pickle.load(f)
        print(f"  LDA dimuat: {_lda_model.n_components} dims"); return _lda_model

    X_list, y_list = [], []
    for ph, chunk in _segment_frames(cache, transcripts, alignment, train_ids):
        X_list.append(chunk); y_list.extend([ph]*chunk.shape[0])

    X = np.vstack(X_list); y = np.array(y_list)
    n_comp = min(N_LDA_DIM, len(set(y))-1, X.shape[1])
    print(f"  LDA: {X.shape[0]:,} frame, {len(set(y))} kelas → {n_comp} dim")
    _lda_model = LDA(n_components=n_comp)
    _lda_model.fit(X, y)
    print(f"  Explained variance: {_lda_model.explained_variance_ratio_.sum():.3f}")
    if lda_path:
        os.makedirs(os.path.dirname(lda_path) or ".", exist_ok=True)
        with open(lda_path,"wb") as f: pickle.dump(_lda_model,f,protocol=4)
    return _lda_model

def apply_lda(cache: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    if _lda_model is None: return cache
    if _CP:
        W    = to_gpu(_lda_model.scalings_[:,:_lda_model.n_components].astype(np.float32))
        xbar = to_gpu(_lda_model.xbar_.astype(np.float32))
        return {k: to_cpu((to_gpu(v)-xbar)@W).astype(np.float32) for k,v in cache.items()}
    return {k: _lda_model.transform(v).astype(np.float32) for k,v in cache.items()}


# ── Alignment Loader ──────────────────────────────────────────────────────────

class AlignmentLoader:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        self._index = {aid: grp.reset_index(drop=True)
                       for aid, grp in df.groupby("audio_id")}
        print(f"  Alignment: {len(df):,} interval, "
              f"{df['audio_id'].nunique():,} audio, {df['kata'].nunique():,} kata")

    def get_segments(self, aid, feat):
        if aid not in self._index: return []
        T, out = feat.shape[0], []
        for _, row in self._index[aid].iterrows():
            fs  = max(0, min(int(row["frame_start"]), T-1))
            fe  = max(fs+1, min(int(row["frame_end"]), T))
            seg = feat[fs:fe]
            if seg.shape[0] > 0: out.append((str(row["kata"]),seg,fs,fe))
        return out


# ── GMM ───────────────────────────────────────────────────────────────────────

def adaptive_k(n: int) -> int:
    for thr, k in K_TABLE:
        if n < thr: return min(k, N_MIX)
    return N_MIX

class GMM:
    def __init__(self, k=4, n_iter=N_ITER_GMM, tol=TOL, seed=None):
        self.k, self.n_iter, self.tol, self.seed = k, n_iter, tol, seed
        self.w = self.mu = self.var = None
        self.converged = False

    def _kmeans_pp(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed); T = X.shape[0]
        idx = [int(rng.integers(0,T))]
        for _ in range(1, self.k):
            chosen = X[idx]
            dist   = ((X[:,None,:]-chosen[None,:,:])**2).sum(axis=2).min(axis=1)
            p = dist.astype(np.float64); s = p.sum()
            p = p/s if s > 0 else np.full(T, 1.0/T)
            idx.append(int(rng.choice(T, p=p)))
        self.mu  = X[idx].astype(np.float32)
        self.w   = np.full(self.k, 1.0/self.k, dtype=np.float32)
        self.var = np.tile(X.var(axis=0).astype(np.float32)+1e-6, (self.k,1))

    def _log_prob(self, X_g):
        mu_g, var_g, w_g = to_gpu(self.mu), to_gpu(self.var), to_gpu(self.w)
        D = X_g.shape[1]; diff = X_g[:,None,:] - mu_g[None,:,:]
        log_det = xp.sum(xp.log(var_g+1e-12), axis=1)
        maha    = xp.sum(diff**2/(var_g[None]+1e-12), axis=2)
        return xp.log(w_g+1e-12)[None,:] - 0.5*(D*math.log(2*math.pi)+log_det[None,:]+maha)

    def _logsumexp(self, lp):
        lse = xp.max(lp, axis=1, keepdims=True)
        return lse.squeeze(1) + xp.log(xp.sum(xp.exp(lp-lse), axis=1)+1e-12)

    def _em_step(self, X: np.ndarray) -> float:
        T, D    = X.shape
        Nk_tot  = xp.zeros(self.k, dtype=xp.float64)
        sx_tot  = xp.zeros((self.k,D), dtype=xp.float64)
        sxx_tot = xp.zeros((self.k,D), dtype=xp.float64)
        ll_sum  = 0.0
        for start in range(0, T, EM_BATCH):
            X_g  = to_gpu(X[start:start+EM_BATCH])
            lp   = self._log_prob(X_g); log_sum = self._logsumexp(lp)
            resp = xp.exp(lp-log_sum[:,None]); r64 = resp.astype(xp.float64); X64 = X_g.astype(xp.float64)
            Nk_tot += r64.sum(axis=0); sx_tot += r64.T@X64; sxx_tot += r64.T@(X64**2)
            ll_sum += float(to_cpu(log_sum.astype(xp.float64)).sum())
            del X_g; free_gpu()

        Nk  = Nk_tot+1e-10
        w_c = to_cpu((Nk_tot/T).astype(xp.float32))
        m_c = to_cpu((sx_tot/Nk[:,None]).astype(xp.float32))
        v_c = to_cpu((sxx_tot/Nk[:,None]-(sx_tot/Nk[:,None])**2).astype(xp.float32))+1e-6

        min_w = 1.0/(10*self.k); dead = np.where(w_c < min_w)[0]
        if dead.size:
            live = np.where(w_c >= min_w)[0]; rng = np.random.default_rng()
            for dk in dead:
                src = rng.choice(live) if live.size else 0
                m_c[dk] = X[rng.integers(0,T)]; v_c[dk] = v_c[src]
                w_c[dk] = w_c[src]/2.0; w_c[src] /= 2.0
            w_c /= w_c.sum()
        self.w, self.mu, self.var = w_c, m_c, v_c; free_gpu()
        return ll_sum/T

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], EM_BATCH):
            X_g = to_gpu(X[s:s+EM_BATCH])
            out[s:s+EM_BATCH] = to_cpu(self._logsumexp(self._log_prob(X_g))).astype(np.float32)
            del X_g
        free_gpu(); return out

    def fit(self, X: np.ndarray) -> "GMM":
        self.k = min(self.k, X.shape[0]); self._kmeans_pp(X)
        prev = -np.inf
        for _ in range(self.n_iter):
            ll = self._em_step(X)
            if not np.isfinite(ll) or abs(ll-prev) < self.tol:
                self.converged = np.isfinite(ll); break
            prev = ll
        return self

    def partial_fit(self, X: np.ndarray) -> "GMM":
        if self.w is None: return self.fit(X)
        prev = -np.inf
        for _ in range(self.n_iter):
            ll = self._em_step(X)
            if not np.isfinite(ll) or abs(ll-prev) < self.tol: break
            prev = ll
        return self


# ── HMM ───────────────────────────────────────────────────────────────────────

class PhonemeHMM:
    def __init__(self, phoneme: str):
        self.phoneme   = phoneme
        self.gmms      = [GMM(seed=abs(hash(f"{phoneme}_{s}"))%100_000) for s in range(N_STATES)]
        ls, ln         = math.log(P_SELF+1e-12), math.log(1-P_SELF+1e-12)
        self.log_trans = np.full((N_STATES,N_STATES), -np.inf, np.float32)
        for s in range(N_STATES):
            self.log_trans[s,s] = ls
            if s+1 < N_STATES: self.log_trans[s,s+1] = ln
        self.log_init    = np.full(N_STATES, -np.inf, np.float32)
        self.log_init[0] = 0.0
        self.trained     = False

    def emission(self, X: np.ndarray) -> np.ndarray:
        B = np.full((X.shape[0],N_STATES), -np.inf, np.float32)
        for s, gmm in enumerate(self.gmms):
            if gmm.w is not None: B[:,s] = gmm.log_likelihood(X)
        return B

    def fit(self, X: np.ndarray):
        T = X.shape[0]; seg_len = max(1, T//N_STATES)
        buckets: Dict[int,List[np.ndarray]] = defaultdict(list)
        for t in range(T): buckets[min(t//seg_len, N_STATES-1)].append(X[t])
        for s, gmm in enumerate(self.gmms):
            if s in buckets and len(buckets[s]) >= 2:
                Xs = np.vstack(buckets[s]); gmm.k = adaptive_k(Xs.shape[0]); gmm.fit(Xs)
        self.trained = True


# ── Viterbi GPU ───────────────────────────────────────────────────────────────

def _viterbi_gpu(emit: torch.Tensor, trans: torch.Tensor,
                 log_init: torch.Tensor) -> Tuple[torch.Tensor, float]:
    T, S     = emit.shape
    delta    = log_init + emit[0]
    psi      = torch.zeros(T, S, dtype=torch.int32, device=TD)
    arange_S = torch.arange(S, device=TD)
    for t in range(1, T):
        scores    = delta.unsqueeze(1) + trans
        best_prev = scores.argmax(dim=0)
        delta     = scores[best_prev, arange_S] + emit[t]
        psi[t]    = best_prev
    path      = torch.zeros(T, dtype=torch.int32, device=TD)
    path[T-1] = delta.argmax()
    ll        = float(delta[path[T-1]].item())
    for t in range(T-2, -1, -1): path[t] = psi[t+1, path[t+1]]
    path_cpu = path.cpu()
    del psi, delta, path, scores, best_prev
    if _PT_GPU: torch.cuda.empty_cache()
    return path_cpu, ll


# ── GMM-HMM System ────────────────────────────────────────────────────────────

class GMMHMMSystem:
    def __init__(self):
        self.hmms: Dict[str,PhonemeHMM] = {}

    def _get(self, ph: str) -> PhonemeHMM:
        if ph not in self.hmms: self.hmms[ph] = PhonemeHMM(ph)
        return self.hmms[ph]

    def _concat_tensors(self, X: np.ndarray, phones: List[str]):
        T, S, F = X.shape[0], N_STATES, len(phones); S_tot = F*S
        emit    = np.full((T,S_tot), -np.inf, np.float32)
        for fi, ph in enumerate(phones):
            hmm = self._get(ph)
            if hmm.trained: emit[:,fi*S:(fi+1)*S] = hmm.emission(X)
        ls, ln = math.log(P_SELF+1e-12), math.log(1-P_SELF+1e-12)
        trans  = np.full((S_tot,S_tot), -np.inf, np.float32)
        for fi in range(F):
            for s in range(S):
                gs = fi*S+s; trans[gs,gs] = ls
                if s+1 < S: trans[gs,gs+1] = ln
                elif fi+1 < F: trans[gs,(fi+1)*S] = ln
        log_init    = np.full(S_tot, -np.inf, np.float32); log_init[0] = 0.0
        emit_t  = torch.from_numpy(emit).to(TD)
        trans_t = torch.from_numpy(trans).to(TD)
        init_t  = torch.from_numpy(log_init).to(TD)
        del emit, trans, log_init
        return emit_t, trans_t, init_t, S

    def viterbi(self, X: np.ndarray, phones: List[str]) -> Tuple[np.ndarray,np.ndarray,float]:
        emit_t, trans_t, init_t, S = self._concat_tensors(X, phones)
        path_cpu, ll = _viterbi_gpu(emit_t, trans_t, init_t)
        del emit_t, trans_t, init_t; free_gpu()
        path = path_cpu.numpy()
        return path, path//S, ll

    def collect_frames(self, cache, transcripts, alignment, train_ids):
        frames: Dict[str,List[np.ndarray]] = defaultdict(list)
        for ph, chunk in _segment_frames(cache, transcripts, alignment, train_ids):
            frames[ph].append(chunk)
        return dict(frames)

    def fit(self, frames: Dict[str,List[np.ndarray]]):
        print(f"  Melatih {len(frames)} fonem HMM ...")
        t0 = time.time()
        for i, (ph,segs) in enumerate(frames.items(), 1):
            self._get(ph).fit(np.vstack(segs)); free_gpu()
            if i%10 == 0: print(f"    [{i}/{len(frames)}] {ph}  {time.time()-t0:.0f}s")

    def reestimate(self, cache, transcripts, alignment, train_ids) -> float:
        buckets: Dict[str,Dict[int,List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        n_ok = n_fail = 0
        for i, aid in enumerate(train_ids):
            feat = cache.get(aid); label = transcripts.get(aid,"")
            if feat is None or not label: continue
            phones = [p for p in sentence_to_phonemes(clean(label)) if p in self.hmms]
            if not phones: continue
            try:
                state_path, phone_path, _ = self.viterbi(feat, phones)
            except Exception as e:
                n_fail += 1
                if n_fail <= 5: print(f"  [viterbi gagal] {aid}: {e}")
                continue
            F_len = len(phones)
            phone_arr = np.asarray(phone_path, dtype=np.int32)
            state_arr = np.asarray(state_path, dtype=np.int32)
            valid     = (phone_arr >= 0) & (phone_arr < F_len)
            for fi in range(F_len):
                mask = valid & (phone_arr == fi)
                if not mask.any(): continue
                ph    = phones[fi]
                local = np.clip(state_arr[mask]-fi*N_STATES, 0, N_STATES-1)
                frames_here = feat[mask]
                for s in range(N_STATES):
                    sel = frames_here[local==s]
                    if sel.shape[0]: buckets[ph][s].append(sel)
            n_ok += 1
            if (i+1)%FREE_EVERY == 0: free_gpu()
        free_gpu()
        if n_fail: print(f"  reestimasi: {n_ok} ok, {n_fail} gagal")
        for ph, state_dict in buckets.items():
            hmm = self.hmms.get(ph)
            if hmm is None: continue
            for s, gmm in enumerate(hmm.gmms):
                segs = state_dict.get(s)
                if segs and sum(x.shape[0] for x in segs) >= 2:
                    gmm.partial_fit(np.vstack(segs)); free_gpu()
        ll_sum = ll_cnt = 0
        for aid in train_ids[:100]:
            feat = cache.get(aid); label = transcripts.get(aid,"")
            if feat is None or not label: continue
            phones = [p for p in sentence_to_phonemes(clean(label)) if p in self.hmms]
            if not phones: continue
            try:
                _, _, ll = self.viterbi(feat, phones)
                if np.isfinite(ll): ll_sum += ll/max(feat.shape[0],1); ll_cnt += 1
            except Exception: pass
        free_gpu()
        return ll_sum/max(ll_cnt,1)

    def forced_align(self, aid, feat, label, alignment=None) -> List[Dict]:
        rows = []
        def append(phones, seg, frame_offset, word=""):
            if not phones: return
            state_path, ph_path, _ = self.viterbi(seg, phones)
            for t in range(seg.shape[0]):
                fi = int(ph_path[t])
                ph = phones[fi] if 0 <= fi < len(phones) else phones[-1]
                s  = max(0, min(int(state_path[t])-fi*N_STATES, N_STATES-1))
                rows.append({"audio_id":aid,"frame":frame_offset+t,
                             "phoneme":ph,"word":word,"state":s,"label_tdnn":f"{ph}_{s}"})
        if alignment and aid in alignment._index:
            for word, seg, fs_word, _ in alignment.get_segments(aid, feat):
                phones = [p for p in word_to_phonemes(word) if p in self.hmms]
                append(phones, seg, fs_word, word)
        else:
            phones = [p for p in sentence_to_phonemes(clean(label)) if p in self.hmms]
            append(phones, feat, 0)
        return rows

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f,protocol=4)
        print(f"  Model disimpan: {path} ({os.path.getsize(path)/1024**2:.1f} MB)")

    @staticmethod
    def load(path: str) -> "GMMHMMSystem":
        with open(path,"rb") as f: obj = pickle.load(f)
        print(f"  Model dimuat: {len(obj.hmms)} fonem"); return obj


# ── Build Alignment CSV ───────────────────────────────────────────────────────

def build_alignment(system, cache, transcripts, alignment,
                    ids, out_path) -> pd.DataFrame:
    rows, n_ok, n_skip = [], 0, 0
    for aid in ids:
        feat = cache.get(aid); label = transcripts.get(aid,"")
        if feat is None or not label: n_skip += 1; continue
        try:
            rows.extend(system.forced_align(aid, feat, label, alignment)); n_ok += 1
        except Exception as e:
            print(f"  [skip] {aid}: {e}"); n_skip += 1
        if n_ok%FREE_EVERY == 0: free_gpu()
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {n_ok} audio  {len(df):,} frame  {df['label_tdnn'].nunique()} kelas  → {out_path}")
    else:
        print(f"  PERINGATAN: alignment kosong ({n_skip} dilewati)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True); t0 = time.time()
    
    print(CSV_PATH)
    print(OUT_DIR)
    print(ALIGNMENT_CSV)
    print(AUDIO_DIR)

    print("\n[1] Muat transkripsi")
    transcripts = load_transcripts(CSV_PATH)
    print(f"  {len(transcripts):,} audio")

    print("\n[2] Ekstraksi MFCC")
    cache, meta = build_cache(AUDIO_DIR, transcripts, cache_path=f"{OUT_DIR}/mfcc_cache.pkl")
    pd.DataFrame(meta).to_csv(f"{OUT_DIR}/HasilMFCC.csv", index=False, encoding="utf-8")
    if not cache: print("  ERROR: tidak ada audio"); return

    print("\n[3] Pembagian train/test")
    all_ids             = list(cache.keys())
    train_ids, test_ids = train_test_split(all_ids, test_size=TEST_SIZE, random_state=SEED)
    pd.DataFrame(
        [{"audio_id":a,"split":"train"} for a in train_ids] +
        [{"audio_id":a,"split":"test"}  for a in test_ids]
    ).to_csv(f"{OUT_DIR}/HasilTrainTestSplit.csv", index=False, encoding="utf-8")
    print(f"  Train: {len(train_ids):,}  Test: {len(test_ids):,}")

    print("\n[4] Muat alignment MFA")
    alignment = None
    if os.path.exists(ALIGNMENT_CSV):
        try: alignment = AlignmentLoader(ALIGNMENT_CSV)
        except Exception as e: print(f"  PERINGATAN: {e}")
    else:
        print("  PERINGATAN: alignment_map.csv tidak ditemukan — fallback proporsional")

    print("\n[5] Latih LDA")
    train_lda(cache, transcripts, alignment, train_ids, lda_path=f"{OUT_DIR}/lda_model.pkl")
    cache = apply_lda(cache)
    print(f"  Dimensi fitur setelah LDA: {next(iter(cache.values())).shape[1]}")

    model_path = f"{OUT_DIR}/gmm_hmm_system.pkl"
    system: Optional[GMMHMMSystem] = None
    if os.path.exists(model_path):
        print("\n[6] Muat model GMM-HMM yang sudah ada")
        try: system = GMMHMMSystem.load(model_path)
        except Exception as e: print(f"  PERINGATAN: {e} — latih ulang")

    if system is None:
        system = GMMHMMSystem()
        print("\n[6] Kumpulkan frame per fonem")
        frames = system.collect_frames(cache, transcripts, alignment, train_ids)
        pd.DataFrame([{"phoneme":ph,"segments":len(s),
                       "total_frames":sum(x.shape[0] for x in s)}
                      for ph,s in frames.items()]
        ).to_csv(f"{OUT_DIR}/HasilFramePerFonem.csv", index=False, encoding="utf-8")
        print(f"  {len(frames)} fonem")

        print("\n[7] Inisialisasi GMM-HMM")
        system.fit(frames)

        print(f"\n[8] Viterbi re-estimasi ({N_ITER_VIT} iterasi)")
        prev_ll = -np.inf
        for it in range(N_ITER_VIT):
            ll = system.reestimate(cache, transcripts, alignment, train_ids)
            print(f"  iter {it+1}/{N_ITER_VIT}  avg LL/frame = {ll:.6f}")
            if not np.isfinite(ll): print("  PERINGATAN: LL tidak finite — berhenti"); break
            if abs(ll-prev_ll) < TOL: print(f"  Konvergen pada iterasi {it+1}"); break
            prev_ll = ll
        system.save(model_path)

    print("\n[9] Forced alignment — data train")
    df_train = build_alignment(system, cache, transcripts, alignment,
                               train_ids, f"{OUT_DIR}/alignment_tdnn_train.csv")

    print("\n[10] Forced alignment — data test")
    df_test  = build_alignment(system, cache, transcripts, alignment,
                               test_ids, f"{OUT_DIR}/alignment_tdnn_test.csv")

    df_all    = pd.concat([df_train, df_test], ignore_index=True)
    label_map = {lbl:i for i,lbl in enumerate(sorted(df_all["label_tdnn"].unique()))}
    pd.DataFrame([{"label_tdnn":k,"label_id":v} for k,v in label_map.items()]
    ).to_csv(f"{OUT_DIR}/label_map_tdnn.csv", index=False, encoding="utf-8")
    print(f"\n  label_map_tdnn.csv — {len(label_map)} kelas")

    print("\n[11] Buat HasilLexicon.csv")
    if alignment:
        df_align  = pd.read_csv(ALIGNMENT_CSV)
        kata_list = sorted([str(k) for k in df_align["kata"].dropna().unique() if str(k).strip()])
        lex_rows  = [{"Kata":clean(kata),"Fonem":" ".join(word_to_phonemes(clean(kata)))}
                     for kata in kata_list if word_to_phonemes(clean(kata))]
        pd.DataFrame(lex_rows).drop_duplicates("Kata")\
            .to_csv(f"{OUT_DIR}/HasilLexicon.csv", index=False, encoding="utf-8")
        print(f"  HasilLexicon.csv — {len(lex_rows)} kata")

    print(f"\n{'='*60}")
    print(f"  Selesai dalam {(time.time()-t0)/60:.1f} menit  |  Output: {OUT_DIR}")
    print(f"{'='*60}\n")
    for fname in ["HasilMFCC.csv","HasilTrainTestSplit.csv","HasilFramePerFonem.csv",
                  "lda_model.pkl","gmm_hmm_system.pkl","mfcc_cache.pkl",
                  "alignment_tdnn_train.csv","alignment_tdnn_test.csv",
                  "label_map_tdnn.csv","HasilLexicon.csv"]:
        tick = "✓" if os.path.isfile(f"{OUT_DIR}/{fname}") else "–"
        print(f"  {tick}  {fname}")


if __name__ == "__main__":
    main()