import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.impgmm as GMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.impgmm import *

_SR         = 16000
_N_MFCC     = 13
_N_FFT      = 512
_HOP_LENGTH = 160
_WIN_LENGTH = 400
_N_MELS     = 26
_PRE_EMPH   = 0.97
_FMIN       = 0.0
_FMAX       = 4000.0
_DELTA_W    = 2


def extract_features(wav_path: str) -> np.ndarray:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"File tidak ditemukan: {wav_path}")

    if _LIBROSA:
        return _mfcc_librosa(wav_path)
    elif _SCIPY:
        return _mfcc_manual(wav_path)
    else:
        raise ImportError("Install librosa atau scipy: pip install librosa scipy")


def _mfcc_librosa(wav_path: str) -> np.ndarray:
    """Ekstrak MFCC dengan librosa (lebih cepat dan stabil)."""
    y, _ = librosa.load(wav_path, sr=_SR, mono=True)
    if len(y) == 0:
        raise ValueError(f"Audio kosong: {wav_path}")

    # Pre-emphasis: X(n) = y(n) - 0.97 × y(n-1)  [formula 3.3.4.1.1]
    y = np.append(y[0], y[1:] - _PRE_EMPH * y[:-1])

    mfcc    = librosa.feature.mfcc(
        y=y, sr=_SR, n_mfcc=_N_MFCC,
        n_fft=_N_FFT, hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH, n_mels=_N_MELS,
        fmin=_FMIN, fmax=_FMAX, window="hamming",
    )
    delta   = librosa.feature.delta(mfcc, width=_DELTA_W * 2 + 1)
    delta2  = librosa.feature.delta(mfcc, order=2, width=_DELTA_W * 2 + 1)

    return np.vstack([mfcc, delta, delta2]).T.astype(np.float32)


def _mfcc_manual(wav_path: str) -> np.ndarray:
    """Fallback manual MFCC tanpa librosa."""
    sr, audio = _wavfile.read(wav_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != _SR:
        n_new = int(len(audio) * _SR / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_new),
            np.arange(len(audio)), audio
        )

    y = np.append(audio[0], audio[1:] - _PRE_EMPH * audio[:-1])

    # Framing
    n_frames = max(1, 1 + (len(y) - _WIN_LENGTH) // _HOP_LENGTH)
    frames   = np.zeros((n_frames, _WIN_LENGTH), dtype=np.float32)
    for i in range(n_frames):
        s = i * _HOP_LENGTH
        e = s + _WIN_LENGTH
        chunk = y[s:e]
        frames[i, :len(chunk)] = chunk

    frames *= np.hamming(_WIN_LENGTH).astype(np.float32)

    fft_mag = np.abs(np.fft.rfft(frames, n=_N_FFT))  # (T, N_FFT//2+1)

    # Mel filter bank
    def hz2mel(f): return 1127.0 * np.log1p(f / 700.0)
    def mel2hz(m): return 700.0 * (np.exp(m / 1127.0) - 1.0)

    mel_pts = np.linspace(hz2mel(_FMIN), hz2mel(_FMAX), _N_MELS + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((_N_FFT + 1) * hz_pts / _SR).astype(int)
    fb      = np.zeros((_N_MELS, _N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, _N_MELS + 1):
        for k in range(bins[m - 1], bins[m]):
            fb[m-1, k] = (k - bins[m-1]) / max(bins[m] - bins[m-1], 1)
        for k in range(bins[m], bins[m + 1]):
            fb[m-1, k] = (bins[m+1] - k) / max(bins[m+1] - bins[m], 1)

    mel_e = np.maximum(fft_mag ** 2 @ fb.T, 1e-10)
    log_e = np.log10(mel_e)  # (T, N_MELS)

    # DCT → MFCC
    N = _N_MELS
    dct_m = np.cos(np.pi * np.arange(_N_MFCC)[:, None]
                   * (np.arange(N) + 0.5)[None, :] / N)
    mfcc  = (log_e @ dct_m.T) * np.sqrt(2.0 / N)  # (T, N_MFCC)

    # Delta
    def delta_feat(f, w=_DELTA_W):
        T, D  = f.shape
        denom = 2.0 * sum(n**2 for n in range(1, w + 1))
        d     = np.zeros_like(f)
        for t in range(T):
            for n in range(1, w + 1):
                d[t] += n * (f[min(t+n, T-1)] - f[max(t-n, 0)])
        return d / denom

    d1 = delta_feat(mfcc)
    d2 = delta_feat(d1)
    return np.concatenate([mfcc, d1, d2], axis=1).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 6 — CACHE MFCC (OPTIMASI UTAMA)
# ══════════════════════════════════════════════════════════════════════════════

def baca_transkrip_csv(csv_path: str) -> Dict[str, str]:
    """Baca CSV → {audio_id: kalimat}."""
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        audio_id = os.path.splitext(os.path.basename(str(row["Path_Data"])))[0]
        result[audio_id] = str(row["sentences"]).lower().strip()
    return result


def build_mfcc_cache(
    sampled_root   : str,
    dict_transkrip : Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    Extract MFCC semua audio SEKALI → simpan di dict.
    Semua tahap selanjutnya baca dari dict ini, tidak buka file lagi.

    Returns:
        cache_mfcc : {audio_id: (T, 39)}
        meta_list  : list info untuk HasilMFCC.csv
    """
    wav_files  = glob.glob(os.path.join(sampled_root, "**/*.wav"),  recursive=True)
    wav_files += glob.glob(os.path.join(sampled_root, "**/*.flac"), recursive=True)

    cache_mfcc: Dict[str, np.ndarray] = {}
    meta_list : List[Dict]            = []
    n_total   = len(wav_files)

    print(f"  Mengekstrak MFCC dari {n_total} file ...")
    t0 = time.time()

    for idx, wav_path in enumerate(wav_files, 1):
        try:
            audio_id = os.path.splitext(os.path.basename(wav_path))[0]
            label    = dict_transkrip.get(audio_id, "")
            if not label:
                continue

            fitur = extract_features(wav_path)  # (T, 39)
            cache_mfcc[audio_id] = fitur
            meta_list.append({
                "Id"           : audio_id,
                "Label"        : label,
                "Jumlah Frame" : fitur.shape[0],
                "Dimensi Fitur": fitur.shape[1],
            })

            if idx % 100 == 0:
                ela = time.time() - t0
                eta = ela / idx * (n_total - idx)
                print(f"    [{idx}/{n_total}] cache={len(cache_mfcc)}  "
                      f"elapsed={ela:.0f}s  ETA={eta:.0f}s")

        except Exception as e:
            print(f"    [SKIP] {os.path.basename(wav_path)}: {e}")

    print(f"  Selesai: {len(cache_mfcc)} audio dalam {time.time()-t0:.1f}s")
    return cache_mfcc, meta_list

