import numpy as np
import os

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
SR          = 16000   # sample rate (Hz)
N_MFCC      = 13      # jumlah koefisien MFCC
N_FFT       = 512     # ukuran FFT (zero-padded)
HOP_LENGTH  = 160     # frame shift ≈ 10ms pada 16kHz
WIN_LENGTH  = 400     # frame size  ≈ 25ms pada 16kHz
N_MELS      = 26      # jumlah mel filter bank (sub-bab 3.3.4.5)
PRE_EMPH    = 0.97    # koefisien pre-emphasis (sub-bab 3.3.4.1)
FMIN        = 0.0     # batas bawah frekuensi mel filter
FMAX        = 4000.0  # batas atas frekuensi mel filter (sesuai tabel 3.3.4.5.7)
DELTA_WIDTH = 2       # lebar window delta (standar MFCC)



def extract_features(wav_path: str) -> np.ndarray:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"File tidak ditemukan: {wav_path}")

    if LIBROSA_AVAILABLE:
        return _extract_librosa(wav_path)
    else:
        return _extract_manual(wav_path)


def _extract_librosa(wav_path: str) -> np.ndarray:
    # Load audio
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    if len(y) == 0:
        raise ValueError(f"Audio kosong: {wav_path}")

    # Pre-emphasis: X(n) = y(n) - 0.97 × y(n-1)  [formula 3.3.4.1.1]
    y_emph = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])

    # MFCC 13 koefisien
    mfcc = librosa.feature.mfcc(
        y         = y_emph,
        sr        = SR,
        n_mfcc    = N_MFCC,
        n_fft     = N_FFT,
        hop_length= HOP_LENGTH,
        win_length= WIN_LENGTH,
        n_mels    = N_MELS,
        fmin      = FMIN,
        fmax      = FMAX,
        window    = "hamming",        # Hamming window [formula 2.3]
    )   # shape: (13, T)

    # Delta (turunan pertama dan turunan kedua)
    delta    = librosa.feature.delta(mfcc, width=DELTA_WIDTH * 2 + 1)
    delta2   = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH * 2 + 1)

    fitur = np.vstack([mfcc, delta, delta2]).T

    return fitur.astype(np.float32)


def _extract_manual(wav_path: str) -> np.ndarray:
    # Load WAV
    if SCIPY_AVAILABLE:
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
    else:
        raise ImportError("librosa atau scipy diperlukan")
    #resample
    if sr != SR:
        ratio = SR / sr
        n_new = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_new),
            np.arange(len(audio)),
            audio
        )

    # Pre-emphasis
    y_emph = np.append(audio[0], audio[1:] - PRE_EMPH * audio[:-1])

    # Framing
    frames = _framing(y_emph, WIN_LENGTH, HOP_LENGTH)   # (T, WIN)
    T      = frames.shape[0]

    # Hamming window
    hamming = np.hamming(WIN_LENGTH)
    frames  = frames * hamming

    # FFT → magnitude spectrum
    fft_mag = np.abs(np.fft.rfft(frames, n=N_FFT))      # (T, N_FFT//2+1)

    # Mel filter bank
    mel_fb  = _mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX)   # (N_MELS, N_FFT//2+1)

    # Energi mel → log
    mel_energy = np.dot(fft_mag ** 2, mel_fb.T)          # (T, N_MELS)
    mel_energy = np.where(mel_energy > 0, mel_energy, 1e-10)
    log_mel    = np.log10(mel_energy)                     # (T, N_MELS)

    # DCT → MFCC
    mfcc = _dct(log_mel, N_MFCC)                          # (T, N_MFCC)

    # Delta & Delta-Delta
    delta  = _compute_delta(mfcc, DELTA_WIDTH)
    delta2 = _compute_delta(delta, DELTA_WIDTH)

    return np.concatenate([mfcc, delta, delta2], axis=1).astype(np.float32)

def _framing(y: np.ndarray, win_len: int, hop_len: int) -> np.ndarray:
    """Potong sinyal menjadi frame-frame."""
    n_frames = 1 + (len(y) - win_len) // hop_len
    frames   = np.zeros((n_frames, win_len), dtype=np.float32)
    for i in range(n_frames):
        start        = i * hop_len
        frames[i, :] = y[start: start + win_len]
    return frames


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """
    Buat mel filter bank.
    Formula mel: Mel(f) = 1127 × ln(1 + f/700)
    """
    def hz_to_mel(f):  return 1127.0 * np.log1p(f / 700.0)
    def mel_to_hz(m):  return 700.0 * (np.exp(m / 1127.0) - 1.0)

    mel_min    = hz_to_mel(fmin)
    mel_max    = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)

    freq_bins  = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb         = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        f_m1, f_m, f_m1p = freq_bins[m - 1], freq_bins[m], freq_bins[m + 1]
        for k in range(f_m1, f_m):
            fb[m - 1, k] = (k - f_m1) / (f_m - f_m1 + 1e-10)
        for k in range(f_m, f_m1p):
            fb[m - 1, k] = (f_m1p - k) / (f_m1p - f_m + 1e-10)

    return fb


def _dct(x: np.ndarray, n_coeff: int) -> np.ndarray:
    T, N  = x.shape
    n_idx = np.arange(n_coeff)
    k_idx = np.arange(N) + 0.5
    dct_m = np.cos(np.pi * n_idx[:, np.newaxis] * k_idx[np.newaxis, :] / N)
    return (x @ dct_m.T) * np.sqrt(2.0 / N)


def _compute_delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    T, D    = feat.shape
    delta   = np.zeros_like(feat)
    denom   = 2.0 * sum(i ** 2 for i in range(1, width + 1))

    for t in range(T):
        for n in range(1, width + 1):
            t_plus  = min(t + n, T - 1)
            t_minus = max(t - n, 0)
            delta[t] += n * (feat[t_plus] - feat[t_minus])
    delta /= denom
    return delta