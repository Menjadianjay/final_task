import sys
sys.path.insert(0, r"D:\doc\2026\FINAL_CODE\src") 
from codeStat.modelimports.functionImporrLIB import *
# ─────────────────────────────────────────────
#  Step 1 : Pre-Emphasis
# ─────────────────────────────────────────────
def pre_emphasis(signal : np.ndarray, a = 0.97):
    return np.append(signal[0],signal[1:] - (a* signal[:-1]))

# ─────────────────────────────────────────────
#  Step 2 : Frame Blocking
# ─────────────────────────────────────────────
def frame_blocking(signal: np.ndarray, sample_rate: int,
                   frame_size=0.025, frame_distance=0.010):
    frame_length  = int(round(frame_size     * sample_rate))
    frame_diff    = int(round(frame_distance * sample_rate))
    n_frames      = 1 + (len(signal) - frame_length) // frame_diff

    frames = np.array([
        signal[i * frame_diff : i * frame_diff + frame_length]
        for i in range(n_frames)
    ])
    return frames

# ─────────────────────────────────────────────
#  Step 3 : Windowing (Hamming)
# ─────────────────────────────────────────────
def windowing(frames: np.ndarray):
    frame_shape = frames.shape[1]
    n = np.arange(frame_shape)

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_shape - 1))  # ✅
    windowed = frames * hamming
    return windowed

# ─────────────────────────────────────────────
#  Step 4 : Fast Fourier Transform (FFT)
# ─────────────────────────────────────────────
def apply_fft(windowed_frames: np.ndarray):
    frame_length = windowed_frames.shape[1]        # ambil panjang 1 bingkai
    n_fft = 1 << (frame_length - 1).bit_length()  # cari perpangkatan 2 terdekat
    magnitude_spectrum = np.abs(np.fft.rfft(windowed_frames, n=n_fft))  # hitung FFT
    power_spectrum = (1.0 / n_fft) * (magnitude_spectrum ** 2)          # hitung daya
    return power_spectrum

# ─────────────────────────────────────────────
#  Step 5 : Mel Filter Bank
# ─────────────────────────────────────────────
def mel_filter_bank(power_spectrum: np.ndarray,
                    sample_rate: int,
                    n_mels: int = 40,
                    fmin: float = 0.0,
                    fmax: float = None) -> np.ndarray:

    n_fft = (power_spectrum.shape[1] - 1) * 2   # turunkan n_fft dari power_spectrum
    fmax  = fmax or sample_rate / 2.0

    to_mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    to_hz  = lambda m: 700.0 * (np.exp(m / 1127.0) - 1.0)

    # Buat n_mels+2 titik merata di skala Mel lalu balik ke Hz
    mel_points = np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2)
    hz_points  = to_hz(mel_points)

    # Konversi ke indeks bin FFT
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Bangun matriks filter segitiga (n_mels × n_freq_bins)
    n_freq_bins   = n_fft // 2 + 1
    filter_matrix = np.zeros((n_mels, n_freq_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]         # left, center, right
        if c != l: filter_matrix[m-1, l:c] = (np.arange(l, c) - l) / (c - l)  # sisi kiri
        if r != c: filter_matrix[m-1, c:r] = (r - np.arange(c, r)) / (r - c)  # sisi kanan

    # Terapkan filter → hindari log(0)
    return np.maximum(np.dot(power_spectrum, filter_matrix.T), 1e-10)


# ─────────────────────────────────────────────
#  Step 6 : Logarithm
# ─────────────────────────────────────────────
def logarithm_log(filter_bank_energies: np.ndarray):
    return np.log(filter_bank_energies)

# ─────────────────────────────────────────────
#  Step 7 : DCT
# ─────────────────────────────────────────────
def dct(log_energies: np.ndarray, n_mfcc: int = 13):
    K = log_energies.shape[1]                    # total filter Mel
    n = np.arange(n_mfcc)                        # indeks koefisien 0..n_mfcc-1
    k = np.arange(1, K + 1)                      # indeks filter 1..K
    
    phase = np.pi / K * (k - 0.5)               # hitung bagian (π/K * (k-0.5)) per filter
    cos_matrix = np.cos(np.outer(n, phase))      # outer: broadcast n × phase → (n_mfcc × K)

    return np.dot(log_energies, cos_matrix.T)    # (num_frames × K) dot (K × n_mfcc)


# ─────────────────────────────────────────────
#  Step 8 : Deltas and Double Deltas
# ─────────────────────────────────────────────
def compute_deltas(mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Delta: selisih bingkai berurutan
    delta        = np.diff(mfcc,        axis=0, prepend=mfcc[:1])
    # Double delta: selisih dari delta
    double_delta = np.diff(delta,       axis=0, prepend=delta[:1])
    return delta, double_delta


def extract_features(wav_path: str) -> np.ndarray:
    audio_id = os.path.splitext(os.path.basename(wav_path))[0]

    # Step 0a - Sampling
    samples, sr = Sampling_Quantization.sampling(wav_path)

    # Step 0b - Quantization
    kode_pcm, quantized = Sampling_Quantization.quantization(samples)

    # Step 0c - Voice Activity Detection
    sinyal_bersih, frame_results = VAD.voice_activity_detection(quantized, sr)

    # Step 1 - Pre Emphasis
    signal = pre_emphasis(sinyal_bersih)

    # Step 2 - Frame Blocking
    frames = frame_blocking(signal, sr)

    # Step 3 - Windowing
    frames = windowing(frames)

    # Step 4 - FFT
    power_spectrum = apply_fft(frames)

    # Step 5 - Mel Filter Bank
    filter_energies = mel_filter_bank(power_spectrum, sr)

    # Step 6 - Logarithm
    log_energies = np.log(filter_energies)

    # Step 7 - DCT
    mfcc = dct(log_energies)

    # Step 8 - Delta & Double Delta
    delta        = np.diff(mfcc,  axis=0, prepend=mfcc[:1])
    double_delta = np.diff(delta, axis=0, prepend=delta[:1])

    # Gabungkan [MFCC | Δ | ΔΔ]
    return np.concatenate([mfcc, delta, double_delta], axis=1)