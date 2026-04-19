# HYBRID GMM-TDNN DAN TDNN ASR — Bahasa Jawa

> Perbandingan Kinerja Model Hybrid GMM-TDNN dan TDNN dalam Pengenalan Bahasa Jawa

Pipeline Automatic Speech Recognition (ASR) untuk Bahasa Jawa menggunakan Gaussian Mixture Model (GMM) + Lexicon-HMM + N-Gram Language Model, dibangun dari scratch dengan optimasi cache MFCC dan early stopping.

---

## Metadata Proyek

| Keterangan | Detail |
|---|---|
| Sumber data | Yogyakarta, 2017 |
| Kolaborasi | Universitas Gadjah Mada × Google |
| Format audio | WAV / FLAC |
| Fitur akustik | MFCC 39-dimensi |
| Bahasa | Python 3.x |

---

## Struktur File

```
FINAL_CODE/
├── .vscode/
├── data/
│   ├── Data/                        # Dataset penuh ASR Bahasa Jawa (Google × UGM 2017)
│   │   └── [0-15]/asr_javanese/data/[sub-hash]/   # Audio WAV/FLAC per speaker hash
│   ├── sampled/                     # Data hasil stratified sampling
│   │   └── [0-15]/[sub-hash]/
│   └── sampled_small/               # Subset kecil untuk eksperimen cepat
│       └── [0-15]/[sub-hash]/
├── skripsi/                         # Virtual environment Python
│   └── Lib/site-packages/           # Dependensi terinstall
├── src/
│   ├── codeStat/
│   │   ├── Filecsv_pdf/             # Konversi dan ekspor hasil ke CSV/PDF
│   │   ├── modelimports/            # Helper import modul
│   │   ├── modelling/
│   │   │   ├── languageModelling/
│   │   │   │   └── srcLanguageModelling/          # N-Gram LM
│   │   │   └── voiceModelling/
│   │   │       └── srcVoiceModelling/
│   │   │           └── GMM_TDNN/
│   │   │               ├── out/GMM/               # Output hasil model GMM
│   │   │               ├── nonfix_out/GMM/         # Output iterasi awal (non-final)
│   │   │               └── src/
│   │   │                   ├── GMM/               # Kode model GMM + test
│   │   │                   └── TDNN/              # Kode model TDNN + test
│   │   ├── pastModelling/
│   │   │   └── wordErrorRate/src_WER/             # Evaluasi WER versi lama
│   │   └── preModelling/
│   │       ├── featureExtraction/                 # Ekstraksi MFCC 39-dim
│   │       │   ├── out_featureExtraction/
│   │       │   └── src_featureExtraction/
│   │       ├── list_file/src_list_file/           # Utilitas listing file audio
│   │       ├── samplingStratified/                # Stratified sampling script
│   │       │   ├── out_samplingStratified/
│   │       │   └── src_StratifiedSampling/
│   │       ├── sampling_Quantization/             # Sampling & kuantisasi audio
│   │       │   ├── out_Sampling_Quantization/
│   │       │   ├── src_Sampling_Quantization/
│   │       │   └── visualizations/                # Plot waveform, VAD, sampling
│   │       └── voiceActivityDetection/            # VAD preprocessing
│   │           ├── hasilVAD_detail_pecahan/
│   │           ├── out_VoiceActivityDetection/
│   │           └── src_VoiceActivityDetection/
│   ├── src_testPart/                              # Script pengujian parsial
│   └── connection/                                # Utilitas koneksi/integrasi
└── README.md
```

---

## Cara Penggunaan

### 1. Buat virtual environment

Isolasi dependensi agar tidak konflik dengan proyek lain.

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 2. Install dependensi

Jalankan sebelum melanjutkan ke langkah berikutnya.

```bash
pip install -r requirements.txt
# atau manual:
pip install numpy pandas librosa scipy
```

### 3. Edit konfigurasi path di `main_gmm.py`

Sesuaikan path ke direktori lokal Anda sebelum menjalankan pipeline.

```python
SAMPLED_ROOT = r"path/ke/folder/audio"        # folder WAV/FLAC
CSV_PATH     = r"path/ke/dataSampledSmall.csv" # CSV transkrip
DIR_SAVE     = r"path/ke/folder/output"        # folder hasil
```

### 4. Jalankan pipeline

```bash
python main_gmm.py
```

---

## Optimasi yang Diterapkan

| Masalah Versi Lama | Solusi Versi Baru |
|---|---|
| Extract MFCC 3× per audio (tahap 1, kumpulkan frame, dekode) | 1× saja → cache dict |
| Hyperparameter tuning 24× full pipeline (extract + train + decode) | 24× hanya train + decode |
| GMM score semua vocab per segmen | Top-K saja |
| N-gram kandidat: referensi ikut sebagai kandidat (curang) | Hanya hipotesis GMM |
| Konvergensi selalu sampai `max_iter` | Early stopping otomatis |

### Estimasi Penghematan Waktu

| Tahap | Sebelum | Sesudah |
|---|---|---|
| Extract MFCC (500 audio) | ~60 menit (3×) | ~20 menit (1×) |
| Hyperparameter tuning | ~2.5 jam (24× full) | ~30 menit (24× train+decode) |
| **Total** | **~3+ jam** | **~50 menit** |

---

## Parameter Utama

| Parameter | Default | Keterangan |
|---|---|---|
| `N_COMPONENTS` | 16 | K komponen GMM per kata |
| `N_ITER` | 100 | Maks iterasi EM |
| `TOL` | 1e-4 | Toleransi konvergensi (early stopping) |
| `N_GRAM` | 3 | Orde N-gram (trigram) |
| `TOP_K_DECODE` | 10 | Kandidat kata per posisi dekode |

---

## Output Files

| File | Isi |
|---|---|
| `HasilLexicon.csv` | Pemetaan kata → fonem Bahasa Jawa |
| `HasilMFCC.csv` | Info frame MFCC tiap audio |
| `HasilGMMGlobal.csv` | Parameter GMM global (mean, covariance, weight) |
| `HasilGMMPerKata.csv` | Parameter GMM tiap kata dalam leksikon |
| `HasilWER_GMM.csv` | WER detail tiap audio (S, D, I, N) |
| `HasilRingkasanWER.csv` | Statistik WER keseluruhan (mean, min, max) |
| `HasilHyperparameterTuning_GMM.csv` | Grid search K, tol, iter — semua kombinasi |

---

## Referensi Skripsi

| Sub-bab | Topik | Implementasi |
|---|---|---|
| 3.3.4 | Feature Extraction (MFCC) | `feature_extraction.py` |
| 3.3.5.1 | GMM + EM Algorithm | `gmm_model.py` |
| 3.3.6 | Lexicon fonem Bahasa Jawa | `lexicon_ngram.py` |
| 3.3.7 | N-Gram (Kneser-Ney) | `lexicon_ngram.py` |
| 3.3.9 | Word Error Rate (WER) | `lexicon_ngram.py` |
| Tabel 3.4.1 | Hyperparameter Tuning | `gmm_pipeline.py` |

---

## Catatan Penting

> **Data:** Bersumber dari proyek kolaborasi Google × Universitas Gadjah Mada (2017), dikumpulkan dari penutur asli Yogyakarta. Tersedia dalam format WAV dan dikompresi ke FLAC untuk efisiensi jaringan. Periksa dokumen lisensi resmi sebelum publikasi atau pengembangan lebih lanjut.

> **Path:** Semua path direktori bersifat lokal. Pastikan `SAMPLED_ROOT`, `CSV_PATH`, dan `DIR_SAVE` di `main_gmm.py` sudah disesuaikan sebelum menjalankan pipeline. Kode merupakan gabungan kode scraped dan personal — verifikasi kompatibilitas pustaka sebelum digunakan.

---

*GMM-TDNN ASR · Bahasa Jawa · J.S*
