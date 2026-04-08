<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GMM-TDNN ASR — Bahasa Jawa</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --accent:#1D9E75; --accent-light:#E1F5EE; --accent-mid:#0F6E56; --accent-dark:#085041;
      --bg:#f9f8f6; --surface:#f1efe8; --card:#ffffff; --border:rgba(0,0,0,0.10);
      --text:#1a1a18; --muted:#666660; --code-bg:#eeecea;
      --blue-bg:#E6F1FB; --blue-text:#0C447C; --amber-bg:#FAEEDA; --amber-text:#633806;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg:#1a1a18; --surface:#242422; --card:#2a2a28; --border:rgba(255,255,255,0.10);
        --text:#e8e6e0; --muted:#999890; --code-bg:#2e2e2c;
        --accent-light:#04342C; --accent-mid:#5DCAA5; --accent-dark:#9FE1CB;
        --blue-bg:#042C53; --blue-text:#B5D4F4; --amber-bg:#412402; --amber-text:#FAC775;
      }
    }
    html { scroll-behavior: smooth; }
    body { font-family:'Sora',sans-serif; font-size:15px; line-height:1.75; color:var(--text); background:var(--bg); }
    .layout { display:grid; grid-template-columns:230px 1fr; min-height:100vh; }

    /* Sidebar */
    .sidebar { position:sticky; top:0; height:100vh; overflow-y:auto; background:var(--surface); border-right:0.5px solid var(--border); padding:2rem 1.25rem; }
    .sidebar-logo { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500; color:var(--accent-mid); letter-spacing:0.8px; text-transform:uppercase; margin-bottom:1.75rem; padding-bottom:1rem; border-bottom:0.5px solid var(--border); }
    .nav-label { font-size:10px; font-weight:500; text-transform:uppercase; letter-spacing:0.8px; color:var(--muted); margin-bottom:0.4rem; margin-top:1.25rem; }
    .nav-link { display:block; font-size:13px; color:var(--muted); text-decoration:none; padding:4px 10px; border-radius:6px; margin-bottom:2px; transition:background .12s,color .12s; }
    .nav-link:hover { background:var(--card); color:var(--text); }

    /* Main */
    .main { padding:3rem 4rem 4rem; max-width:880px; }

    /* Hero */
    .hero { background:var(--card); border:0.5px solid var(--border); border-radius:14px; padding:2.5rem; margin-bottom:3rem; position:relative; overflow:hidden; }
    .hero::after { content:''; position:absolute; top:-50px; right:-50px; width:250px; height:250px; background:radial-gradient(circle,var(--accent-light) 0%,transparent 65%); pointer-events:none; opacity:0.5; }
    .badge-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1.25rem; }
    .badge { font-family:'IBM Plex Mono',monospace; font-size:11px; padding:3px 10px; border-radius:20px; border:0.5px solid; white-space:nowrap; }
    .badge-green { background:var(--accent-light); color:var(--accent-dark); border-color:var(--accent); }
    .badge-blue  { background:var(--blue-bg); color:var(--blue-text); border-color:#378ADD; }
    .badge-amber { background:var(--amber-bg); color:var(--amber-text); border-color:#EF9F27; }
    .badge-gray  { background:var(--surface); color:var(--muted); border-color:var(--border); }
    .hero-title  { font-size:26px; font-weight:600; letter-spacing:-0.4px; margin-bottom:0.3rem; position:relative; }
    .hero-sub    { font-size:14px; color:var(--muted); margin-bottom:1.25rem; }
    .hero-desc   { font-size:14.5px; line-height:1.75; max-width:600px; position:relative; }
    .meta-row    { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:1.75rem; }
    .meta-card   { background:var(--surface); border:0.5px solid var(--border); border-radius:8px; padding:11px 13px; }
    .meta-label  { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
    .meta-value  { font-size:13px; font-weight:500; }

    /* Sections */
    section { margin-bottom:3rem; scroll-margin-top:2rem; }
    .section-title { font-size:15px; font-weight:600; padding-bottom:10px; border-bottom:0.5px solid var(--border); margin-bottom:1.25rem; display:flex; align-items:center; gap:10px; }
    .section-icon { width:22px; height:22px; border-radius:6px; background:var(--accent-light); display:flex; align-items:center; justify-content:center; flex-shrink:0; }
    .section-icon svg { width:13px; height:13px; }

    /* File tree */
    .file-tree { font-family:'IBM Plex Mono',monospace; font-size:12.5px; background:var(--surface); border:0.5px solid var(--border); border-radius:10px; padding:1.5rem 1.75rem; line-height:2; overflow-x:auto; }
    .tree-root { color:var(--accent-mid); font-weight:600; }
    .tree-dir  { color:var(--accent-mid); font-weight:500; }
    .tree-file { color:var(--text); }
    .tree-note { color:var(--muted); font-size:11px; }
    .tree-leaf { color:var(--muted); }

    /* Steps */
    .steps { display:flex; flex-direction:column; gap:12px; }
    .step { display:flex; gap:16px; align-items:flex-start; background:var(--card); border:0.5px solid var(--border); border-radius:10px; padding:16px 18px; }
    .step-num { width:28px; height:28px; border-radius:50%; background:var(--accent-light); color:var(--accent-mid); font-size:12px; font-weight:600; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
    .step-title { font-size:14px; font-weight:500; margin-bottom:4px; }
    .step-desc  { font-size:13px; color:var(--muted); margin-bottom:10px; }

    /* Code */
    pre { font-family:'IBM Plex Mono',monospace; font-size:12.5px; background:var(--code-bg); border:0.5px solid var(--border); border-radius:8px; padding:1rem 1.25rem; overflow-x:auto; line-height:1.9; margin-top:8px; }
    code { font-family:'IBM Plex Mono',monospace; font-size:12px; background:var(--code-bg); padding:1px 6px; border-radius:4px; color:var(--accent-mid); }
    .kw  { color:#185FA5; }
    .str { color:#3B6D11; }
    .cmt { color:var(--muted); }

    /* Tables */
    .table-wrap { overflow-x:auto; border-radius:10px; border:0.5px solid var(--border); }
    table { width:100%; border-collapse:collapse; font-size:13.5px; }
    thead tr { background:var(--surface); }
    th { font-weight:500; font-size:12px; color:var(--muted); text-align:left; padding:11px 16px; border-bottom:0.5px solid var(--border); white-space:nowrap; }
    td { padding:11px 16px; border-bottom:0.5px solid var(--border); vertical-align:top; }
    tr:last-child td { border-bottom:none; }
    tbody tr:nth-child(even) { background:var(--surface); }

    /* Timeline */
    .timeline { display:grid; grid-template-columns:repeat(3,1fr); background:var(--surface); border:0.5px solid var(--border); border-radius:10px; overflow:hidden; margin-top:14px; }
    .time-block { padding:16px 18px; border-right:0.5px solid var(--border); }
    .time-block:last-child { border-right:none; }
    .time-label  { font-size:11px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.4px; }
    .time-before { font-family:'IBM Plex Mono',monospace; font-size:12.5px; color:#A32D2D; margin-bottom:3px; }
    .time-after  { font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:500; color:var(--accent-mid); }

    /* Notes */
    .info-box { background:var(--accent-light); border:0.5px solid var(--accent); border-radius:10px; padding:14px 16px; font-size:13px; color:var(--accent-dark); line-height:1.65; }
    .warn-box { background:var(--amber-bg); border:0.5px solid #EF9F27; border-radius:10px; padding:14px 16px; font-size:13px; color:var(--amber-text); line-height:1.65; margin-top:10px; }

    /* Ref grid */
    .ref-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; }
    .ref-card  { background:var(--card); border:0.5px solid var(--border); border-radius:10px; padding:14px 16px; }
    .ref-num   { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted); margin-bottom:4px; }
    .ref-topic { font-size:13px; font-weight:500; margin-bottom:4px; }
    .ref-file  { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--accent-mid); }

    /* Output list */
    .output-list { display:flex; flex-direction:column; gap:6px; }
    .output-row  { display:flex; align-items:flex-start; gap:12px; padding:10px 14px; background:var(--card); border:0.5px solid var(--border); border-radius:8px; font-size:13px; }
    .output-file { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--accent-mid); min-width:260px; flex-shrink:0; }
    .output-desc { color:var(--muted); }

    /* Footer */
    .footer { margin-top:2.5rem; padding-top:1.25rem; border-top:0.5px solid var(--border); font-size:12px; color:var(--muted); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }

    @media (max-width:768px) {
      .layout { grid-template-columns:1fr; }
      .sidebar { display:none; }
      .main { padding:2rem 1.5rem; }
      .meta-row { grid-template-columns:repeat(2,1fr); }
      .timeline { grid-template-columns:1fr; }
      .timeline .time-block { border-right:none; border-bottom:0.5px solid var(--border); }
      .timeline .time-block:last-child { border-bottom:none; }
    }
  </style>
</head>
<body>
<div class="layout">

  <nav class="sidebar">
    <div class="sidebar-logo">GMM-TDNN ASR</div>
    <div class="nav-label">Mulai</div>
    <a href="#struktur" class="nav-link">Struktur Proyek</a>
    <a href="#penggunaan" class="nav-link">Cara Penggunaan</a>
    <div class="nav-label">Teknis</div>
    <a href="#optimasi" class="nav-link">Optimasi</a>
    <a href="#parameter" class="nav-link">Parameter Utama</a>
    <a href="#output" class="nav-link">Output Files</a>
    <div class="nav-label">Referensi</div>
    <a href="#skripsi" class="nav-link">Referensi Skripsi</a>
    <a href="#catatan" class="nav-link">Catatan Penting</a>
  </nav>

  <main class="main">

    <div class="hero">
      <div class="badge-row">
        <span class="badge badge-green">ASR · Bahasa Jawa</span>
        <span class="badge badge-blue">GMM + HMM + N-Gram</span>
        <span class="badge badge-amber">Python 3.x</span>
        <span class="badge badge-gray">UGM × Google 2017</span>
        <span class="badge badge-gray">feature/7-modelling-hybrid</span>
      </div>
      <div class="hero-title">GMM-TDNN ASR — Bahasa Jawa</div>
      <div class="hero-sub">Perbandingan Kinerja Model Hybrid GMM-TDNN dan TDNN dalam Pengenalan Bahasa Jawa</div>
      <div class="hero-desc">Pipeline Automatic Speech Recognition (ASR) untuk Bahasa Jawa menggunakan Gaussian Mixture Model (GMM) + Lexicon-HMM + N-Gram Language Model, dibangun dari scratch dengan optimasi cache MFCC dan early stopping.</div>
      <div class="meta-row">
        <div class="meta-card"><div class="meta-label">Sumber data</div><div class="meta-value">Yogyakarta, 2017</div></div>
        <div class="meta-card"><div class="meta-label">Kolaborasi</div><div class="meta-value">UGM × Google</div></div>
        <div class="meta-card"><div class="meta-label">Format audio</div><div class="meta-value">WAV / FLAC</div></div>
        <div class="meta-card"><div class="meta-label">Fitur akustik</div><div class="meta-value">MFCC 39-dim</div></div>
      </div>
    </div>

    <!-- Struktur Proyek -->
    <section id="struktur">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><rect x="1" y="1" width="10" height="10" rx="2" stroke="#1D9E75" stroke-width="1.2"/><path d="M4 6h4M4 8.5h2" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Struktur Proyek
      </div>
      <div class="file-tree">
<span class="tree-root">FINAL_CODE/</span>
├── <span class="tree-dir">.vscode/</span>
├── <span class="tree-dir">data/</span>
│   ├── <span class="tree-dir">Data/</span>                         <span class="tree-note"># Dataset penuh ASR Bahasa Jawa (Google × UGM 2017)</span>
│   │   └── <span class="tree-leaf">[0–15]/asr_javanese/data/[hash]/</span>    <span class="tree-note"># Audio per speaker hash</span>
│   ├── <span class="tree-dir">sampled/</span>                      <span class="tree-note"># Hasil stratified sampling</span>
│   │   └── <span class="tree-leaf">[0–15]/[hash]/</span>
│   └── <span class="tree-dir">sampled_small/</span>                <span class="tree-note"># Subset kecil untuk eksperimen cepat</span>
│       └── <span class="tree-leaf">[0–15]/[hash]/</span>
├── <span class="tree-dir">skripsi/</span>                          <span class="tree-note"># Virtual environment Python</span>
│   └── <span class="tree-leaf">Lib/site-packages/</span>
├── <span class="tree-dir">src/</span>
│   └── <span class="tree-dir">codeStat/</span>
│       ├── <span class="tree-dir">Filecsv_pdf/</span>              <span class="tree-note"># Konversi dan ekspor hasil ke CSV / PDF</span>
│       ├── <span class="tree-dir">modelimports/</span>             <span class="tree-note"># Helper import modul antar notebook</span>
│       ├── <span class="tree-dir">modelling/</span>
│       │   ├── <span class="tree-dir">languageModelling/</span>
│       │   │   └── <span class="tree-dir">srcLanguageModelling/</span>       <span class="tree-note"># N-Gram Language Model</span>
│       │   └── <span class="tree-dir">voiceModelling/</span>
│       │       └── <span class="tree-dir">srcVoiceModelling/</span>
│       │           └── <span class="tree-dir">GMM_TDNN/</span>
│       │               ├── <span class="tree-dir">out/GMM/</span>            <span class="tree-note"># Output final model GMM</span>
│       │               ├── <span class="tree-dir">nonfix_out/GMM/</span>     <span class="tree-note"># Output iterasi awal (non-final)</span>
│       │               └── <span class="tree-dir">src/</span>
│       │                   ├── <span class="tree-dir">GMM/</span>            <span class="tree-note"># Kode model GMM + unit test</span>
│       │                   └── <span class="tree-dir">TDNN/</span>           <span class="tree-note"># Kode model TDNN + unit test</span>
│       ├── <span class="tree-dir">pastModelling/</span>
│       │   └── <span class="tree-dir">wordErrorRate/src_WER/</span>          <span class="tree-note"># Evaluasi WER versi lama</span>
│       └── <span class="tree-dir">preModelling/</span>
│           ├── <span class="tree-dir">featureExtraction/</span>              <span class="tree-note"># Ekstraksi MFCC 39-dimensi</span>
│           │   ├── <span class="tree-dir">out_featureExtraction/</span>
│           │   └── <span class="tree-dir">src_featureExtraction/</span>
│           ├── <span class="tree-dir">list_file/src_list_file/</span>        <span class="tree-note"># Listing file audio</span>
│           ├── <span class="tree-dir">samplingStratified/</span>             <span class="tree-note"># Stratified sampling</span>
│           │   ├── <span class="tree-dir">out_samplingStratified/</span>
│           │   └── <span class="tree-dir">src_StratifiedSampling/</span>
│           ├── <span class="tree-dir">sampling_Quantization/</span>          <span class="tree-note"># Sampling &amp; kuantisasi audio</span>
│           │   ├── <span class="tree-dir">out_Sampling_Quantization/</span>
│           │   ├── <span class="tree-dir">src_Sampling_Quantization/</span>
│           │   └── <span class="tree-dir">visualizations/</span>             <span class="tree-note"># Plot waveform, VAD, sampling</span>
│           ├── <span class="tree-dir">src_testPart/</span>                   <span class="tree-note"># Script pengujian parsial</span>
│           └── <span class="tree-dir">voiceActivityDetection/</span>         <span class="tree-note"># VAD preprocessing</span>
│               ├── <span class="tree-dir">hasilVAD_detail_pecahan/</span>
│               ├── <span class="tree-dir">out_VoiceActivityDetection/</span>
│               └── <span class="tree-dir">src_VoiceActivityDetection/</span>
│
├── <span class="tree-dir">connection/</span>                           <span class="tree-note"># Utilitas koneksi / integrasi</span>
└── <span class="tree-file">README.md</span>
      </div>
    </section>

    <!-- Cara Penggunaan -->
    <section id="penggunaan">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4.5" stroke="#1D9E75" stroke-width="1.2"/><path d="M6 4v2l1.5 1.5" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Cara Penggunaan
      </div>
      <div class="steps">
        <div class="step">
          <div class="step-num">1</div>
          <div>
            <div class="step-title">Aktifkan virtual environment</div>
            <div class="step-desc">Gunakan env <code>skripsi/</code> yang sudah ada, atau buat baru untuk isolasi dependensi.</div>
            <pre><span class="cmt"># Aktifkan env yang sudah ada (Windows Git Bash)</span>
source skripsi/Scripts/activate

<span class="cmt"># Atau buat env baru</span>
python -m venv skripsi
source skripsi/Scripts/activate</pre>
          </div>
        </div>
        <div class="step">
          <div class="step-num">2</div>
          <div>
            <div class="step-title">Install dependensi</div>
            <pre>pip install -r requirements.txt
<span class="cmt"># atau manual:</span>
pip install numpy pandas librosa scipy</pre>
          </div>
        </div>
        <div class="step">
          <div class="step-num">3</div>
          <div>
            <div class="step-title">Edit konfigurasi path di <code>main_gmm.py</code></div>
            <div class="step-desc">Data audio ada di <code>data/sampled_small/</code> atau <code>data/sampled/</code>. Output disimpan ke <code>src/.../GMM_TDNN/out/GMM/</code>.</div>
            <pre><span class="kw">SAMPLED_ROOT</span> = <span class="str">r"D:\doc\2026\FINAL_CODE\data\sampled_small"</span>
<span class="kw">CSV_PATH</span>     = <span class="str">r"D:\doc\2026\FINAL_CODE\data\dataSampledSmall.csv"</span>
<span class="kw">DIR_SAVE</span>     = <span class="str">r"D:\doc\2026\FINAL_CODE\src\codeStat\modelling\voiceModelling\srcVoiceModelling\GMM_TDNN\out\GMM"</span></pre>
          </div>
        </div>
        <div class="step">
          <div class="step-num">4</div>
          <div>
            <div class="step-title">Jalankan pipeline</div>
            <pre>cd src/codeStat/modelling/voiceModelling/srcVoiceModelling/GMM_TDNN/src/GMM
python main_gmm.py</pre>
          </div>
        </div>
      </div>
    </section>

    <!-- Optimasi -->
    <section id="optimasi">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><path d="M2 10L10 2M7 2h3v3" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        Optimasi yang Diterapkan
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Masalah versi lama</th><th>Solusi versi baru</th></tr></thead>
          <tbody>
            <tr><td>Extract MFCC 3× per audio (tahap 1, kumpulkan frame, dekode)</td><td>1× saja → cache dict</td></tr>
            <tr><td>Hyperparameter tuning 24× full pipeline (extract + train + decode)</td><td>24× hanya train + decode</td></tr>
            <tr><td>GMM score semua vocab per segmen</td><td>Top-K saja</td></tr>
            <tr><td>N-gram kandidat: referensi ikut sebagai kandidat (curang)</td><td>Hanya hipotesis GMM</td></tr>
            <tr><td>Konvergensi selalu sampai <code>max_iter</code></td><td>Early stopping otomatis</td></tr>
          </tbody>
        </table>
      </div>
      <div class="timeline">
        <div class="time-block">
          <div class="time-label">Extract MFCC (500 audio)</div>
          <div class="time-before">~60 menit (3×)</div>
          <div class="time-after">~20 menit (1×)</div>
        </div>
        <div class="time-block">
          <div class="time-label">Hyperparameter tuning</div>
          <div class="time-before">~2.5 jam (24× full)</div>
          <div class="time-after">~30 menit (24× train)</div>
        </div>
        <div class="time-block">
          <div class="time-label">Total estimasi</div>
          <div class="time-before">~3+ jam</div>
          <div class="time-after">~50 menit</div>
        </div>
      </div>
    </section>

    <!-- Parameter -->
    <section id="parameter">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><path d="M2 9h8M2 6h5M2 3h7" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Parameter Utama
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Parameter</th><th>Default</th><th>Keterangan</th></tr></thead>
          <tbody>
            <tr><td><code>N_COMPONENTS</code></td><td>16</td><td>K komponen GMM per kata</td></tr>
            <tr><td><code>N_ITER</code></td><td>100</td><td>Maks iterasi EM</td></tr>
            <tr><td><code>TOL</code></td><td>1e-4</td><td>Toleransi konvergensi (early stopping)</td></tr>
            <tr><td><code>N_GRAM</code></td><td>3</td><td>Orde N-gram (trigram)</td></tr>
            <tr><td><code>TOP_K_DECODE</code></td><td>10</td><td>Kandidat kata per posisi dekode</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Output Files -->
    <section id="output">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><rect x="1.5" y="2" width="9" height="8" rx="1.5" stroke="#1D9E75" stroke-width="1.2"/><path d="M4 5.5h4M4 7.5h2.5" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Output Files
      </div>
      <div class="output-list">
        <div class="output-row"><div class="output-file">HasilLexicon.csv</div><div class="output-desc">Pemetaan kata → fonem Bahasa Jawa</div></div>
        <div class="output-row"><div class="output-file">HasilMFCC.csv</div><div class="output-desc">Info frame MFCC tiap audio</div></div>
        <div class="output-row"><div class="output-file">HasilGMMGlobal.csv</div><div class="output-desc">Parameter GMM global (mean, covariance, weight)</div></div>
        <div class="output-row"><div class="output-file">HasilGMMPerKata.csv</div><div class="output-desc">Parameter GMM tiap kata dalam leksikon</div></div>
        <div class="output-row"><div class="output-file">HasilWER_GMM.csv</div><div class="output-desc">WER detail tiap audio (S, D, I, N)</div></div>
        <div class="output-row"><div class="output-file">HasilRingkasanWER.csv</div><div class="output-desc">Statistik WER keseluruhan (mean, min, max)</div></div>
        <div class="output-row"><div class="output-file">HasilHyperparameterTuning_GMM.csv</div><div class="output-desc">Grid search K, tol, iter — semua kombinasi</div></div>
      </div>
    </section>

    <!-- Referensi Skripsi -->
    <section id="skripsi">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><path d="M6 1v10M1 6h10" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Referensi Skripsi
      </div>
      <div class="ref-grid">
        <div class="ref-card"><div class="ref-num">Sub-bab 3.3.4</div><div class="ref-topic">Feature Extraction (MFCC)</div><div class="ref-file">preModelling/featureExtraction/</div></div>
        <div class="ref-card"><div class="ref-num">Sub-bab 3.3.5.1</div><div class="ref-topic">GMM + EM Algorithm</div><div class="ref-file">GMM_TDNN/src/GMM/</div></div>
        <div class="ref-card"><div class="ref-num">Sub-bab 3.3.6</div><div class="ref-topic">Lexicon fonem Bahasa Jawa</div><div class="ref-file">languageModelling/srcLanguageModelling/</div></div>
        <div class="ref-card"><div class="ref-num">Sub-bab 3.3.7</div><div class="ref-topic">N-Gram (Kneser-Ney)</div><div class="ref-file">languageModelling/srcLanguageModelling/</div></div>
        <div class="ref-card"><div class="ref-num">Sub-bab 3.3.9</div><div class="ref-topic">Word Error Rate (WER)</div><div class="ref-file">pastModelling/wordErrorRate/src_WER/</div></div>
        <div class="ref-card"><div class="ref-num">Tabel 3.4.1</div><div class="ref-topic">Hyperparameter Tuning</div><div class="ref-file">GMM_TDNN/out/GMM/</div></div>
      </div>
    </section>

    <!-- Catatan Penting -->
    <section id="catatan">
      <div class="section-title">
        <div class="section-icon"><svg viewBox="0 0 12 12" fill="none"><path d="M6 2a4 4 0 100 8A4 4 0 006 2zM6 5v2M6 8.2v.3" stroke="#1D9E75" stroke-width="1.2" stroke-linecap="round"/></svg></div>
        Catatan Penting
      </div>
      <div class="info-box">
        Data audio bersumber dari proyek kolaborasi Google × Universitas Gadjah Mada (2017), dikumpulkan dari penutur asli Yogyakarta. Tersimpan di <code>data/Data/</code> dalam format WAV dan dikompresi ke FLAC untuk efisiensi jaringan. Periksa dokumen lisensi resmi sebelum publikasi atau pengembangan lebih lanjut.
      </div>
      <div class="warn-box">
        Semua path bersifat lokal — sesuaikan <code>SAMPLED_ROOT</code>, <code>CSV_PATH</code>, dan <code>DIR_SAVE</code> sebelum menjalankan pipeline. Virtual environment ada di <code>skripsi/</code>. Kode merupakan gabungan kode scraped dan personal — verifikasi kompatibilitas pustaka sebelum digunakan.
      </div>
    </section>

    <div class="footer">
      <span>GMM-TDNN ASR · Bahasa Jawa · J.S</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;">feature/7-modelling-hybrid</span>
    </div>

  </main>
</div>
</body>
</html>
