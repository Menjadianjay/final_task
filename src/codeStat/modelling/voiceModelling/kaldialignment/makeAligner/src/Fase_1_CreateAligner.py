import os, re, sys, glob, math, shutil
import pandas as pd
from typing import Dict, List, Tuple
import tgt

SAMPLED_ROOT = r"D:\118_Rio\118\datasampled"
CSV_PATH     = r"D:\118_Rio\118\src\codeStat\preModelling\out\databersih.csv"
DIR_MFA_WORK = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\MFA\sampleddata\mfa_workspace"
DIR_TEXTGRID = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\MFA\sampleddata\mfa_textgrid"
DIR_OUTPUT   = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\makeAligner\sampleddata"

HOP_LENGTH      = 160
SR              = 16000
FRAME_SHIFT_SEC = HOP_LENGTH / SR
MIN_FRAME_VALID = 3

LABEL_SKIP = {"", "<unk>", "sp", "sil", "<eps>", "spn", "<s>", "</s>"}

PETA_FONEM: Dict[str, str] = {
    "ng":"ng","ny":"ny","th":"th","dh":"dh",
    "a":"a","b":"b","c":"ch","d":"d","e":"e","f":"f","g":"g","h":"h",
    "i":"i","j":"j","k":"k","l":"l","m":"m","n":"n","o":"o","p":"p",
    "r":"r","s":"s","t":"t","u":"u","v":"v","w":"w","y":"y","z":"z",
    "x":"ks","q":"k",
}

_MJ = {
    "Ã©":"é","Ã¨":"è","Ã ":"à","Ã¢":"â","Ã®":"î","Ã´":"ô",
    "Ã»":"û","Ã«":"ë","Ã¯":"ï","Ã¼":"ü","Ã±":"ñ","Ã§":"ç",
}
_AC = str.maketrans({
    "é":"e","è":"e","ê":"e","ë":"e","à":"a","á":"a","â":"a","ä":"a",
    "î":"i","ï":"i","í":"i","ì":"i","ô":"o","ó":"o","ò":"o","ö":"o",
    "û":"u","ú":"u","ù":"u","ü":"u","ñ":"n","ç":"c",
})


def normalisasi(teks: str) -> str:
    for salah, benar in _MJ.items(): teks = teks.replace(salah, benar)
    teks = teks.lower().strip()
    teks = re.sub(r'["\'`\u201c\u201d\u2018\u2019()\[\]{}«»]', " ", teks)
    teks = re.sub(r"[+\-*/=<>%^&|~\\@#$.,;:!?]", " ", teks)
    teks = teks.translate(_AC)
    teks = re.sub(r"[^a-z\s]", "", teks)
    return re.sub(r"\s+", " ", teks).strip()

def kata_ke_fonem(kata: str) -> List[str]:
    kata = normalisasi(kata); fonem, i = [], 0
    while i < len(kata):
        if i+1 < len(kata) and kata[i:i+2] in PETA_FONEM:
            fonem.append(PETA_FONEM[kata[i:i+2]]); i += 2
        elif kata[i] in PETA_FONEM:
            fonem.append(PETA_FONEM[kata[i]]); i += 1
        else:
            print(f"[WARN] Karakter tidak dikenal dibuang: '{kata[i]}' dalam kata '{kata}'")
            i += 1
    return fonem if fonem else list(kata)


def baca_transkrip(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path, sep=",", encoding="utf-8", dtype=str)
    return {os.path.splitext(os.path.basename(str(r["Path_Data"])))[0]:
            str(r["sentences_bersih"]).strip() for _, r in df.iterrows()}


def siapkan_korpus(sampled_root: str, transkrip: Dict[str, str],
                   dir_mfa_work: str) -> Tuple[int, int]:
    dir_corpus = os.path.join(dir_mfa_work, "corpus")
    os.makedirs(dir_corpus, exist_ok=True)

    wav_files  = glob.glob(os.path.join(sampled_root, "**/*.wav"),  recursive=True)
    n_ok = n_skip = 0
    print(f"Mempersiapkan {len(wav_files)} audio untuk MFA ...")

    for wav_path in wav_files:
        audio_id     = os.path.splitext(os.path.basename(wav_path))[0]
        label_bersih = normalisasi(transkrip.get(audio_id, "").strip())
        if not label_bersih: n_skip += 1; continue

        ext     = os.path.splitext(wav_path)[1]
        dst_wav = os.path.join(dir_corpus, audio_id + ext)
        dst_lab = os.path.join(dir_corpus, audio_id + ".lab")

        if not os.path.exists(dst_wav): shutil.copy2(wav_path, dst_wav)
        with open(dst_lab, "w", encoding="utf-8") as f: f.write(label_bersih)
        n_ok += 1

    print(f"Selesai: {n_ok} audio siap, {n_skip} dilewati (tidak ada transkripsi)")
    print(f"Folder korpus: {dir_corpus}")
    return n_ok, n_skip


def buat_pronunciation_dictionary(transkrip: Dict[str, str],
                                   dir_mfa_work: str) -> str:
    kata_set = set()
    for label in transkrip.values():
        for kata in normalisasi(label).split():
            kata_set.add(kata)

    dict_path = os.path.join(dir_mfa_work, "javanese_dictionary.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for kata in sorted(kata_set):
            fonem_list = kata_ke_fonem(kata)
            if fonem_list:
                f.write(f"{kata}\t{' '.join(fonem_list)}\n")

    print(f"Dictionary: {len(kata_set)} kata → {dict_path}")
    return dict_path


def tampilkan_perintah_mfa(dir_mfa_work: str, dir_textgrid: str) -> None:
    dir_corpus = os.path.join(dir_mfa_work, "corpus")
    dict_path  = os.path.join(dir_mfa_work, "javanese_dictionary.txt")
    model_path = os.path.join(dir_mfa_work, "javanese_acoustic_model.zip")
    print("\n LANGKAH SELANJUTNYA — Jalankan di Anaconda Prompt")
    print(f"""
  1. Aktifkan environment MFA:
     conda activate mfa

  2. Latih model akustik sekaligus alignment:
     mfa train "{dir_corpus}" "{dict_path}" "{model_path}" --output_directory "{dir_textgrid}" --clean --num_jobs 4

  3. Setelah selesai, jalankan validasi:
     python Fase_1_CreateAligner.py --validate
""")


def parse_textgrid(tg_path: str) -> List[Tuple[str, float, float]]:
    hasil = []
    try:
        tg   = tgt.io.read_textgrid(tg_path, include_empty_intervals=False)
        tier = next((t for t in tg.tiers if t.name.lower() in ("words","word")), None)
        if tier is None:
            tier = next((t for t in tg.tiers if t.name.lower() not in ("phones","phone")), None)
        if tier is None: return hasil

        for interval in tier.intervals:
            teks = interval.text.strip()
            if teks.lower() in LABEL_SKIP: continue
            kata = normalisasi(teks)
            if kata: hasil.append((kata, interval.start_time, interval.end_time))
    except Exception:
        print(f"[WARN] Gagal parse: {tg_path}")
    return hasil

def detik_ke_frame_start(t: float) -> int: return round(t / FRAME_SHIFT_SEC)
def detik_ke_frame_end(t: float)   -> int: return math.ceil(t / FRAME_SHIFT_SEC)


def buat_alignment_map(dir_textgrid: str, transkrip: Dict[str, str],
                       dir_output: str) -> pd.DataFrame:
    tg_files  = glob.glob(os.path.join(dir_textgrid, "**/*.TextGrid"), recursive=True)
    tg_files += glob.glob(os.path.join(dir_textgrid, "**/*.textgrid"), recursive=True)
    tg_files  = list({p.lower(): p for p in tg_files}.values())
    print(f"Memproses {len(tg_files)} file TextGrid ...")

    rows, n_ok, n_kosong, n_filter = [], 0, 0, 0

    for tg_path in tg_files:
        audio_id  = os.path.splitext(os.path.basename(tg_path))[0]
        intervals = parse_textgrid(tg_path)
        if not intervals: n_kosong += 1; continue

        n_valid = 0
        for kata, t_start, t_end in intervals:
            if not kata: continue
            fs, fe = detik_ke_frame_start(t_start), detik_ke_frame_end(t_end)
            durasi = fe - fs
            if durasi < MIN_FRAME_VALID: n_filter += 1; continue
            rows.append({"audio_id":audio_id,"kata":kata,"frame_start":fs,
                         "frame_end":fe,"t_start":round(t_start,4),
                         "t_end":round(t_end,4),"durasi_frame":durasi})
            n_valid += 1

        if n_valid > 0: n_ok += 1
        else: n_kosong += 1

    df = pd.DataFrame(rows)
    if df.empty: print("[ERROR] Tidak ada alignment yang berhasil di-parse!"); return df

    os.makedirs(dir_output, exist_ok=True)
    out_path = os.path.join(dir_output, "alignment_map.csv")
    df.to_csv(out_path, index=False)

    coverage = n_ok / len(transkrip) * 100 if transkrip else 0
    print(f"\n  ── Ringkasan Alignment ──────────────────────────")
    print(f"  TextGrid berhasil  : {n_ok} ({coverage:.1f}% dari total audio)")
    print(f"  TextGrid kosong    : {n_kosong}")
    print(f"  Interval difilter  : {n_filter} (< {MIN_FRAME_VALID} frame = noise)")
    print(f"  Total kata-interval: {len(df)}")
    print(f"  Kata unik          : {df['kata'].nunique()}")
    print(f"  Audio unik         : {df['audio_id'].nunique()}")
    print(f"  Output             : {out_path}")
    print(f"  ────────────────────────────────────────────────")
    return df


def validasi_alignment(df: pd.DataFrame) -> None:
    if df.empty: return
    invalid = df[df["frame_start"] >= df["frame_end"]]
    if not invalid.empty:
        print(f"  [WARN] {len(invalid)} interval dengan frame_start >= frame_end")

    print(f"\n  Statistik durasi per kata (1 frame = 10ms):")
    print(f"    Min    : {df['durasi_frame'].min():.0f} frame")
    print(f"    Median : {df['durasi_frame'].median():.0f} frame")
    print(f"    Rerata : {df['durasi_frame'].mean():.1f} frame")
    print(f"    Max    : {df['durasi_frame'].max():.0f} frame")

    kemunculan = df.groupby("kata").size()
    print(f"\n  Kata unik dengan alignment : {df['kata'].nunique()}")
    print(f"  Kata dengan ≥5 kemunculan  : {(kemunculan >= 5).sum()}")
    print(f"  Kata dengan 1 kemunculan   : {(kemunculan == 1).sum()} (kemungkinan OOV)")
    print(f"\n  Alignment siap digunakan oleh Fase 2.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--prepare"
    print("  FASE 1 — Pembuatan Word Alignment dengan MFA")

    transkrip = baca_transkrip(CSV_PATH)
    print(f"  Transkripsi dimuat: {len(transkrip)} audio")

    if mode == "--prepare":
        print("\nMenyiapkan folder korpus MFA ...")
        siapkan_korpus(SAMPLED_ROOT, transkrip, DIR_MFA_WORK)

        print("\nMembuat pronunciation dictionary ...")
        buat_pronunciation_dictionary(transkrip, DIR_MFA_WORK)

        tampilkan_perintah_mfa(DIR_MFA_WORK, DIR_TEXTGRID)

    elif mode == "--validate":
        print("\nMembaca hasil MFA ...")
        df = buat_alignment_map(DIR_TEXTGRID, transkrip, DIR_OUTPUT)
        print("\nValidasi hasil alignment ...")
        validasi_alignment(df)

    else:
        print(f"Mode tidak dikenal: {mode}  (gunakan --prepare atau --validate)")


if __name__ == "__main__":
    main()