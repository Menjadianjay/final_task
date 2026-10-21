import os, re, glob
import pandas as pd
import soundfile as sf
from typing import Dict, Tuple

# ── Paths 

SAMPLED_ROOT = r"D:\doc\2026\FINAL_CODE\data\sampledData"
CSV_PATH     = r"D:\doc\2026\FINAL_CODE\src\codeStat\preModelling\out\sampelData.csv"
OUT_DIR      = r"D:\doc\2026\FINAL_CODE\src\codeStat\preModelling\out"

# ── Kamus Bahasa Inggris 

ENGLISH_WORDS = {
    "the","at","towards","speed","room","moved","glacial","exam","and","with",
    "from","this","that","have","been","they","their","there","where","which",
    "would","could","should","about","after","before","between","through",
    "during","without","was","were","has","had","not","but","for","nor","yet",
    "so","both","either","neither","whether","while","when","what","who","how",
    "why","these","those","such","more","most","also","just","very","can","will",
    "shall","may","might","must","does","did","its","our","your","his","her",
    "him","she","are","all","any","each","every","some","other","new","old",
    "good","well","even","still","only",
}

# ── Peta Normalisasi 

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
    teks = re.sub(r'["\'`\u201c\u201d\u2018\u2019()\[\]{}]', " ", teks)
    teks = re.sub(r"[+\-*/=<>%^&|~\\@#$.,;:!?]", " ", teks)
    teks = teks.translate(_AC)
    teks = re.sub(r"[^a-z\s]", "", teks)
    return re.sub(r"\s+", " ", teks).strip()

# ── Tahap Pengolahan Transkripsi ─
def olah_transkripsi(csv_path: str, sampled_root: str,
                     out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    print(f"  Total transkripsi awal: {len(df):,}")

    # a. Normalisasi teks transkripsi
    df["sentences_bersih"] = df["sentences"].apply(
        lambda t: normalisasi(str(t))
    )
    print(f"  Normalisasi selesai")

    # b. Validasi langsung dari Path_Data di CSV (tidak perlu glob ulang)
    valid, dropped = [], []
    for _, row in df.iterrows():
        path = str(row["Path_Data"])

        if not os.path.isfile(path):
            dropped.append({**row.to_dict(), "alasan_drop": "berkas suara tidak ditemukan"})
            continue
        try:
            info = sf.info(path)
            if info.frames == 0:
                raise ValueError("audio kosong (0 frame)")
            valid.append(row.to_dict())
        except Exception as e:
            dropped.append({**row.to_dict(), "alasan_drop": str(e)})

    df_valid   = pd.DataFrame(valid)
    df_dropped = pd.DataFrame(dropped)

    os.makedirs(out_dir, exist_ok=True)
    if not df_dropped.empty:
        df_dropped.to_csv(os.path.join(out_dir, "dropped.csv"), index=False, encoding="utf-8")

    print(f"  Berkas suara ditemukan : {len(df_valid):,}")
    print(f"  Transkripsi valid      : {len(df_valid):,}")
    print(f"  Transkripsi dibuang    : {len(df_dropped):,}")
    return df_valid, df_dropped


# ── Tahap Filter Bahasa Inggris 

def filter_bahasa_inggris(df: pd.DataFrame,
                          out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bersih, dibuang = [], []
    for _, row in df.iterrows():
        n_en = sum(1 for w in str(row["sentences_bersih"]).split() if w in ENGLISH_WORDS)
        if n_en >= 1: dibuang.append({**row.to_dict(), "n_english": n_en})
        else:         bersih.append(row.to_dict())

    df_bersih  = pd.DataFrame(bersih)
    df_dibuang = pd.DataFrame(dibuang)

    os.makedirs(out_dir, exist_ok=True)
    df_bersih.to_csv(os.path.join(out_dir, "dataBersih.csv"),    index=False, encoding="utf-8")
    if not df_dibuang.empty:
        df_dibuang.to_csv(os.path.join(out_dir, "dropenglist.csv"), index=False, encoding="utf-8")

    print(f"  Data bersih            : {len(df_bersih):,}")
    print(f"  Mengandung kata Inggris: {len(df_dibuang):,}")
    if not df_dibuang.empty:
        print(f"  Rata-rata kata Inggris : {df_dibuang['n_english'].mean():.1f}")
        print(f"  Maksimum kata Inggris  : {df_dibuang['n_english'].max()}")
    return df_bersih, df_dibuang

def main():
    print("  Pra-pemrosesan Data")

    # print("\nKonversi FLAC → WAV")
    # konversi_flac_ke_wav(SAMPLED_ROOT, OUT_DIR)

    print("\nPengolahan transkripsi")
    df_valid, _ = olah_transkripsi(CSV_PATH, SAMPLED_ROOT, OUT_DIR)
    if df_valid.empty:
        print("  ERROR: tidak ada transkripsi yang valid"); return

    print("\nFilter bahasa Inggris")
    df_bersih, _ = filter_bahasa_inggris(df_valid, OUT_DIR)

    print(f"  Total akhir data bersih: {len(df_bersih):,} audio")
    print(f"  Output: {OUT_DIR}")
    for fname in ["dropped.csv","dropenglist.csv","dataBersih.csv"]:
        print(f"{fname}")

if __name__ == "__main__":
    main()