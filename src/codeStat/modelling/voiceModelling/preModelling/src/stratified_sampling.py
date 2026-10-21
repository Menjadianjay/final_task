import os, random, shutil
import pandas as pd
from glob import glob

DATA_ROOT    = r"D:\doc\2026\FINAL_CODE\data\Data"
SAMPLED_ROOT = r"D:\doc\2026\FINAL_CODE\data\sampledData"
OUT_CSV      = r"D:\doc\2026\FINAL_CODE\src\codeStat\preModelling\out\sampelData.csv"

SEED   = 42
PERSEN = 0.1
EXT    = "*.wav"

def sampling_stratified(persen: float, dest_root: str) -> list[str]:
    random.seed(SEED)
    semua, sampel = 0, 0
    semua_dipilih = []   # ← simpan path file hasil copy

    for folder in sorted(os.listdir(DATA_ROOT), key=lambda x: int(x)):
        data_path = os.path.join(DATA_ROOT, folder, "asr_javanese", "data")
        if not os.path.isdir(data_path):
            continue

        hex_prefix = format(int(folder), "x")

        for i in range(16):
            subfolder = hex_prefix + format(i, "x")
            files     = glob(os.path.join(data_path, subfolder, EXT))
            if not files:
                continue

            n       = max(1, int(len(files) * persen))
            dipilih = random.sample(files, n)
            dest    = os.path.join(dest_root, folder, subfolder)
            os.makedirs(dest, exist_ok=True)

            for f in dipilih:
                shutil.copy(f, dest)
                dest_path = os.path.join(dest, os.path.basename(f))
                semua_dipilih.append(dest_path)   # ← catat path tujuan

            semua  += len(files)
            sampel += n

    print(f"  Total keseluruhan: {semua:,}")
    print(f"  Total sampel     : {sampel:,} ({persen*100:.0f}%)")
    return semua_dipilih   # ← return list path


def buat_dataframe(file_list: list[str], data_root: str) -> pd.DataFrame:
    tsv_frames = []
    for folder in sorted(os.listdir(data_root), key=lambda x: int(x)):
        tsv_path = os.path.join(data_root, folder, "asr_javanese", "utt_spk_text.tsv")
        if not os.path.isfile(tsv_path):
            continue
        df_tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                             names=["file_name", "id", "sentences"])
        tsv_frames.append(df_tsv)

    df_all_tsv = pd.concat(tsv_frames, ignore_index=True)
    df_all_tsv["stem"] = df_all_tsv["file_name"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0])

    # ← Tambah ini: hapus duplikat stem, keep first
    dup_count = df_all_tsv["stem"].duplicated().sum()
    print(f"  Duplikat stem di TSV: {dup_count:,}")
    df_all_tsv = df_all_tsv.drop_duplicates(subset="stem", keep="first")

    baris = [{"stem": os.path.splitext(os.path.basename(f))[0],
              "Path_Data": os.path.abspath(f)}
             for f in file_list]

    df_sampel = pd.DataFrame(baris)
    df_gabung = df_sampel.merge(df_all_tsv[["stem", "sentences"]],
                                on="stem", how="left")
    df_gabung = df_gabung.drop(columns=["stem"])

    print(f"  Total file sampel : {len(df_gabung):,}")
    print(f"  Cocok transkripsi : {df_gabung['sentences'].notna().sum():,}")
    print(f"  Tidak cocok       : {df_gabung['sentences'].isna().sum():,}")
    return df_gabung

def main():
    print("Stratified Sampling")

    print(f"\nSampling {PERSEN*100:.0f}% dari tiap subfolder ...")
    file_list = sampling_stratified(PERSEN, SAMPLED_ROOT)   # ← terima list

    print("\nMembuat DataFrame dan mencocokkan transkripsi ...")
    df = buat_dataframe(file_list, DATA_ROOT)   # ← kirim list, bukan path folder

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  Disimpan: {OUT_CSV}")

if __name__ == "__main__":
    main()