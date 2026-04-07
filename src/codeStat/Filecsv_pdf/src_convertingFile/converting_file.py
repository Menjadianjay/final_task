import shutil
import pdfplumber
import soundfile as sf
import glob
import csv
from pathlib import Path
from math import ceil
import os
import math
import pandas as pd
from fpdf import FPDF

def csv_to_pdf(file_csv, delim, output_pdf, col_widths, output_dir=None):

    # ── Tentukan path output ──────────────────────────────────────────────────
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Directory tidak ditemukan: '{output_dir}'")
        output_path = os.path.join(output_dir, os.path.basename(output_pdf))
    else:
        base_dir    = os.path.dirname(os.path.abspath(file_csv))
        output_path = os.path.join(base_dir, os.path.basename(output_pdf))

    # ── Baca CSV ─────────────────────────────────────────────────────────────
    df  = pd.read_csv(file_csv, delimiter=delim).fillna("")
    pdf = FPDF(orientation="L")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_font("helvetica", size=7)

    safe = lambda v: "".join(c if ord(c) < 256 else "?" for c in str(v))
    W    = pdf.w - 2 * pdf.l_margin
    cw   = [w / sum(col_widths) * W for w in col_widths]
    LH   = pdf.font_size * 1.5

    def set_style(bold=False, fill=(255, 255, 255), text=(0, 0, 0)):
        pdf.set_font("helvetica", "B" if bold else "", 7)
        pdf.set_fill_color(*fill)
        pdf.set_text_color(*text)

    def row_h(row):
        max_lines = 1
        for v, w in zip(row, cw):
            text = safe(v)
            if not text:
                continue
            avg_char_w     = pdf.get_string_width("x")
            chars_per_line = max(1, int(w / avg_char_w)) if avg_char_w > 0 else 20
            lines          = math.ceil(len(text) / chars_per_line)
            max_lines      = max(max_lines, lines)
        return max_lines * LH

    def header():
        set_style(bold=True, fill=(52, 73, 94), text=(255, 255, 255))
        pdf.set_x(pdf.l_margin)
        for w, c in zip(cw, df.columns):
            pdf.cell(w, LH, safe(c), border=1, align="C", fill=True)
        pdf.ln()
        set_style()

    header()

    # ── Isi tabel ─────────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(df.iterrows()):
        rh = row_h(row)
        if pdf.get_y() + rh > pdf.h - pdf.b_margin:
            pdf.add_page()
            header()

        y          = pdf.get_y()
        x          = pdf.l_margin
        fill_color = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)

        for v, w in zip(row, cw):
            pdf.set_xy(x, y)
            pdf.multi_cell(w, LH, safe(v), border=1, fill=True)
            x += w

        pdf.set_xy(pdf.l_margin, y + rh)

    # ── Simpan ke path tujuan ─────────────────────────────────────────────────
    pdf.output(output_path)
    return output_path

def pdf_to_csv(input_pdf, output_csv):
    all_rows = []

    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if row:
                        all_rows.append(row)

    if not all_rows:
        print("Tidak ada tabel ditemukan.")
        return

    header = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(all_rows[0])]
    data   = [row for row in all_rows[1:] if row != all_rows[0]]

    df = pd.DataFrame(data, columns=header)
    df = df.map(lambda x: str(x).strip() if x is not None else "")
    if "sentences" in df.columns:
        df["sentences"] = df["sentences"].str.replace("\n", " ", regex=True)

    df.to_csv(output_csv, index=False)
    print(f"Selesai: {output_csv} ({len(df)} rows, {len(df.columns)} columns)")

def convert_flac(path_root: str, csv_output: str) -> pd.DataFrame:
    records = []

    for flac_path in glob.glob(os.path.join(path_root, "**/*.flac"), recursive=True):
        try:
            data, sr = sf.read(flac_path)
            wav_path = flac_path.replace(".flac", ".wav")
            sf.write(wav_path, data, sr)

            flac_info = sf.info(flac_path)
            wav_info  = sf.info(wav_path)

            records.append({
                "flac_path"   : flac_path,
                "wav_path"    : wav_path,
                "format_flac" : flac_info.format,
                "format_wav"  : wav_info.format,
                "subtype_flac": flac_info.subtype,
                "subtype_wav" : wav_info.subtype,
                "sr"          : sr,
                "duration_s"  : round(flac_info.duration, 3),
                "size_flac_kb": round(os.path.getsize(flac_path) / 1024, 2),
                "size_wav_kb" : round(os.path.getsize(wav_path)  / 1024, 2),
            })

        except Exception as e:
            print(f"{flac_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(csv_output, index=False)
    print(f"\nCSV saved -> {csv_output}")

    return df

def clear_flac(path_root: str):
    for path in glob.glob(os.path.join(path_root, "**/*.flac"), recursive=True):
        try:
            os.remove(path)
            print(f"Deleted     : {path}")
        except Exception as e:
            print(f"Error {path}: {e}")
            
def pecah_csv(file, n):
    p = Path(file)
    out = p.parent / f"{p.stem}_pecahan"
    out.mkdir(exist_ok=True)
 
    duplikat = shutil.copy2(file, p.parent / f"{p.stem}_duplikat.csv")
 
    with open(duplikat, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        baris = list(reader)
 
    per_file = ceil(len(baris) / n)
 
    for i in range(n):
        potongan = baris[i * per_file : (i + 1) * per_file]
        if not potongan:
            break
        nama = out / f"{p.stem}_part{i+1}.csv"
        with open(nama, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([header] + potongan)
        print(f"{nama.name}={len(potongan):,} baris")
 
    return out