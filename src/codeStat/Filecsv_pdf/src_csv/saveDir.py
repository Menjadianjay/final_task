import os

def save_csv(df, folder, filename):
    os.makedirs("src\Filecsv_pdf\csv_files", exist_ok=True)
    df.to_csv(os.path.join(folder, filename), index=False)