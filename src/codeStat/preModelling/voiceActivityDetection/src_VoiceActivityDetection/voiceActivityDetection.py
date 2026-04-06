import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-GUI backend



def voice_activity_detection(
    quantized  : np.ndarray,
    sr         : int,
    frame_ms   : int = 300,     # durasi frame: 300 ms menghasilkan 480 sampel (dokumen)
    snr_min    : float = 6.0,   # ambang batas bawah  
    snr_max    : float = 10.0,  # ambang batas atas   
) -> tuple[np.ndarray, list[dict]]:

    #Bagi sinyal menjadi frame 
    frame_size = int(sr * frame_ms / 1000)      # 16000 * 300/1000 = 480
    frames     = [
        quantized[i : i + frame_size]
        for i in range(0, len(quantized), frame_size)
        if len(quantized[i : i + frame_size]) == frame_size  # buang frame tidak lengkap
    ]

    frame_results = []
    voice_frames  = []

    for idx, frame in enumerate(frames):

        # Step 1: Hitung Energi per Frame 
        # Dokumen: "Setiap nilai amplitudo diskrit dipangkatkan dan ditotalkan"
        energi = np.sum(frame ** 2)

        # Step 2: Perhitungan Daya 
        N    = len(frame)
        daya = energi / N

        # Step 3: Estimasi Daya 
        delta         = 1 / 32_767
        estimasi_daya = (delta ** 2) / 12

        # Step 4: Rasio Signal to Noise
        snr = daya / estimasi_daya if estimasi_daya > 0 else 0

        # Step 5: Konversi SNR ke Logaritmik (dB) 
        snr_db = 10 * np.log10(snr) if snr > 0 else 0

        # Step 6: Perbandingan terhadap Ambang Batas 
        if snr_db > snr_max:
            label = "wicara"
            voice_frames.append(frame)
        elif snr_db < snr_min:
            label = "noise"
        else:
            label = "tidak pasti"

        frame_results.append({
            "frame_idx"    : idx,
            "energi"       : round(energi, 10),
            "daya"         : round(daya, 10),
            "estimasi_daya": round(estimasi_daya, 10),
            "snr"          : round(snr, 4),
            "snr_db"       : round(snr_db, 10),
            "label"        : label,
        })
    # Gabungkan semua frame wicara menjadi satu sinyal wicara utuh
    sinyal_bersih = np.concatenate(voice_frames) if voice_frames else np.array([])

    return sinyal_bersih, frame_results