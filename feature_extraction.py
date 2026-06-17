import numpy as np
import pandas as pd

SAMPLE_RATE = 10000

def extract_features(signal):
    signal = np.array(signal, dtype=float)
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-10)
    kurtosis = np.mean((signal - mean)**4) / (std**4 + 1e-10)
    skewness = np.mean((signal - mean)**3) / (std**3 + 1e-10)
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/SAMPLE_RATE)
    magnitude = np.abs(fft_vals)
    power = magnitude**2
    total_power = np.sum(power) + 1e-10
    spectral_centroid = np.sum(freqs * power) / total_power
    peak_freq = freqs[np.argmax(magnitude)]
    spectral_flatness = (np.exp(np.mean(np.log(magnitude + 1e-10))) /
                         (np.mean(magnitude) + 1e-10))
    low_mask  = freqs < 1000
    mid_mask  = (freqs >= 1000) & (freqs < 3000)
    high_mask = freqs >= 3000
    low_energy  = np.sum(power[low_mask])  / total_power
    mid_energy  = np.sum(power[mid_mask])  / total_power
    high_energy = np.sum(power[high_mask]) / total_power
    return {
        "mean": mean, "std": std, "rms": rms, "peak": peak,
        "crest_factor": crest_factor, "kurtosis": kurtosis,
        "skewness": skewness, "zero_crossings": zero_crossings,
        "spectral_centroid": spectral_centroid, "peak_freq": peak_freq,
        "spectral_flatness": spectral_flatness,
        "low_energy": low_energy, "mid_energy": mid_energy, "high_energy": high_energy,
    }

def extract_all_features(csv_path="data/sample_data.csv", out_path="data/features.csv"):
    print("Extracting features from signal data...")
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        label = row["label"]
        signal = row.drop("label").values.astype(float)
        feats = extract_features(signal)
        feats["label"] = label
        records.append(feats)
    feat_df = pd.DataFrame(records)
    feat_df.to_csv(out_path, index=False)
    print(f"Saved features -> {out_path}  ({len(feat_df)} rows, {len(feat_df.columns)-1} features)")
    return feat_df

if __name__ == "__main__":
    extract_all_features()
