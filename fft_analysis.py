import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

SAMPLE_RATE = 10000
NUM_SAMPLES = 1000

def compute_fft(signal, sample_rate=SAMPLE_RATE):
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sample_rate)
    magnitude = np.abs(fft_vals)
    return freqs, magnitude

def plot_sample_spectra(csv_path="data/sample_data.csv", out_dir="dsp/plots"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    classes = df["label"].unique()
    fig, axes = plt.subplots(len(classes), 2, figsize=(14, 4 * len(classes)))
    fig.suptitle("PLC Noise -- Time Domain & Frequency Spectrum", fontsize=14, fontweight='bold')
    colors = {"impulsive": "#D85A30", "narrowband": "#1D9E75", "periodic_async": "#7F77DD"}
    for i, cls in enumerate(classes):
        row = df[df["label"] == cls].iloc[0]
        signal = row.drop("label").values.astype(float)
        freqs, magnitude = compute_fft(signal)
        t = np.linspace(0, 0.1, NUM_SAMPLES)
        axes[i, 0].plot(t * 1000, signal, color=colors.get(cls, "steelblue"), linewidth=0.8)
        axes[i, 0].set_title(f"{cls} -- time domain")
        axes[i, 0].set_xlabel("Time (ms)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(freqs, magnitude, color=colors.get(cls, "steelblue"), linewidth=0.8)
        axes[i, 1].set_title(f"{cls} -- frequency spectrum")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Magnitude")
        axes[i, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "spectra.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrum plot -> {out_path}")

if __name__ == "__main__":
    plot_sample_spectra()
