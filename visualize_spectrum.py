import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsp.feature_extraction import extract_features

SAMPLE_RATE = 10000

def plot_signal_analysis(signal, label="unknown", out_path="dsp/plots/analysis.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    signal = np.array(signal, dtype=float)
    t = np.linspace(0, len(signal)/SAMPLE_RATE, len(signal)) * 1000
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/SAMPLE_RATE)
    magnitude = np.abs(fft_vals)
    feats = extract_features(signal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Signal Analysis -- Class: {label}", fontsize=13, fontweight='bold')
    axes[0].plot(t, signal, color="#378ADD", linewidth=0.9)
    axes[0].set_title("Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    info = (f"RMS={feats['rms']:.3f}  Peak={feats['peak']:.3f}\n"
            f"Kurtosis={feats['kurtosis']:.2f}  ZC={int(feats['zero_crossings'])}")
    axes[0].text(0.98, 0.97, info, transform=axes[0].transAxes,
                 fontsize=9, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    axes[1].plot(freqs, magnitude, color="#1D9E75", linewidth=0.9)
    axes[1].set_title("Frequency Spectrum (FFT)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(feats['peak_freq'], color='red', linestyle='--', linewidth=1,
                    label=f"Peak: {feats['peak_freq']:.0f} Hz")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/sample_data.csv")
    for cls in df["label"].unique():
        row = df[df["label"] == cls].iloc[0]
        signal = row.drop("label").values.astype(float)
        plot_signal_analysis(signal, label=cls, out_path=f"dsp/plots/{cls}.png")
