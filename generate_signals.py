import numpy as np
import pandas as pd
import os

np.random.seed(42)

SAMPLE_RATE = 10000
DURATION = 0.1
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
NUM_PER_CLASS = 200

def generate_impulsive(n):
    signals = []
    for _ in range(n):
        sig = np.random.normal(0, 0.1, NUM_SAMPLES)
        num_impulses = np.random.randint(3, 10)
        positions = np.random.randint(0, NUM_SAMPLES, num_impulses)
        amplitudes = np.random.uniform(2, 5, num_impulses)
        for pos, amp in zip(positions, amplitudes):
            sig[pos] += amp * np.random.choice([-1, 1])
        signals.append(sig)
    return signals

def generate_narrowband(n):
    signals = []
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    for _ in range(n):
        freq = np.random.uniform(1000, 4000)
        amp = np.random.uniform(0.5, 2.0)
        sig = amp * np.sin(2 * np.pi * freq * t)
        sig += np.random.normal(0, 0.05, NUM_SAMPLES)
        signals.append(sig)
    return signals

def generate_periodic_async(n):
    signals = []
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    for _ in range(n):
        f1 = np.random.uniform(200, 600)
        f2 = np.random.uniform(800, 1500)
        a1 = np.random.uniform(0.3, 1.0)
        a2 = np.random.uniform(0.3, 1.0)
        sig = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
        sig += np.random.normal(0, 0.08, NUM_SAMPLES)
        signals.append(sig)
    return signals

if __name__ == "__main__":
    print("Generating synthetic PLC noise signals...")
    imp = generate_impulsive(NUM_PER_CLASS)
    narr = generate_narrowband(NUM_PER_CLASS)
    per = generate_periodic_async(NUM_PER_CLASS)
    all_signals = imp + narr + per
    labels = (["impulsive"] * NUM_PER_CLASS +
              ["narrowband"] * NUM_PER_CLASS +
              ["periodic_async"] * NUM_PER_CLASS)
    df = pd.DataFrame(all_signals, columns=[f"t{i}" for i in range(NUM_SAMPLES)])
    df.insert(0, "label", labels)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_data.csv", index=False)
    print(f"Saved data/sample_data.csv ({len(df)} samples)")
