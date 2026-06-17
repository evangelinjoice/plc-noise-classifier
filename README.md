# PLC Noise Classifier

A complete ECE project that simulates Power Line Communication (PLC) noise, extracts DSP features, classifies noise types using Machine Learning, and displays results on a live web dashboard -- all without requiring any hardware.

## About the Project

Power Line Communication (PLC) transmits data over existing electrical wiring. One of the biggest challenges in PLC systems is noise interference. This project:

- Synthetically generates 3 real-world PLC noise types
- Applies FFT-based signal processing to extract meaningful features
- Trains a Random Forest classifier to identify noise types
- Displays everything on a live Flask web dashboard with real-time charts

Hardware note: This project currently runs in simulation mode. Hardware integration (STM32 + coupling circuit) is planned as a future extension.

## Noise Types Classified

| Noise Type | Description |
|---|---|
| Impulsive | Sudden high-amplitude spikes (e.g. from switching appliances) |
| Narrowband | Single-frequency interference (e.g. from radio signals) |
| Periodic Async | Multi-tone interference not synced to 50/60 Hz mains |

## Tech Stack

| Layer | Tools |
|---|---|
| Signal Simulation | Python, NumPy |
| DSP / Feature Extraction | NumPy, SciPy, Matplotlib |
| Machine Learning | scikit-learn (Random Forest) |
| Web Dashboard | Flask, Chart.js |
| Data Handling | Pandas |

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/plc-noise-classifier.git
cd plc-noise-classifier
pip install -r requirements.txt
python run_all.py
```

Then open: http://127.0.0.1:5000

## Results

- Model: Random Forest (150 estimators)
- Cross-validation accuracy: ~99%
- Top features: Kurtosis, Crest Factor, Peak Frequency, Spectral Centroid

## Author

Eva -- 3rd Year ECE Student
NPTEL Certified -- Database Management Systems (IIT Kharagpur, Elite)

## License

MIT License
