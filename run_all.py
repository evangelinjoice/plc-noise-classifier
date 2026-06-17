"""
run_all.py  --  One script to build the entire project from scratch.
Run: python run_all.py
"""
import subprocess, sys, os

def run(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or os.getcwd())
    if result.returncode != 0:
        print(f"ERROR running: {cmd}")
        sys.exit(1)

print("=" * 55)
print("  PLC Noise Classifier -- Full Build Pipeline")
print("=" * 55)

run("python data/generate_signals.py")
run("python dsp/feature_extraction.py")
run("python ml/train_model.py")
run("python ml/evaluate_model.py")

print("\n" + "=" * 55)
print("  All done! Starting dashboard...")
print("  Open your browser at: http://127.0.0.1:5000")
print("=" * 55 + "\n")

run("python dashboard/app.py")
