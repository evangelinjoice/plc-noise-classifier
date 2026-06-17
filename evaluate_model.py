import pandas as pd
import numpy as np
import pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def evaluate(features_csv="data/features.csv", model_path="ml/saved_model/noise_classifier.pkl"):
    df = pd.read_csv(features_csv)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipeline = bundle["pipeline"]
    le = bundle["label_encoder"]
    y_enc = le.transform(y)
    _, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2,
                                             random_state=42, stratify=y_enc)
    y_pred = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix -- PLC Noise Classifier", fontweight='bold')
    plt.tight_layout()
    os.makedirs("ml/plots", exist_ok=True)
    plt.savefig("ml/plots/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved -> ml/plots/confusion_matrix.png")
    feat_importances = pipeline.named_steps["clf"].feature_importances_
    feat_names = bundle["feature_names"]
    sorted_idx = np.argsort(feat_importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_names[i] for i in sorted_idx[::-1]],
             feat_importances[sorted_idx[::-1]], color="#378ADD")
    ax.set_title("Top 10 Feature Importances", fontweight='bold')
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("ml/plots/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance saved -> ml/plots/feature_importance.png")

if __name__ == "__main__":
    evaluate()
