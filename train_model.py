import pandas as pd
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def train(features_csv="data/features.csv", model_out="ml/saved_model/noise_classifier.pkl"):
    print("Loading features...")
    df = pd.read_csv(features_csv)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2,
                                                          random_state=42, stratify=y_enc)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=10,
                                        random_state=42, n_jobs=-1))
    ])
    print("Training Random Forest classifier...")
    pipeline.fit(X_train, y_train)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=5, scoring="accuracy")
    test_acc = pipeline.score(X_test, y_test)
    print(f"Cross-val accuracy : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    print(f"Test set accuracy  : {test_acc*100:.2f}%")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": le,
                     "feature_names": list(df.drop("label", axis=1).columns)}, f)
    print(f"Model saved -> {model_out}")
    return pipeline, le

if __name__ == "__main__":
    train()
