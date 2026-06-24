import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model():

    # 1. Load dataset
    df = pd.read_excel("data/dataset_2.xlsx")

    # 2. Features and target
    features = [
        "norm_X_5kHz",
        "norm_X_100kHz",
        "norm_θ2_100kHz",
        "norm_R_5kHz/R_100kHz",
    ]

    target = "Fatigue_Class"

    # 3. Preprocessing
    df_model = df.dropna(subset=features + [target, "Subjekt_ID"]).copy()

    X = df_model[features]
    y = df_model[target]
    subjects = df_model["Subjekt_ID"]

    # 4. LOSO CV
    unique_subjects = subjects.unique()

    for test_subj in unique_subjects:

        train_idx = subjects != test_subj
        test_idx = subjects == test_subj

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt"
        )

        model.fit(X_train, y_train)

    # 5. Final model (trained on all data)
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt"
    )

    final_model.fit(X, y)

    # 6. Save model
    joblib.dump(final_model, "models/final_model.pkl")

    print("Model trained and saved successfully.")
