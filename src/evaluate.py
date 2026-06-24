import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_model():

    # 1. Load model
    model = joblib.load("models/final_model.pkl")

    # 2. Load dataset
    df = pd.read_excel("data/dataset_2.xlsx")

    # 3. Features
    features = [
        "norm_X_5kHz",
        "norm_X_100kHz",
        "norm_θ2_100kHz",
        "norm_R_5kHz/R_100kHz",
    ]

    target = "Fatigue_Class"

    # 4. Preprocessing
    df_model = df.dropna(subset=features + [target]).copy()

    X = df_model[features]
    y = df_model[target]

    # 5. Prediction
    y_pred = model.predict(X)

    # 6. Metrics
    print("ACCURACY:")
    print(accuracy_score(y, y_pred))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y, y_pred))

    # 7. Confusion matrix
    labels = ["Nízká", "Střední", "Vysoká"]

    cm = confusion_matrix(y, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False
    )

    ax.set_title("Confusion Matrix - Model Evaluation")

    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=300)
    plt.show()

    print("Evaluation finished.")
