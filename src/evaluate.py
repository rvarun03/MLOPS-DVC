import os
import json
import joblib
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

FEATURE_DIR = "data/features_v1"
MODEL_DIR = "models"
REPORT_DIR = "reports"

def main():
    os.makedirs(REPORT_DIR,exist_ok=True)

    model=joblib.load(
        os.path.join(MODEL_DIR, "svm.pkl")
    )

    encoder=joblib.load(
        os.path.join(MODEL_DIR,"label_encoder.pkl")
    )

    X_test=np.load(
        os.path.join(FEATURE_DIR,"X_test.npy")
    )

    y_test=np.load(
        os.path.join(FEATURE_DIR, "y_test.npy")
    )

    y_test_encoded = encoder.transform(y_test)

    y_pred= model.predict(X_test)

    cm = confusion_matrix(
        y_test_encoded,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.savefig(
        os.path.join(REPORT_DIR, "confusion_matrix.png")
    )

    plt.close()
    
    macro_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="macro"
    )

    print(f"Macro F1 Score: {macro_f1:.4f}")

    metrics = {
        "macro_f1": float(macro_f1)
    }

    with open(
        os.path.join(REPORT_DIR, "metrics.json"),
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
