import os
import json
import joblib
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import(
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

FEATURE_DIR = "data/features_v1"
MODEL_DIR = "models_mlp"
REPORT_DIR = "reports_mlp"

def main():

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    X_test = np.load(
        os.path.join(FEATURE_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(FEATURE_DIR, "y_test.npy")
    )

    model = load_model(
        os.path.join(
            MODEL_DIR,
            "mlp.keras"
        )
    )

    label_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    y_test_encoded = label_encoder.transform(
        y_test
    )

    predictions = model.predict(
        X_test
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    macro_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="macro"
    )

    print(
        f"Macro F1 Score: {macro_f1:.4f}"
    )

    with open(
        os.path.join(
            REPORT_DIR,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            {"macro_f1": float(macro_f1)},
            f,
            indent=4
        )

    cm = confusion_matrix(
        y_test_encoded,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

if __name__ == "__main__":
    main()    
