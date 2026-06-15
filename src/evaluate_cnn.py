import os
import json
import joblib
import numpy as np

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


TEST_DIR = "data/processed_v1/test"

MODEL_DIR = "models_cnn"

REPORT_DIR = "reports_cnn"


def main():

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    model = load_model(
        os.path.join(
            MODEL_DIR,
            "mobilenet.keras"
        )
    )

    test_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    predictions = model.predict(
        test_data
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    y_true = test_data.classes

    macro_f1 = f1_score(
        y_true,
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
            {
                "macro_f1": float(macro_f1)
            },
            f,
            indent=4
        )

    cm = confusion_matrix(
        y_true,
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