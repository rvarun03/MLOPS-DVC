import os
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

FEATURE_DIR = "data/features_v1"
MODEL_DIR = "models"

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train=np.load(
        os.path.join(FEATURE_DIR,"X_train.npy")
    )

    y_train = np.load(
        os.path.join(FEATURE_DIR, "y_train.npy")
    )

    encoder=LabelEncoder()

    y_train_encoded=encoder.fit_transform(y_train)

    model = SVC(
        kernel="rbf",
        C=1.0
    )

    model.fit(
        X_train,
        y_train_encoded
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "svm.pkl")
    )

    joblib.dump(
        encoder,
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    print("SVM trained successfully")


if __name__ == "__main__":
    main()