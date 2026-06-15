import os 
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

FEATURE_DIR = "data/features_v1"
MODEL_DIR = "models_mlp"

def main():

    os.makedirs(MODEL_DIR,exist_ok=True)

    X_train = np.load(
        os.path.join(FEATURE_DIR, "X_train.npy")
    )

    y_train = np.load(
        os.path.join(FEATURE_DIR, "y_train.npy")
    )

    # encode labels

    label_encoder=LabelEncoder()

    y_train_encoded=label_encoder.fit_transform(
        y_train
    )

    num_classes=len(label_encoder.classes_)

    y_train_categorical=to_categorical(
        y_train_encoded,
        num_classes=num_classes
    )

    model = Sequential([
        Dense(
            512,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),

        Dropout(0.3),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train

    model.fit(
        X_train,
        y_train_categorical,
        epochs=20,
        batch_size=32,
        validation_split=0.1
    )

    # Save model

    model.save(
        os.path.join(MODEL_DIR, "mlp.keras")
    )

    joblib.dump(
        label_encoder,
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    print("MLP trained successfully")


if __name__ == "__main__":
    main()    