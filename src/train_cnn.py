import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


DATA_DIR = "data/processed_v1/train"

MODEL_DIR = "models_cnn"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------------------
    # Label Encoder
    # -------------------

    class_names = sorted(
        [
            d
            for d in os.listdir(DATA_DIR)
            if os.path.isdir(
                os.path.join(DATA_DIR, d)
            )
        ]
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(class_names)

    joblib.dump(
        label_encoder,
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    num_classes = len(class_names)

    # -------------------
    # Data Generator
    # -------------------

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1
    )

    train_data = train_gen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_data = train_gen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    # -------------------
    # MobileNetV2
    # -------------------

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    predictions = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(
        os.path.join(
            MODEL_DIR,
            "mobilenet.keras"
        )
    )

    print("CNN trained successfully")


if __name__ == "__main__":
    main()