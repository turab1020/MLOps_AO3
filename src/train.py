"""Stage 3 - build and train the ANN, save the model and its training history."""
import os

import numpy as np
import tensorflow as tf
import yaml

DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"


def build_model(dense_units, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    tf.keras.utils.set_random_seed(params["seed"])

    x_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(DATA_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    model = build_model(params["dense_units"], params["dropout_rate"], params["learning_rate"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    history_log = tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, "history.csv"))

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[history_log],
        verbose=2,
    )

    model.save(os.path.join(MODEL_DIR, "model.h5"))


if __name__ == "__main__":
    main()
