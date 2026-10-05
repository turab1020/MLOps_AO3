"""Stage 1 - download Fashion-MNIST and save the raw arrays to data/raw/."""
import os

import numpy as np
import tensorflow as tf

RAW_DIR = os.path.join("data", "raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    np.save(os.path.join(RAW_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(RAW_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RAW_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(RAW_DIR, "y_test.npy"), y_test)

    print(f"Saved raw data to {RAW_DIR}: train {x_train.shape}, test {x_test.shape}")


if __name__ == "__main__":
    main()
