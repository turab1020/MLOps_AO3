"""Stage 2 - normalize the pixels and split a validation set from the training data."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")


def normalize(x):
    return (x.astype("float32") / 255.0 - 0.2860) / 0.3530  # scale to [0, 1], then standardize with the dataset mean and std


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    x_train = normalize(np.load(os.path.join(RAW_DIR, "x_train.npy")))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = normalize(np.load(os.path.join(RAW_DIR, "x_test.npy")))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUT_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    print(f"Saved to {OUT_DIR}: train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")


if __name__ == "__main__":
    main()
