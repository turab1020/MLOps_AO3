"""Stage 4 - evaluate the trained model on the test set and write metrics.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")  # draw to a file, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")
CM_PATH = os.path.join("models", "confusion_matrix.png")
METRICS_PATH = "metrics.json"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Fashion-MNIST test set (accuracy {accuracy:.4f})")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=120)

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(accuracy), 4)}
    with open(METRICS_PATH, "w") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")

    print(f"test_loss={metrics['test_loss']}  test_accuracy={metrics['test_accuracy']}")


if __name__ == "__main__":
    main()
