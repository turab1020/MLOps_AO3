# fashion-ann-pipeline

A fully-connected neural network (ANN) that classifies Fashion-MNIST images into 10 clothing
categories. Code is versioned with Git, data and models with DVC, and the DVC remote is a
Google Drive folder.

## Pipeline

| Stage      | Script              | Output                                   |
|------------|---------------------|------------------------------------------|
| prepare    | `src/prepare.py`    | raw arrays in `data/raw/`                |
| preprocess | `src/preprocess.py` | train/val/test arrays in `data/processed/` |
| train      | `src/train.py`      | `models/model.h5`, `models/history.csv`  |
| evaluate   | `src/evaluate.py`   | `metrics.json`, confusion matrix image   |

All hyperparameters live in `params.yaml`.

## Run

```bash
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib
dvc pull     # fetch data and model from the Google Drive remote
dvc repro    # re-run only the stages whose inputs changed
```
