"""Data loading and splitting utilities."""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")


def split_features_and_targets(train, target_col="Survived", drop_cols = ("Survived", "PassengerId", "Name", "Ticket", "Cabin")):
    
    cols_to_drop = []

    for c in [target_col, *drop_cols]:
        if c in train.columns:
            cols_to_drop.append(c)


    X = train.drop(columns=cols_to_drop)
    y = train[target_col]

    return X, y


def get_train_val_split(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val