"""Build the scikit-learn pipeline for Titanic survival classification.

Supports multiple model types (Random Forest, Logistic Regression).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X, model_type="random_forest", random_state=42):
    """
    Build a preprocessing + classification pipeline.

    Parameters
    ----------
    X : DataFrame
        The training features. Used to identify numeric vs categorical columns.
    model_type : str
        Either "random_forest" or "logistic". Determines which classifier is
        used and whether scaling is applied to numeric features.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Pipeline
        Untrained sklearn Pipeline ready to be fit on data.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Categorical path is the same for both models
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Numeric path differs by model type
    # Logistic Regression needs scaling; Random Forest does not
    if model_type == "logistic":
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Use 'random_forest' or 'logistic'."
        )

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return model