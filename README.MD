# Titanic — Survival Prediction (Kaggle)

Binary classification project predicting passenger survival on the Titanic.

**Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Project Structure

```
titanic-survival/
├── data/
│   ├── raw/                # Kaggle data (download from competition page)
│   └── processed/          # Cleaned/transformed data
├── notebooks/
│   ├── 01-explore.ipynb         # EDA
│   └── 02-logistic-regression.ipynb   # Logistic Regression experiments
├── src/
│   ├── data.py             # Data loading + train/val splitting
│   ├── pipeline.py         # build_pipeline() — supports multiple models
│   ├── train.py            # End-to-end training, with --model flag
│   └── predict.py          # Generate Kaggle submission from saved model
├── weights/                # Trained models (gitignored)
├── submission.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/G-Laluyashwanth/titanic-survival.git
cd titanic-survival
mkdir -p data/raw data/processed weights
pip install -r requirements.txt
# Download train.csv, test.csv from Kaggle into data/raw/
```

## Usage

Train a model:
```bash
python -m src.train --model random_forest
python -m src.train --model logistic
```

Generate submission:
```bash
python -m src.predict --model random_forest
python -m src.predict --model logistic
```

## Approach

End-to-end scikit-learn workflow with two classification models compared on the same dataset.

**Preprocessing pipeline:**
- `SimpleImputer` (median) for numeric features
- `SimpleImputer` (constant `"missing"`) + `OneHotEncoder` for categorical features
- `StandardScaler` added for Logistic Regression (linear models need scaled features)
- Dropped: `PassengerId`, `Name`, `Ticket`, `Cabin` (high cardinality / heavily missing)

**Models compared:**
- `RandomForestClassifier` (100 trees, no scaling needed)
- `LogisticRegression` with L2 regularisation

**Evaluation:**
- Held-out validation split (20%)
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Coefficient interpretation for Logistic Regression

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8212 | 0.7917 | 0.7703 | 0.7808 | 0.8944 |
| Logistic Regression | 0.8101 | 0.7857 | 0.7432 | 0.7639 | 0.8824 |

**Kaggle public score (Random Forest):** 0.75837

**Key insight:** Logistic Regression had slightly lower accuracy but produced more interpretable coefficients. The strongest learned features were `Sex_male` (-1.33) and `Sex_female` (+1.32), confirming sex was the dominant survival factor — followed by passenger class.

## Next Steps

- [ ] Feature engineering: extract titles from `Name` (Mr/Mrs/Miss/Master), create `family_size` from SibSp + Parch, bin `Age`
- [ ] Try gradient boosting (XGBoost / LightGBM)
- [ ] Hyperparameter tuning with `RandomizedSearchCV`
- [ ] Deploy best model as a FastAPI service