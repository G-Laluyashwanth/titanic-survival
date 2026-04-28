# Titanic — Survival Prediction (Kaggle)

Binary classification project predicting passenger survival on the Titanic.

**Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Project Structure

titanic/
├── data/
│   ├── raw/                # Kaggle data (download from competition page)
│   └── processed/          # Cleaned/transformed data
├── notebooks/
│   └── pipeline.ipynb      # Exploratory work
├── src/
│   ├── data.py             # Data loading and splitting
│   ├── pipeline.py         # build_pipeline() — sklearn Pipeline factory
│   ├── train.py            # End-to-end training + model persistence
│   └── predict.py          # Generate Kaggle submission from saved model
├── weights/                # Trained models (gitignored)
├── submission.csv
├── requirements.txt
└── README.md

## Setup

```bash
git clone https://github.com/G-Laluyashwanth/titanic.git
cd titanic
mkdir -p data/raw data/processed weights
pip install -r requirements.txt
# Download train.csv, test.csv, gender_submission.csv from Kaggle into data/raw/
```

## Usage

Train the model:
```bash
python -m src.train
```

Generate Kaggle submission:
```bash
python -m src.predict
```

## Approach

*Coming soon — project in progress.*

## Results

*Coming soon.*