# Titanic — Survival Prediction (Kaggle)

Binary classification project predicting passenger survival on the Titanic.

**Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Project Structure

```
titanic/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── pipeline.ipynb
├── src/
│   ├── data.py
│   ├── pipeline.py
│   ├── train.py
│   └── predict.py
├── weights/
├── submission.csv
├── requirements.txt
└── README.md
```

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

This project follows a structured end-to-end machine learning workflow:

- **Data Cleaning**
  - Handled missing values in `Age`, `Embarked`, and `Fare`
  - Dropped less useful features like `Cabin` due to high missingness

- **Feature Engineering**
  - Extracted titles from passenger names (Mr, Mrs, Miss, etc.)
  - Converted categorical variables using encoding techniques
  - Created family-related features (e.g., family size)

- **Exploratory Data Analysis (EDA)**
  - Analyzed survival distribution across gender, class, and age
  - Identified key patterns (e.g., higher survival rate for females and 1st class passengers)

- **Pipeline Development**
  - Built a reusable ML pipeline using `scikit-learn`
  - Included preprocessing steps (imputation, encoding, scaling)
  - Ensured clean and modular workflow for training and inference

- **Modeling**
  - Trained classification models (e.g., Logistic Regression / Random Forest)
  - Evaluated performance using validation techniques

- **Prediction**
  - Generated predictions on test data
  - Created submission file for Kaggle evaluation

## Results

- **Kaggle Public Score:** `0.75837`
- Built a complete end-to-end ML pipeline from data preprocessing to prediction
- Established a strong baseline model for further improvements