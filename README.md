# Supply Chain Disruption and Failure Predictor

## Overview

A machine-learning project designed to estimate late-delivery risk
from historical supply-chain order data.

The project explores operational patterns, trains classification
models, evaluates predictions, and translates risk estimates into
potential fulfillment actions.

## Objectives

- Analyze historical supply-chain data.
- Identify patterns associated with late-delivery risk.
- Train and compare classification models.
- Evaluate performance using precision, recall, F1-score,
  ROC-AUC, and average precision.
- Explain model behavior using permutation importance.
- Provide a dashboard for model evaluation and predictions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pytest

## Project Structure

- `data/`: Raw and processed data.
- `notebooks/`: Exploratory analysis.
- `src/`: Data preparation, training, evaluation, and prediction.
- `models/`: Locally trained model artifacts.
- `outputs/`: Evaluation reports and visualizations.
- `app/`: Streamlit dashboard.
- `tests/`: Automated tests.

## Setup

Create and activate a virtual environment.

Install dependencies:

    python -m pip install -r requirements.txt

Place the dataset at:

    data/raw/DataCoSupplyChainDataset.csv

## Run Data Validation

    python -m src.data_validation

## Train Models

    python -m src.train

## Evaluate Models

    python -m src.evaluate

## Explain Model Predictions

    python -m src.explainability

## Run the Dashboard

    python -m streamlit run app/app.py

## Testing

    python -m pytest -v

## Important Limitations

Model performance depends on the quality and timing of the input
features. Columns containing delivery outcomes or information
unavailable at prediction time must not be used as predictors.

The current implementation is a starting point. Chronological
validation, threshold calibration, feature review, and additional
operational testing are required before deployment.

Predictions indicate estimated risk, not guaranteed outcomes.