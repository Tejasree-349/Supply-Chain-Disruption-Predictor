# Supply Chain Disruption and Failure Predictor

## Overview

A machine-learning project designed to estimate late-delivery risk from historical supply-chain order data.

The project analyzes operational patterns, prepares data for machine learning, trains classification models, evaluates predictions, and translates risk estimates into potential fulfillment actions through an interactive Streamlit dashboard.

## Objectives

* Analyze historical supply-chain data.
* Identify patterns associated with late-delivery risk.
* Preprocess data and engineer relevant features.
* Train and compare Logistic Regression and Random Forest models.
* Evaluate model performance using precision, recall, F1-score, ROC-AUC, and average precision.
* Explain model behavior using permutation feature importance.
* Provide an interactive dashboard for data exploration, model evaluation, and predictions.
* Estimate late-delivery risk for new orders using compatible input data.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Pytest

## Project Structure

* `data/`: Raw and processed supply-chain data.
* `notebooks/`: Exploratory data analysis and visualizations.
* `src/`: Data validation, preprocessing, feature engineering, model training, evaluation, prediction, and explainability.
* `models/`: Locally trained machine-learning model artifacts.
* `outputs/`: Evaluation reports and visualizations.
* `app/`: Streamlit dashboard.
* `tests/`: Automated tests for data validation and prediction.
* `main.py`: Main project entry point.
* `requirements.txt`: Required Python dependencies.
* `.gitignore`: Specifies files and directories excluded from Git tracking.
* `README.md`: Project documentation and setup instructions.

## Prerequisites

* Python 3.11 or another version compatible with the project dependencies.
* Git.
* The supply-chain dataset required by the project.

## Setup

Clone the repository:

```bash
git clone https://github.com/Tejasree-349/Supply-Chain-Disruption-Predictor.git
cd Supply-Chain-Disruption-Predictor
```

Create a virtual environment:

```
python -m venv venv
```

Activate the virtual environment.

**Windows PowerShell:**

```
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```
venv\Scripts\activate.bat
```

**Linux/macOS:**

```
source venv/bin/activate
```

Install dependencies:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the dataset at:

```
data/raw/DataCoSupplyChainDataset.csv
```

Ensure the dataset filename and path match the project configuration. If the dataset is not included in the repository, obtain it separately before running the project.

## Run Data Validation

Validate the dataset before training the models.

```
python -m src.data_validation
```

## Train Models

Train the Logistic Regression and Random Forest models and generate the model artifacts configured by the project.

```
python -m src.train
```

## Evaluate Models

Evaluate the trained models using the configured classification metrics and generate evaluation outputs.

```
python -m src.evaluate
```

## Explain Model Predictions

Run permutation feature-importance analysis to understand which input features influence model predictions.

```
python -m src.explainability
```

## Run the Dashboard

Launch the interactive Streamlit application.

```
python -m streamlit run app/app.py
```

After the application starts, open the local URL displayed in the terminal to access the dashboard.

The dashboard provides dataset insights, model performance comparisons, prediction analysis, and an interface for scoring compatible CSV files.

## Prediction Lab

To predict late-delivery risk for new orders:

* Prepare a CSV file containing the input features expected by the trained model.
* Ensure the column names and data types match the model's requirements.
* Do not include the target column in the uploaded input data.
* Open the Prediction Lab section in the dashboard.
* Upload the CSV file and run the prediction operation.
* Review the predicted classes, risk probabilities, and risk categories.

The uploaded data must follow the model's expected input schema. Matching column names alone may not be sufficient if feature types or preprocessing requirements differ.

## Testing

Run the automated test suite:

```
python -m pytest -v
```

This executes the project's tests and reports their results.

## Running the Project from Scratch

For a new user setting up the project, follow this order:

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Place the dataset in `data/raw/`.
5. Run data validation.
6. Train the models.
7. Evaluate model performance.
8. Generate feature-importance analysis.
9. Launch the Streamlit dashboard.
10. Run the automated tests.

If trained model files are not included in the repository, training must be completed before using the dashboard's prediction functionality.

## Important Limitations

Model performance depends on data quality, preprocessing, feature selection, and the timing of the input features.

Columns containing delivery outcomes or information unavailable at prediction time must not be used as predictors, as they can introduce data leakage.

The current implementation is a starting point for supply-chain risk analysis. Chronological validation, threshold calibration, feature review, and additional operational testing are required before deployment.

Classification thresholds should be selected according to operational requirements and the costs of missed delays and false alarms.

Predictions indicate estimated risk, not guaranteed outcomes.
