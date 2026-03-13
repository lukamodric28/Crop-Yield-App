# Crop Yield Predictor

Streamlit app that predicts crop yield (hg/ha) using six trained regression models. Users enter climate inputs, select a country and crop, and choose a model to generate a prediction with an uncertainty range when the year is outside the training window.

## Features

- Predict yield using Lasso, Polynomial, Random Forest, Gradient Boosting, K-Nearest Neighbors, or Support Vector Regression.
- Input fields for rainfall, average temperature, pesticides used, year, country, and crop.
- Prediction output with optional uncertainty and a comparison to global benchmark ranges.
- Statistics page with model evaluation metrics and model descriptions.
- Datasets page showing full, training, and testing data.
- Graphs page with bar charts, scatter plots, and residual plots.

## Project structure

- [app.py](app.py): Streamlit app UI and prediction logic.
- [models_and_datasets](models_and_datasets): Trained models and CSV datasets used by the app.
- [function_transformers](function_transformers): Feature engineering utilities used during model training.

## Requirements

Install Python 3.9+ and the dependencies in [requirements.txt.txt](requirements.txt.txt).

## Run the app

Choose one of the options below.

### Option 1: Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Create a new app.
3. Point it to this repository so Streamlit can host it.

### Option 2: Run locally with VS Code

1. Clone or download this repository.
2. Install the required Python packages.
3. Open the project in VS Code.
4. Run the app in the integrated terminal:

```bash
streamlit run app.py
```

## Data source

Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

