"""
This module provides utility functions for machine learning model
training, evaluation, and preprocessing.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from IPython.display import display
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

def rescale_function(X: Union[pd.Series,
np.ndarray]) -> Union[pd.Series, np.ndarray]:
  """
  Rescales the input data by adding 1 to each element.

  Args:
    X: A pandas Series or NumPy array containing the data to be rescaled.

  Returns:
    A pandas Series or NumPy array with the same shape as the input,
    where each element has been incremented by 1.
  """
  return X + 1

def evaluate_models(
    models: List,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    scoring: str = "recall",
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Evaluates multiple models using cross-validation and returns a DataFrame
    of results.

    Args:
      models: A list of trained machine learning models.
      X_train: The training feature matrix as a Pandas DataFrame.
      y_train: The training target variable array as a Pandas Series.
      cv: Cross-validation strategy (e.g., KFold, StratifiedKFold).
      scoring: The scoring metric to use (e.g., 'recall', 'roc_auc').
               Defaults to 'recall'.
      n_jobs: The number of jobs to run in parallel. Defaults to 8.

    Returns:
      A Pandas DataFrame containing the cross-validation results
       for each model.
    """
    results = [
        pd.Series(
            cross_validate(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )["test_score"],
            name=f"{model.__class__.__name__} class predictor",
        )
        for model in models
    ]

    results_df = pd.concat(results, axis=1)
    return results_df


def undersample_data(
    X: pd.DataFrame, y: pd.Series, random_state: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Undersamples the majority class in a dataset to balance class distribution.

    Args:
      X: The feature matrix.
      y: The target variable array.
      random_state: An integer to control the randomness of the undersampling.

    Returns:
      A tuple containing the resampled feature matrix
      and target variable array.
    """
    undersampler = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    return X_resampled, y_resampled


def preprocess_data(
    X: pd.DataFrame, pipeline: Pipeline, pipelined_features: list
) -> pd.DataFrame:
    """
    Applies a pipeline to transform data and converts it to a DataFrame.

    Args:
      X: The data to be preprocessed (pandas DataFrame).
      pipeline: The pipeline to apply (scikit-learn Pipeline).
      pipelined_features: The names of the features after pipeline
      transformation (list of strings).

    Returns:
      A DataFrame with the transformed data.
    """
    X_pipelined = pipeline.fit_transform(X)
    X_pipelined = pd.DataFrame(X_pipelined)
    X_pipelined.columns = pipelined_features
    return X_pipelined


def predict_bmi(X: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts missing 'bmi' values in a DataFrame using a
    DecisionTreeRegressor.

    Args:
        X (pd.DataFrame): The DataFrame containing the 'bmi'
        and other features.
    Returns:
        pd.DataFrame: The DataFrame with imputed 'bmi' values.
    """

    X = X.copy()
    X = X.reset_index(drop=True)

    encoder = OrdinalEncoder()
    X["gender"] = encoder.fit_transform(X[["gender"]])

    missing_bmi = X[X["bmi"].isna()]
    X_train = X[~X["bmi"].isna()]

    X_train_features = X_train[["age", "gender"]]
    y_train = X_train["bmi"]
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_features, y_train)

    X_missing = missing_bmi[["age", "gender"]]
    predicted_bmi = model.predict(X_missing)

    X.loc[missing_bmi.index, "bmi"] = predicted_bmi

    return X


def create_interactions(X: np.ndarray) -> np.ndarray:
    """Creates interaction terms from a NumPy array.

    Assumes the following column order (0-indexed):
    0: age_binned
    1: gender
    2: hypertension
    3: heart_disease
    4: ever_married
    5: work_type
    6: Residence_type
    7: smoking_status
    8: age
    9: avg_glucose_level
    10: bmi

    The function generates the following interaction terms:
    - age * gender
    - bmi * gender
    - hypertension * age_binned
    - heart_disease * age_binned
    - smoking_status * age_binned
    - avg_glucose_level * age

    Args:
      X: A NumPy array containing the features.

    Returns:
      A NumPy array with the added interaction terms.
    """
    X = X.copy()

    X = np.c_[X, X[:, 8] * X[:, 1]]
    X = np.c_[X, X[:, 10] * X[:, 1]]
    X = np.c_[X, X[:, 2] * X[:, 0]]
    X = np.c_[X, X[:, 3] * X[:, 0]]
    X = np.c_[X, X[:, 7] * X[:, 0]]
    X = np.c_[X, X[:, 9] * X[:, 8]]

    return X


def check_df(dataframe: pd.DataFrame, head: int = 2,
             transpose: bool = True) -> None:
    """
    Prints a comprehensive summary of a Pandas DataFrame, including shape,
    data types,
    a preview of the first and last few rows, null value counts, quantile
    statistics,
    and information on duplicate rows.

    Args:
        dataframe: The DataFrame to be analyzed.
        head: The number of rows to display from the beginning and end of the
        DataFrame
              (default: 5).
        transpose: If True, transposes the quantile output for better
        readability
                   (default: True).
    """

    print("############## Shape ##############")
    display(dataframe.shape)

    print("\n############## Quantiles ##############")
    quantiles = dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1])
    if transpose:
        display(quantiles.T)
    else:
        display(quantiles)

    print("\n############## Types ##############")
    [
        print(f"{col:>{20}}: {dtype}")
        for col, dtype in zip(dataframe.columns, dataframe.dtypes)
    ]

    print("\n############## Head ##############")
    display(dataframe.head(head))

    print("\n############## Tail ##############")
    display(dataframe.tail(head))

    print("\n############## NA ##############")
    display(dataframe.isnull().sum())

    print("\n############## Duplicate Rows ##############")
    duplicates_exist = dataframe.duplicated().any()
    if duplicates_exist:
        num_duplicates = dataframe.duplicated().sum()
        print(f"DataFrame contains {num_duplicates} duplicate rows.")

        duplicate_rows = dataframe[dataframe.duplicated(keep=False)]
        sorted_duplicate_rows = (
            duplicate_rows.sort_values(by=list(dataframe.columns)))
        print("\nPreview of duplicate rows (sorted):")
        display(sorted_duplicate_rows.head())

    else:
        print("No duplicate rows found in the DataFrame.")
