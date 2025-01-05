from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import seaborn as sns

def filter_classes(y_test, y_pred, selected_classes):
    """
    Returns filtered y_test and y_pred, keeping only rows where y_test 
    is in selected_classes. Also converts y_pred to a Series with the same index.
    NOTE: Assumes the target column is named 'x' if y_test is a DataFrame.
    """
    # Convert y_test to a Series if it's a DataFrame
    if isinstance(y_test, pd.DataFrame):
        # Assuming the single column in your target DataFrame is named 'x'
        y_test_series = y_test['x']
    else:
        y_test_series = y_test

    # Convert y_pred to a Series with same index
    y_pred_series = pd.Series(y_pred, index=y_test_series.index)

    # Keep only the rows for which y_test is in selected_classes
    mask = y_test_series.isin(selected_classes)
    y_test_filtered = y_test_series[mask]
    y_pred_filtered = y_pred_series[mask]

    return y_test_filtered, y_pred_filtered

def substitute_and_calculate_accuracy(feature, replacement_value, model, X, y):
    """
    Replaces values of the specified feature with a given value and calculates the accuracy.

    Parameters:
    - feature: str, the name of the feature to replace.
    - replacement_value: value to replace the feature with.
    - model: trained RandomForestClassifier.
    - X: DataFrame, feature set.
    - y: Series, true labels.

    Returns:
    - Accuracy score after substitution.
    """

    X_modified = X.copy()
    X_modified[feature] = replacement_value
    y_pred_modified = model.predict(X_modified)

    return accuracy_score(y, y_pred_modified)

def calculate_interval_impact(feature, intervals, model, X, y):
    """
    Calculates how replacing feature values within each interval by the 
    midpoint of that interval affects model accuracy.
    """
    impacts = []
    for start, end in intervals:
        X_modified = X.copy()
        if feature in X_modified.columns:
            midpoint = (start + end) / 2
            X_modified[feature] = X_modified[feature].apply(
                lambda x: midpoint if (start <= x <= end) else x
            )
            y_pred_modified = model.predict(X_modified)
            impact = accuracy_score(y, y_pred_modified)
            impacts.append(impact)
        else:
            print(f"Feature '{feature}' not found in the dataset.")
            return []
    return impacts



def get_single_feature_importance(rf_model, X_data, feature_name):
    """
    Returns the feature importance score for a single feature from a trained RandomForest model.
    """
    if feature_name not in X_data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
    
    feature_index = list(X_data.columns).index(feature_name)
    return rf_model.feature_importances_[feature_index]
