import pandas as pd

def load_data():
    """
    Loads the features (X) and target (y) data from CSV files.
    Returns two pandas DataFrames: X_data, y_data.
    """
    
    features_path = 'exampleData/lucas_organic_carbon_training_and_test_data.csv'
    target_path = 'exampleData/lucas_organic_carbon_target.csv'
    X_data = pd.read_csv(features_path)
    y_data = pd.read_csv(target_path)
    return X_data, y_data
