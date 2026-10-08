import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class DataPreprocessor:
    """
    Generic Data Preprocessor for tabular security logs.
    Cleans missing values, strips whitespace, handles column renaming,
    encodes labels, and performs train-test splitting.
    """
    def __init__(self, target_column='label'):
        self.target_column = target_column

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df):
        """Cleans null values, duplicates, and invalid numeric values."""
        df = df.dropna().copy()
        df = df.drop_duplicates().copy()
        
        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            
        return df

    def split_features_target(self, df, target_col=None):
        target = target_col or self.target_column
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe.")
            
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def prepare_train_test(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def chronological_split(self, df, time_column, test_size=0.2):
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1')
        if self.target_column not in df.columns or time_column not in df.columns:
            raise KeyError('Chronological split requires target and time columns.')

        train_indices = []
        test_indices = []
        for _, group in df.groupby(self.target_column, sort=True):
            ordered_indices = group.sort_values(time_column, kind='stable').index.tolist()
            split_at = int(len(ordered_indices) * (1 - test_size))
            if split_at < 1 or split_at >= len(ordered_indices):
                raise ValueError('Each class needs at least two rows for a chronological holdout.')
            train_indices.extend(ordered_indices[:split_at])
            test_indices.extend(ordered_indices[split_at:])

        train = df.loc[train_indices].sort_values(time_column, kind='stable').reset_index(drop=True)
        test = df.loc[test_indices].sort_values(time_column, kind='stable').reset_index(drop=True)
        return train, test

if __name__ == '__main__':
    preprocessor = DataPreprocessor(target_column='label')
    dataset_path = (
        config.CAN_TRAINING_DATA_PATH
        if os.path.exists(config.CAN_TRAINING_DATA_PATH)
        else config.CAN_RAW_DATA_PATH
    )
    df = preprocessor.load_data(dataset_path)
    df_clean = preprocessor.clean_data(df)
    X, y = preprocessor.split_features_target(df_clean)
    print(f"Loaded raw dataset shape: {df.shape}")
    print(f"Cleaned features shape: {X.shape}, target classes: {y.value_counts().to_dict()}")
