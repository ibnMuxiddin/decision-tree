import os
import joblib
import pandas as pd
import numpy as np

# Defined encoding mappings
SEX_MAP = {'F': 0, 'M': 1}
SEX_MAP_INV = {0: 'F', 1: 'M'}

BP_MAP = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}
BP_MAP_INV = {0: 'LOW', 1: 'NORMAL', 2: 'HIGH'}

CHOL_MAP = {'NORMAL': 0, 'HIGH': 1}
CHOL_MAP_INV = {0: 'NORMAL', 1: 'HIGH'}

FEATURE_COLS = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
TARGET_COL = 'Drug'

class DrugDataPreprocessor:
    """Preprocessor for Drug Recommendation dataset."""
    
    def __init__(self):
        self.sex_map = SEX_MAP
        self.bp_map = BP_MAP
        self.chol_map = CHOL_MAP
        self.feature_names = FEATURE_COLS

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocesses full dataframe for training."""
        df_clean = df.copy()
        
        df_clean['Sex'] = df_clean['Sex'].map(self.sex_map)
        df_clean['BP'] = df_clean['BP'].map(self.bp_map)
        df_clean['Cholesterol'] = df_clean['Cholesterol'].map(self.chol_map)
        
        X = df_clean[self.feature_names].values
        y = df_clean[TARGET_COL].values
        return X, y

    def transform_sample(self, age: int, sex: str, bp: str, cholesterol: str, na_to_k: float) -> np.ndarray:
        """Transforms a single patient sample into model feature array."""
        sex_val = self.sex_map.get(sex.upper(), 0)
        bp_val = self.bp_map.get(bp.upper(), 1)
        chol_val = self.chol_map.get(cholesterol.upper(), 0)
        
        features = np.array([[age, sex_val, bp_val, chol_val, float(na_to_k)]])
        return features

def save_preprocessor(preprocessor: DrugDataPreprocessor, filepath: str):
    """Saves preprocessor object to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)

def load_preprocessor(filepath: str) -> DrugDataPreprocessor:
    """Loads preprocessor object from pickle file."""
    return joblib.load(filepath)
