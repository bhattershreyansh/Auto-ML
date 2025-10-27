import pandas as pd
import os
import logging
from typing import Tuple, Dict, Optional
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path



def clean_data(filepath: str) -> str:
    """
    Cleans the dataset by:
    - Filling missing numerical values with the median
    - Filling missing categorical values with the mode
    
    Args:
        filepath (str): Path to the input CSV file
    
    Returns:
        str: Path to the cleaned CSV file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other errors
    """
    # Normalize filepath
    file_path = Path(filepath)
    
    if not file_path.is_absolute():
        if not file_path.exists():
            uploads_path = Path("uploads") / file_path.name
            if uploads_path.exists():
                file_path = uploads_path
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isnull().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
        
        # Categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].isnull().any():
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
        
        # Handle file extension robustly
        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext if ext else '.csv'}"
        df.to_csv(cleaned_path, index=False)
        
        logging.info(f"✅ Data cleaned and saved to: {cleaned_path}")
        return str(cleaned_path).replace("\\", "/")  # Ensure forward slashes for consistency
    
    except Exception as e:
        logging.error(f"❌ Error cleaning data: {e}")
        raise  # Re-raise the exception instead of returning ""



def get_cleaning_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a reusable sklearn ColumnTransformer for data cleaning (imputation only).
    This can be used within a larger Pipeline for production.
    
    Args:
        X (pd.DataFrame): Feature dataframe
    
    Returns:
        ColumnTransformer: Preprocessing transformer
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers = []
    
    if numeric_features:
        numeric_transformer = SimpleImputer(strategy='median')
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = SimpleImputer(strategy='most_frequent')
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Keep other columns as-is
    )
    
    return preprocessor
