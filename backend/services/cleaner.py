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
        import numpy as np
        df = pd.read_csv(file_path)
        
        # 1. Missingness Thresholding (>50% missing -> DROP)
        missing_ratios = df.isnull().mean()
        drop_cols = missing_ratios[missing_ratios > 0.5].index
        if len(drop_cols) > 0:
            logging.info(f"🚨 Dropping columns with massive missing data (>50%): {list(drop_cols)}")
            df = df.drop(columns=drop_cols)
            
        # 2. Silent Type Rectification
        for col in df.select_dtypes(include=['object', 'category']).columns:
            try:
                # Attempt aggressive numeric cast. If it's pure strings, it'll become NaNs.
                parsed = pd.to_numeric(df[col], errors='coerce')
                # If majority of data cleanly parsed to number, it was contaminated numeric data.
                if parsed.notnull().mean() > 0.5:
                    logging.info(f"🔧 Type Rectification: Converting '{col}' from object to numeric.")
                    df[col] = parsed
            except Exception:
                pass
        
        # 3. Numeric: Imputation & IQR Winsorization (Outlier Capping)
        for col in df.select_dtypes(include=['number']).columns:
            # Impute
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
            # IQR Capping (Squeeze extreme outliers down to IQR boundaries)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Only cap if IQR > 0 to prevent flattening binary or identical columns
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Clip values to bounds, ignoring NaNs
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                
        # 4. Categorical: Mode Imputation
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].isnull().any():
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
        
        # Handle file extension robustly
        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext if ext else '.csv'}"
        df.to_csv(cleaned_path, index=False)
        
        logging.info(f"✅ Data aggressively cleaned (Prod-Ready) and saved to: {cleaned_path}")
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
