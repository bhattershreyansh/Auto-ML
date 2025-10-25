import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from typing import Optional, Union, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)

# Mapping string model names to classes
MODEL_MAP = {
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LogisticRegression": LogisticRegression,
}

# Hyperparameter grids for GridSearchCV
PARAM_GRIDS = {
    "RandomForestClassifier": {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    "RandomForestRegressor": {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    "GradientBoostingClassifier": {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5]
    },
    "GradientBoostingRegressor": {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5]
    },
    "LogisticRegression": {
        'model__C': [0.1, 1.0, 10.0],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs', 'liblinear'],
        'model__max_iter': [200, 500]
    }
}


def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    return obj


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.
    
    Args:
        X: Feature dataframe
        scale_numeric: Whether to apply StandardScaler to numeric features
    
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric pipeline: impute + optional scaling
    numeric_transformer_steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    if scale_numeric:
        numeric_transformer_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_transformer_steps)
    
    # Categorical pipeline: impute + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor


def tune_with_optuna(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_trials: int = 50,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Optuna (Bayesian optimization).
    
    Args:
        pipeline: sklearn Pipeline with preprocessor and model
        X_train: Training features
        y_train: Training target
        model_name: Name of the model
        n_trials: Number of optimization trials
        cv_folds: Number of CV folds
    
    Returns:
        Dictionary with best parameters and score
    """
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            # Define hyperparameter search space based on model
            if model_name == "RandomForestClassifier" or model_name == "RandomForestRegressor":
                params = {
                    'model__n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'model__max_depth': trial.suggest_int('max_depth', 5, 50),
                    'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
            elif model_name == "GradientBoostingClassifier" or model_name == "GradientBoostingRegressor":
                params = {
                    'model__n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'model__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'model__max_depth': trial.suggest_int('max_depth', 3, 10),
                    'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
            elif model_name == "LogisticRegression":
                params = {
                    'model__C': trial.suggest_float('C', 0.01, 100.0, log=True),
                    'model__solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                    'model__max_iter': trial.suggest_int('max_iter', 100, 1000),
                }
            else:
                return 0.0
            
            # Set parameters
            pipeline.set_params(**params)
            
            # Cross-validation score
            is_classification = model_name.endswith("Classifier") or model_name == "LogisticRegression"
            scoring = 'accuracy' if is_classification else 'r2'
            score = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1).mean()
            
            return score
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logging.info(f"✅ Optuna found best params with score: {study.best_value:.4f}")
        
        # Convert best params to pipeline format
        best_params = {f'model__{k}': v for k, v in study.best_params.items()}
        
        return {
            'best_params': best_params,
            'best_score': float(study.best_value),
            'method': 'optuna',
            'n_trials': n_trials
        }
    
    except ImportError:
        logging.warning("Optuna not installed. Install with: pip install optuna")
        return None
    except Exception as e:
        logging.error(f"Optuna tuning failed: {e}")
        return None


def train_model(
    filepath: str,
    target_column: str,
    model_name: str,
    model_params: Optional[dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
    tune_hyperparams: Union[bool, str] = False,  # False, 'grid', or 'optuna'
    cv_folds: int = 5,
    n_trials: int = 50,  # For Optuna
) -> Tuple[Dict[str, Any], Pipeline, Optional[LabelEncoder]]:
    """
    Train the specified model on the dataset's target using sklearn Pipeline.
    
    Args:
        filepath (str): Path to CSV dataset.
        target_column (str): Target column name.
        model_name (str): One of the valid model names from MODEL_MAP.
        model_params (Optional[dict]): Hyperparameters for model instantiation (ignored if tuning).
        test_size (float): Fraction for test split.
        random_state (int): Seed for reproducibility.
        scale_numeric (bool): Whether to scale numeric features.
        tune_hyperparams (Union[bool, str]): False, 'grid', or 'optuna' for hyperparameter tuning.
        cv_folds (int): Number of cross-validation folds for tuning.
        n_trials (int): Number of trials for Optuna optimization.
    
    Returns:
        report (dict): Evaluation metrics and metadata (JSON-serializable).
        pipeline (Pipeline): Complete trained pipeline (preprocessor + model).
        label_encoder (Optional[LabelEncoder]): Label encoder for classification targets (if used).
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(MODEL_MAP.keys())}")
    
    logging.info(f"🚀 Training {model_name} (Tuning: {tune_hyperparams})")
    
    # Load data
    df = pd.read_csv(filepath)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Determine if classification based on model_name
    is_classification = model_name.endswith("Classifier") or model_name == "LogisticRegression"
    
    # Encode target if classification and target is categorical
    label_encoder = None
    if is_classification and (y.dtype == 'object' or y.nunique() < 20):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        logging.info(f"✅ Target encoded. Classes: {label_encoder.classes_}")
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X, scale_numeric=scale_numeric)
    
    # Build complete pipeline
    model_class = MODEL_MAP[model_name]
    model = model_class(**(model_params or {}))
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train-test split with stratify for classification
    stratify = y if is_classification else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        logging.warning(f"Stratified split failed: {e}. Retrying without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    if len(y_test) < 5:
        logging.warning(f"Only {len(y_test)} samples in test set, results may be unreliable")
    
    # Hyperparameter tuning
    tuning_results = None
    
    if tune_hyperparams == 'grid':
        logging.info("🔍 Starting GridSearchCV hyperparameter tuning...")
        param_grid = PARAM_GRIDS.get(model_name, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring='accuracy' if is_classification else 'r2',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            
            tuning_results = {
                'best_params': convert_numpy_types(grid_search.best_params_),
                'best_score': float(grid_search.best_score_),
                'method': 'grid_search',
                'cv_folds': cv_folds
            }
            logging.info(f"✅ GridSearch best score: {grid_search.best_score_:.4f}")
            logging.info(f"✅ Best params: {grid_search.best_params_}")
        else:
            logging.warning(f"No parameter grid defined for {model_name}, skipping GridSearch")
            pipeline.fit(X_train, y_train)
    
    elif tune_hyperparams == 'optuna':
        logging.info("🔍 Starting Optuna hyperparameter tuning...")
        tuning_results = tune_with_optuna(pipeline, X_train, y_train, model_name, n_trials, cv_folds)
        
        if tuning_results:
            # Apply best params and refit
            pipeline.set_params(**tuning_results['best_params'])
            pipeline.fit(X_train, y_train)
        else:
            logging.warning("Optuna tuning failed, training with default params")
            pipeline.fit(X_train, y_train)
    
    else:
        # No tuning, just fit
        pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Prepare evaluation report
    if is_classification:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report["accuracy"] = float(accuracy_score(y_test, y_pred))
        report["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        report["meta"] = {
            "task": "classification",
            "model": model_name,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "classes": [int(x) for x in np.unique(y)],
            "feature_count": int(X.shape[1]),
            "pipeline_steps": [step[0] for step in pipeline.steps],
            "tuning": tuning_results if tuning_results else None
        }
    else:
        report = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "meta": {
                "task": "regression",
                "model": model_name,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "target_range": [float(np.min(y)), float(np.max(y))],
                "feature_count": int(X.shape[1]),
                "pipeline_steps": [step[0] for step in pipeline.steps],
                "tuning": tuning_results if tuning_results else None
            },
        }
    
    # Convert all NumPy types in the report
    report = convert_numpy_types(report)
    
    logging.info(f"✅ Training completed. Test accuracy/R²: {report.get('accuracy', report.get('r2_score', 'N/A'))}")
    
    return report, pipeline, label_encoder


if __name__ == "__main__":
    import pprint
    
    # Example usage:
    dataset_path = input("Enter dataset path: ").strip()
    target_col = input("Enter target column name: ").strip()
    model_name = input(f"Enter model name {list(MODEL_MAP.keys())}: ").strip()
    tune = input("Tune hyperparams? (none/grid/optuna): ").strip().lower()
    
    tune_param = False if tune == 'none' else tune
    
    try:
        metrics, trained_pipeline, le = train_model(
            dataset_path, 
            target_col, 
            model_name,
            tune_hyperparams=tune_param
        )
        print("\n✅ Training complete. Evaluation metrics:")
        pprint.pprint(metrics)
        print("\n📦 Pipeline ready for deployment!")
    except Exception as e:
        print(f"❌ Error: {e}")
