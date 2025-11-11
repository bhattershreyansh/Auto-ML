import pandas as pd
import numpy as np
import logging
import time
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from services.trainer import train_model, MODEL_MAP

logging.basicConfig(level=logging.INFO)

def auto_detect_task(df: pd.DataFrame, target_col: str) -> str:
    unique_vals = df[target_col].nunique()
    dtype = df[target_col].dtype
    if dtype == 'object' or dtype == 'bool' or unique_vals <= 10:
        return "classification"
    else:
        return "regression"

def eligible_models(model_map, task_type):
    models = []
    for name, model_cls in model_map.items():
        if task_type == "classification" and name.lower().endswith("classifier"):
            models.append(name)
        if task_type == "regression" and name.lower().endswith("regressor"):
            models.append(name)
    return models

def generate_correlation_matrix(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Generate correlation matrix data for visualization"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Convert to format suitable for frontend
        correlation_data = []
        columns = corr_matrix.columns.tolist()
        
        for i, row_name in enumerate(columns):
            for j, col_name in enumerate(columns):
                correlation_data.append({
                    "x": col_name,
                    "y": row_name,
                    "value": float(corr_matrix.iloc[i, j]) if not pd.isna(corr_matrix.iloc[i, j]) else 0
                })
        
        # Get target correlations (most important features)
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
            target_correlations = [
                {"feature": feature, "correlation": float(corr)}
                for feature, corr in target_corr.items()
                if feature != target_column
            ][:10]  # Top 10 correlations
        else:
            target_correlations = []
        
        return {
            "correlation_matrix": correlation_data,
            "columns": columns,
            "target_correlations": target_correlations,
            "shape": corr_matrix.shape
        }
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}")
        return {"error": str(e)}

def compare_models(
    filepath: str,
    target_column: str,
    model_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    tune_hyperparams: bool = False,
    cv_folds: int = 3
) -> Dict[str, Any]:
    """
    Train and compare every eligible model in MODEL_MAP.
    Returns a ranked result with summary stats and timing info.
    """
    df = pd.read_csv(filepath)
    task_type = auto_detect_task(df, target_column)
    all_models = eligible_models(MODEL_MAP, task_type)
    # Use all by default or restrict if needed
    test_models = model_names if model_names is not None else all_models

    results, timings = [], []
    for model in test_models:
        try:
            logging.info(f"Training {model}...")
            start = time.time()
            metrics, pipeline, label_encoder = train_model(
                filepath=filepath,
                target_column=target_column,
                model_name=model,
                test_size=test_size,
                tune_hyperparams=tune_hyperparams,
                cv_folds=cv_folds
            )
            duration = round(time.time() - start, 2)
            timings.append({"model": model, "seconds": duration})
            if task_type == "classification":
                score = metrics.get("accuracy", 0)
                metric_name = "accuracy"
            else:
                score = metrics.get("r2_score", 0)
                metric_name = "r2_score"
            results.append({
                "model": model,
                "metric": metric_name,
                "score": float(score),
                "train_time": duration,
                "full_metrics": metrics,
                "tuning": metrics["meta"].get("tuning") if "meta" in metrics else None
            })
            logging.info(f"✅ {model}: {metric_name}={score:.4f} (time: {duration:.2f}s)")
        except Exception as e:
            logging.error(f"❌ {model} failed: {e}")
            results.append({
                "model": model,
                "metric": None,
                "score": None,
                "train_time": None,
                "error": str(e)
            })

    # Sort by best score (descending)
    ranked = sorted([r for r in results if r["score"] is not None], key=lambda x: x["score"], reverse=True)
    failed = [r for r in results if r["score"] is None]

    # Format for frontend compatibility
    leaderboard = []
    for result in ranked:
        leaderboard.append({
            "model_name": result["model"],
            "score": result["score"],
            "train_time": result["train_time"],
            "best_params": result.get("tuning"),
            "full_metrics": result["full_metrics"]
        })

    # Generate correlation matrix and chart data
    correlation_data = generate_correlation_matrix(df, target_column)
    
    # Generate performance chart data
    chart_data = []
    for result in ranked:
        chart_data.append({
            "model": result["model"],
            "score": result["score"],
            "train_time": result["train_time"]
        })
    
    return {
        "task_type": task_type,
        "leaderboard": leaderboard,  # Frontend expects this key
        "models_tried": test_models,
        "successful": len(ranked),
        "failed": len(failed),
        "failures": failed,
        "correlation_data": correlation_data,
        "chart_data": chart_data
    }
