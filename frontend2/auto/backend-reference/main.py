from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any,Union,List
import joblib
import logging

# Import your existing services
from services.analyzer import analyze_dataset
from services.cleaner import clean_data
from services.model_selector import select_model
from services.trainer import train_model
from services.tester import evaluate_model
from services.model_comparator import compare_models

app = FastAPI(title="AutoML API", version="1.0.0")

# CORS middleware for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)


# Request models
class AnalyzeRequest(BaseModel):
    filepath: str


class CleanRequest(BaseModel):
    filepath: str


class ModelSelectionRequest(BaseModel):
    filepath: str
    target_column: str


class TrainingRequest(BaseModel):
    filepath: str
    target_column: str
    model_name: str
    test_size: Optional[float] = 0.2
    tune_hyperparams: Optional[Union[bool, str]] = False  # NEW: False, 'grid', or 'optuna'
    cv_folds: Optional[int] = 5  # NEW
    n_trials: Optional[int] = 50  # NEW: For Optuna

    class Config:
        schema_extra = {
            "example": {
                "filepath": "diabetes_cleaned.csv",
                "target_column": "Outcome",
                "model_name": "RandomForestClassifier",
                "test_size": 0.2,
                "tune_hyperparams": "grid",  # or "optuna" or False
                "cv_folds": 5,
                "n_trials": 50
            }
        }



class EvaluationRequest(BaseModel):
    model_path: str
    test_data_path: str
    target_column: str
    task_type: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "model_path": "trained_model_RandomForestClassifier_Outcome.joblib",
                "test_data_path": "diabetes_cleaned.csv",
                "target_column": "Outcome",
                "task_type": "classification"
            }
        }


class PredictRequest(BaseModel):
    model_path: str
    features: Dict[str, Any]

class ModelComparisonRequest(BaseModel):
    filepath: str
    target_column: str
    model_names: Optional[List[str]] = None
    test_size: Optional[float] = 0.2
    tune_hyperparams: Optional[bool] = False
    cv_folds: Optional[int] = 3

class ExplainRequest(BaseModel):
    model_path: str
    data_path: str
    target_column: str
    max_display: Optional[int] = 10
    sample_size: Optional[int] = 100

class ExplainPredictionRequest(BaseModel):
    model_path: str



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file for analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "filepath": str(file_path),
        "message": "File uploaded successfully"
    }


@app.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    """Analyze the uploaded dataset"""
    try:
        absolute_path = UPLOAD_DIR / Path(request.filepath).name
        if not absolute_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {absolute_path}")
        
        results = analyze_dataset(str(absolute_path))
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clean")
async def clean_dataset(request: CleanRequest):
    """Clean the dataset"""
    try:
        # Normalize filepath
        filepath = request.filepath
        if not filepath.startswith("uploads/"):
            filepath = f"uploads/{filepath}"
        
        # Check if file exists before calling clean_data
        file_path = Path(filepath)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"File not found: {filepath}"
            )
        
        cleaned_path = clean_data(str(file_path))
        
        # Check if cleaning actually succeeded
        if not cleaned_path or cleaned_path == "":
            raise HTTPException(
                status_code=500, 
                detail="Data cleaning failed - empty result returned"
            )
        
        return {
            "cleaned_filepath": cleaned_path,
            "message": "Data cleaned successfully"
        }
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error in /clean endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/select-model")
async def select_best_model(request: ModelSelectionRequest):
    """Select the best model for the dataset"""
    try:
        logging.info(f"Received model selection request for: {request.filepath}, target: {request.target_column}")
        
        # Normalize filepath
        filepath = request.filepath
        if filepath.startswith("uploads/uploads/"):
            filepath = filepath.replace("uploads/uploads/", "uploads/", 1)
        elif not filepath.startswith("uploads/"):
            filepath = f"uploads/{filepath}"
        
        input_file_path = Path(filepath)
        if not input_file_path.is_absolute():
            input_file_path = UPLOAD_DIR / input_file_path.name
        
        if not input_file_path.exists():
            alternative_path = UPLOAD_DIR / Path(request.filepath).name
            if alternative_path.exists():
                input_file_path = alternative_path
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found at: {input_file_path.absolute()}"
                )
        
        model_suggestions = select_model(str(input_file_path), request.target_column)
        if not model_suggestions:
            raise HTTPException(status_code=500, detail="Model selection failed")
        
        return model_suggestions
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_selected_model(request: TrainingRequest):
    """Train the selected model using sklearn Pipeline"""
    try:
        # Normalize filepath
        filename = Path(request.filepath).name
        input_file_path = UPLOAD_DIR / filename
        
        logging.info(f"Looking for file at: {input_file_path}")
        
        if not input_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {input_file_path}"
            )
        
        logging.info("File found, calling train_model...")
        
        # NEW: train_model now returns (metrics, pipeline, label_encoder)
        metrics, trained_pipeline, label_encoder = train_model(
            filepath=str(input_file_path),
            target_column=request.target_column,
            model_name=request.model_name,
            test_size=getattr(request, 'test_size', 0.2),
            tune_hyperparams=getattr(request, 'tune_hyperparams', False),  # NEW
            cv_folds=getattr(request, 'cv_folds', 5),  # NEW
            n_trials=getattr(request, 'n_trials', 50)  # NEW
        )
        
        if trained_pipeline is None:
            raise HTTPException(status_code=500, detail="Model training failed unexpectedly.")
        
        # Create model filename
        model_filename = f"trained_{request.model_name}_{Path(filename).stem}_{request.target_column}.joblib"
        model_save_path = UPLOAD_DIR / model_filename
        
        # Save the complete pipeline
        joblib.dump(trained_pipeline, model_save_path)
        logging.info(f"✅ Pipeline saved to: {model_save_path}")
        
        # Save label encoder separately if it exists (for classification)
        if label_encoder is not None:
            encoder_filename = f"encoder_{request.model_name}_{Path(filename).stem}_{request.target_column}.joblib"
            encoder_save_path = UPLOAD_DIR / encoder_filename
            joblib.dump(label_encoder, encoder_save_path)
            logging.info(f"✅ Label encoder saved to: {encoder_save_path}")
        else:
            encoder_filename = None
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_path": f"uploads/{model_filename}",
            "encoder_path": f"uploads/{encoder_filename}" if encoder_filename else None,
            "model_filename": model_filename,
            "model_name": request.model_name,
            "target_column": request.target_column
        }
    
    except ValueError as e:
        logging.warning(f"Validation error in training: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error in /train: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_trained_model(request: EvaluationRequest):
    """Evaluate a trained pipeline on test data"""
    try:
        logging.info(f"Received evaluation request for model: {request.model_path}")
        
        # Normalize paths
        model_path = request.model_path
        if model_path.startswith("uploads/uploads/"):
            model_path = model_path.replace("uploads/uploads/", "uploads/", 1)
        elif not model_path.startswith("uploads/"):
            model_path = f"uploads/{model_path}"
        
        test_data_path = request.test_data_path
        if test_data_path.startswith("uploads/uploads/"):
            test_data_path = test_data_path.replace("uploads/uploads/", "uploads/", 1)
        elif not test_data_path.startswith("uploads/"):
            test_data_path = f"uploads/{test_data_path}"
        
        # Build absolute paths
        model_file_path = Path(model_path)
        if not model_file_path.is_absolute():
            model_file_path = UPLOAD_DIR / model_file_path.name
        
        test_file_path = Path(test_data_path)
        if not test_file_path.is_absolute():
            test_file_path = UPLOAD_DIR / test_file_path.name
        
        # Check files exist
        if not model_file_path.exists():
            alternative_model_path = UPLOAD_DIR / Path(request.model_path).name
            if alternative_model_path.exists():
                model_file_path = alternative_model_path
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found at: {model_file_path.absolute()}"
                )
        
        if not test_file_path.exists():
            alternative_test_path = UPLOAD_DIR / Path(request.test_data_path).name
            if alternative_test_path.exists():
                test_file_path = alternative_test_path
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Test data file not found at: {test_file_path.absolute()}"
                )
        
        # Load pipeline (not just model)
        pipeline = joblib.load(model_file_path)
        logging.info("Pipeline loaded successfully")
        
        # Load test data
        df_test = pd.read_csv(test_file_path)
        logging.info(f"Test data loaded: {df_test.shape}")
        
        if request.target_column not in df_test.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in test data"
            )
        
        # Prepare test data - NO PREPROCESSING NEEDED (pipeline handles it)
        X_test = df_test.drop(columns=[request.target_column])
        y_test = df_test[request.target_column]
        
        # Drop rows with missing target
        valid_idx = y_test.dropna().index
        X_test = X_test.loc[valid_idx]
        y_test = y_test.loc[valid_idx]
        
        if len(X_test) == 0:
            raise HTTPException(status_code=400, detail="No valid test samples")
        
        logging.info(f"Evaluating pipeline on {len(X_test)} test samples...")
        
        # Evaluate (pipeline handles preprocessing automatically)
        results = evaluate_model(
            pipeline,
            X_test,
            y_test,
            task_type=request.task_type,
            plot=False
        )
        
        logging.info("✅ Pipeline evaluation completed successfully")
        
        return {
            "evaluation_results": results,
            "test_samples": len(X_test),
            "model_used": str(model_file_path.name),
            "test_data_used": str(test_file_path.name),
            "message": "Pipeline evaluation completed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in evaluate_trained_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_single(request: PredictRequest):
    """Make prediction using trained pipeline"""
    try:
        # Normalize model path
        model_path = request.model_path
        if model_path.startswith("uploads/uploads/"):
            model_path = model_path.replace("uploads/uploads/", "uploads/", 1)
        elif not model_path.startswith("uploads/"):
            model_path = f"uploads/{model_path}"
        
        model_file_path = Path(model_path)
        if not model_file_path.is_absolute():
            model_file_path = UPLOAD_DIR / model_file_path.name
        
        if not model_file_path.exists():
            alt = UPLOAD_DIR / Path(request.model_path).name
            if alt.exists():
                model_file_path = alt
            else:
                raise HTTPException(status_code=404, detail=f"Model not found: {model_file_path}")
        
        # Load pipeline
        pipeline = joblib.load(model_file_path)
        
        # Build single-row DataFrame from incoming features
        X = pd.DataFrame([request.features])
        
        # Pipeline handles all preprocessing automatically!
        y_pred = pipeline.predict(X)[0]
        
        resp: Dict[str, Any] = {
            "prediction": int(y_pred) if isinstance(y_pred, (np.integer, int)) else float(y_pred)
        }
        
        # Probabilities if available
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X)[0]
                resp["probabilities"] = proba.tolist()
            except Exception as e:
                logging.warning(f"Could not get probabilities: {e}")
        
        return resp
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in predict_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-models")
async def compare_multiple_models(request: ModelComparisonRequest):
    """Compare all eligible models and rank by score."""
    try:
        filename = Path(request.filepath).name
        input_file_path = UPLOAD_DIR / filename

        if not input_file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {input_file_path}")

        result = compare_models(
            filepath=str(input_file_path),
            target_column=request.target_column,
            model_names=request.model_names,
            test_size=request.test_size,
            tune_hyperparams=request.tune_hyperparams,
            cv_folds=request.cv_folds
        )

        return {"message": "Model comparison complete", "comparison": result}
    except Exception as e:
        logging.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "AutoML API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
