from fastapi import FastAPI, File as FastAPIFile, UploadFile, HTTPException, Body
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

# Auth and Database imports
from auth import get_current_user
from database import init_db, get_db, User, File as DBFile, Experiment
from sqlalchemy.orm import Session
from fastapi import Depends

app = FastAPI(title="AutoML API", version="1.0.0")

# CORS middleware for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create base uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

logging.basicConfig(level=logging.INFO)

def get_user_upload_dir(user_id: str) -> Path:
    """Get or create a user-specific upload directory."""
    user_dir = UPLOAD_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


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

# class ExplainRequest(BaseModel):
#     model_path: str
#     data_path: str
#     target_column: str
#     max_display: Optional[int] = 10
#     sample_size: Optional[int] = 100

# class ExplainPredictionRequest(BaseModel):
#     model_path: str



@app.post("/upload")
async def upload_file(
    file: UploadFile = FastAPIFile(...),
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload CSV file for analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Use user-specific directory
    user_dir = get_user_upload_dir(user_id)
    file_path = user_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Record in database
    # First, ensure user exists in our DB
    db_user = db.query(User).filter(User.clerk_id == user_id).first()
    if not db_user:
        db_user = User(clerk_id=user_id)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    
    db_file = DBFile(
        user_id=db_user.id,
        filename=file.filename,
        original_name=file.filename,
        filepath=str(file_path)
    )
    db.add(db_file)
    db.commit()
    
    return {
        "filename": file.filename,
        "filepath": str(file_path).replace("\\", "/"),
        "message": "File uploaded successfully"
    }


@app.post("/analyze")
async def analyze_data(
    request: AnalyzeRequest,
    user_id: str = Depends(get_current_user)
):
    """Analyze the uploaded dataset"""
    try:
        # Security check: ensure path is within user's directory
        user_dir = get_user_upload_dir(user_id)
        filename = Path(request.filepath).name
        absolute_path = user_dir / filename
        
        if not absolute_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        results = analyze_dataset(str(absolute_path))
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clean")
async def clean_dataset(
    request: CleanRequest,
    user_id: str = Depends(get_current_user)
):
    """Clean the dataset"""
    try:
        # Security check: ensure path is within user's directory
        user_dir = get_user_upload_dir(user_id)
        filename = Path(request.filepath).name
        file_path = user_dir / filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"File not found: {filename}"
            )
        
        cleaned_path = clean_data(str(file_path))
        
        if not cleaned_path:
            raise HTTPException(
                status_code=500, 
                detail="Data cleaning failed"
            )
        
        return {
            "cleaned_filepath": cleaned_path,
            "message": "Data cleaned successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /clean endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/select-model")
async def select_best_model(
    request: ModelSelectionRequest,
    user_id: str = Depends(get_current_user)
):
    """Select the best model for the dataset"""
    try:
        user_dir = get_user_upload_dir(user_id)
        filename = Path(request.filepath).name
        input_file_path = user_dir / filename
        
        if not input_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {filename}"
            )
        
        model_suggestions = select_model(str(input_file_path), request.target_column)
        return model_suggestions
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_selected_model(
    request: TrainingRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Train the selected model and record experiment"""
    try:
        user_dir = get_user_upload_dir(user_id)
        filename = Path(request.filepath).name
        input_file_path = user_dir / filename
        
        if not input_file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        metrics, trained_pipeline, label_encoder = train_model(
            filepath=str(input_file_path),
            target_column=request.target_column,
            model_name=request.model_name,
            test_size=request.test_size or 0.2,
            tune_hyperparams=request.tune_hyperparams,
            cv_folds=request.cv_folds or 5,
            n_trials=request.n_trials or 50
        )
        
        # Save model in user's directory
        model_filename = f"trained_{request.model_name}_{Path(filename).stem}_{request.target_column}.joblib"
        model_save_path = user_dir / model_filename
        joblib.dump(trained_pipeline, model_save_path)
        
        # Save experiment to database
        db_user = db.query(User).filter(User.clerk_id == user_id).first()
        db_experiment = Experiment(
            user_id=db_user.id,
            model_name=request.model_name,
            target_column=request.target_column,
            metrics=metrics,
            model_path=str(model_save_path)
        )
        db.add(db_experiment)
        db.commit()
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_path": f"uploads/{user_id}/{model_filename}",
            "model_filename": model_filename,
            "model_name": request.model_name,
            "target_column": request.target_column
        }
    
    except Exception as e:
        logging.error(f"Error in /train: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_trained_model(
    request: EvaluationRequest,
    user_id: str = Depends(get_current_user)
):
    """Evaluate a trained pipeline on test data"""
    try:
        user_dir = get_user_upload_dir(user_id)
        model_filename = Path(request.model_path).name
        test_filename = Path(request.test_data_path).name
        
        model_file_path = user_dir / model_filename
        test_file_path = user_dir / test_filename
        
        if not model_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_filename}")
        if not test_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Test data not found: {test_filename}")
        
        pipeline = joblib.load(model_file_path)
        df_test = pd.read_csv(test_file_path)
        
        if request.target_column not in df_test.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not in test data")
        
        X_test = df_test.drop(columns=[request.target_column])
        y_test = df_test[request.target_column]
        
        results = evaluate_model(pipeline, X_test, y_test, task_type=request.task_type, plot=False)
        
        return {
            "evaluation_results": results,
            "test_samples": len(X_test),
            "model_used": model_filename,
            "test_data_used": test_filename,
            "message": "Evaluation completed"
        }
    except Exception as e:
        logging.error(f"Error in evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_single(
    request: PredictRequest,
    user_id: str = Depends(get_current_user)
):
    """Make prediction using trained pipeline"""
    try:
        user_dir = get_user_upload_dir(user_id)
        model_filename = Path(request.model_path).name
        model_file_path = user_dir / model_filename
        
        if not model_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_filename}")
        
        pipeline = joblib.load(model_file_path)
        X = pd.DataFrame([request.features])
        y_pred = pipeline.predict(X)[0]
        
        resp = {
            "prediction": int(y_pred) if isinstance(y_pred, (np.integer, int)) else float(y_pred)
        }
        if hasattr(pipeline, "predict_proba"):
            try:
                resp["probabilities"] = pipeline.predict_proba(X)[0].tolist()
            except: pass
        return resp
    except Exception as e:
        logging.error(f"Error in predict_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-models")
async def compare_multiple_models(
    request: ModelComparisonRequest,
    user_id: str = Depends(get_current_user)
):
    """Compare all eligible models scoped to user's dataset."""
    try:
        user_dir = get_user_upload_dir(user_id)
        filename = Path(request.filepath).name
        input_file_path = user_dir / filename

        if not input_file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

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

@app.get("/experiments")
async def get_experiments(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve history of training runs for the current user."""
    db_user = db.query(User).filter(User.clerk_id == user_id).first()
    if not db_user:
        return []
    return db.query(Experiment).filter(Experiment.user_id == db_user.id).order_by(Experiment.created_at.desc()).all()


@app.get("/download-assets")
async def download_assets(
    model_path: str,
    user_id: str = Depends(get_current_user)
):
    import io
    import zipfile
    from fastapi.responses import StreamingResponse
    
    try:
        user_dir = get_user_upload_dir(user_id)
        model_filename = Path(model_path).name
        model_file = user_dir / model_filename
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail="Model not found")
            
        io_stream = io.BytesIO()
        with zipfile.ZipFile(io_stream, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            # Add the model
            zip_file.write(model_file, arcname=model_filename)
            
            # Find and add the dataset CSV
            for file_path in user_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.csv':
                    zip_file.write(file_path, arcname=file_path.name)
                        
        io_stream.seek(0)
        
        return StreamingResponse(
            io_stream, 
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="autopilot-deployment.zip"'}
        )
    except Exception as e:
        logging.error(f"Error zipping assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "AutoML API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
