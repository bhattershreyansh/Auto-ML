import pandas as pd
import numpy as np
import logging
import joblib
import shap
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64

logging.basicConfig(level=logging.INFO)


def generate_shap_explanation(
    model_path: str,
    data_path: str,
    target_column: str,
    max_display: int = 10,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a trained model.
    Returns both Matplotlib images and Plotly JSON.
    """
    try:
        # Load pipeline and data
        pipeline = joblib.load(model_path)
        df = pd.read_csv(data_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column])
        
        # Sample data for speed
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        logging.info(f"🔍 Generating SHAP explanations for {len(X_sample)} samples...")
        
        # Get the actual model from pipeline
        model = pipeline.named_steps['model']
        
        # Transform data through preprocessor
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_sample)

        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        
        # Get feature names after preprocessing
        feature_names = _get_feature_names(pipeline)
        
        # ✅ FIX: Choose the right SHAP explainer based on model type
        model_class_name = model.__class__.__name__
        
        if model_class_name in ['XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 
                                 'RandomForestRegressor', 'GradientBoostingClassifier', 
                                 'GradientBoostingRegressor']:
            # Use TreeExplainer for tree-based models
            logging.info(f"Using TreeExplainer for {model_class_name}")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_transformed)
        else:
            # Use generic Explainer for other models (LogisticRegression, etc.)
            logging.info(f"Using generic Explainer for {model_class_name}")
            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed)
        shap_values.feature_names = feature_names
        
        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Generate Matplotlib plots as base64
        matplotlib_plots = _generate_matplotlib_plots(shap_values, max_display)
        
        # Generate Plotly feature importance chart
        plotly_chart = _generate_plotly_importance(importance_df, max_display)
        
        explanation = {
            'feature_importance_table': importance_df.head(20).to_dict('records'),
            'global_importance_plotly': plotly_chart,
            'matplotlib_plots': matplotlib_plots,
            'base_value': float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else None,
            'num_samples': len(X_sample),
            'num_features': len(feature_names)
        }
        
        logging.info("✅ SHAP explanation generated successfully")
        
        return explanation
    
    except Exception as e:
        logging.error(f"❌ SHAP explanation failed: {e}")
        raise



def explain_prediction(
    model_path: str,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Explain a single prediction using SHAP.
    """
    try:
        pipeline = joblib.load(model_path)
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = pipeline.predict(X)[0]
        
        # Get probabilities if available
        proba = None
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(X)[0].tolist()
        
        # Transform data
        X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        model = pipeline.named_steps['model']
        feature_names = _get_feature_names(pipeline)
        
        # ✅ FIX: Choose the right SHAP explainer
        model_class_name = model.__class__.__name__
        
        if model_class_name in ['XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 
                                 'RandomForestRegressor', 'GradientBoostingClassifier', 
                                 'GradientBoostingRegressor']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_transformed)
        
        shap_values = explainer(X_transformed)
        shap_values.feature_names = feature_names
        
        # Feature contributions
        contributions = {}
        for i, feature in enumerate(feature_names):
            if i < len(shap_values.values[0]):
                contributions[feature] = float(shap_values.values[0][i])
        
        # Sort by absolute contribution
        sorted_contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return {
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            'probabilities': proba,
            'shap_contributions': sorted_contributions,
            'base_value': float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else None,
            'top_contributors': list(sorted_contributions.keys())[:5]
        }
    
    except Exception as e:
        logging.error(f"❌ Prediction explanation failed: {e}")
        raise



def _get_feature_names(pipeline) -> list:
    """Extract feature names after preprocessing"""
    feature_names = []
    preprocessor = pipeline.named_steps['preprocessor']
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Get feature names from OneHotEncoder
            encoder = trans.named_steps['onehot']
            encoded_features = encoder.get_feature_names_out(cols)
            feature_names.extend(encoded_features)
    
    return feature_names


def _generate_matplotlib_plots(shap_values, max_display: int) -> Dict[str, str]:
    """Generate Matplotlib SHAP plots as base64 images"""
    plots = {}
    
    try:
        # 1. Summary bar plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plots['summary_bar'] = _fig_to_base64()
        plt.close()
        
        # 2. Beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plots['beeswarm'] = _fig_to_base64()
        plt.close()
        
        # 3. Waterfall plot (first prediction)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
        plots['waterfall_first'] = _fig_to_base64()
        plt.close()
        
    except Exception as e:
        logging.warning(f"Matplotlib plot generation failed: {e}")
    
    return plots


def _generate_plotly_importance(importance_df: pd.DataFrame, max_display: int) -> str:
    """Generate Plotly interactive feature importance chart"""
    try:
        top_features = importance_df.head(max_display)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_features['importance'].round(4),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Global Feature Importance (Mean |SHAP Value|)',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            yaxis=dict(autorange='reversed'),
            height=400 + (max_display * 20),
            showlegend=False,
            hovermode='y'
        )
        
        return pio.to_json(fig)
    
    except Exception as e:
        logging.warning(f"Plotly chart generation failed: {e}")
        return "{}"


def _fig_to_base64():
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_base64}"
