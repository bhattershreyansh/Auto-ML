import os
import pandas as pd
import logging
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

PROMPT_TEMPLATE = """
You are an expert machine learning engineer. Given the sample data below and information about the target column, please:

1. Recommend the single best ML model for this prediction task from these models:
   - RandomForestClassifier
   - RandomForestRegressor
   - GradientBoostingClassifier
   - GradientBoostingRegressor
   - LogisticRegression
   - XGBClassifier
   - XGBRegressor

2. Suggest 2-3 alternative eligible models, with a brief explanation of why each model could be suitable.
3. Explain shortly why you chose the best model.

Dataset sample (first 5 rows):
{sample}

Target column name: {target_col}
Target column type: {target_type}
Number of unique values in target column: {target_unique}

Respond only with JSON in this format:
{{
  "best_model": {{
    "name": "<model_name>",
    "description": "<reason for choosing this model>"
  }},
  "other_models": [
    {{"name": "<model_name>", "description": "<explanation>"}},
    {{"name": "<model_name>", "description": "<explanation>"}}
  ]
}}
"""


def call_llm_model_selector(sample_df: pd.DataFrame, target_col: str) -> dict:
    """
    Calls the LLM to get the best and other ML model suggestions with explanations.
    Returns a dict with keys: best_model, other_options.
    """
    if not groq_client:
        logging.warning("No LLM client available, returning empty suggestions.")
        return {}

    try:
        target_type = str(sample_df[target_col].dtype)
        target_unique = sample_df[target_col].nunique()
        sample_text = sample_df.head(5).to_csv(index=False)

        prompt = PROMPT_TEMPLATE.format(
            sample=sample_text,
            target_col=target_col,
            target_type=target_type,
            target_unique=target_unique,
        )

        print("🤖 Calling LLM for model suggestions...")
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        text = response.choices[0].message.content
        print(f"📝 LLM Response: {text[:200]}...")

        # Extract JSON object from response safely
        import re

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                result_json = json.loads(match.group(1))
                print("✅ Successfully parsed LLM response")
                print(f"🔍 Parsed JSON: {result_json}")
                return result_json
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"🔍 Raw JSON text: {match.group(1)}")
                return {}
        else:
            logging.warning(f"LLM output did not contain parseable JSON: {text}")
            print(f"🔍 Full LLM response: {text}")
            return {}

    except Exception as e:
        logging.error(f"LLM model selector call failed: {e}")
        return {}

def _format_response(suggestions: dict, df: pd.DataFrame, target_column: str) -> dict:
    """Format the response to include all expected fields for the frontend"""
    # Extract model names
    recommended_models = []
    if "best_model" in suggestions and "name" in suggestions["best_model"]:
        recommended_models.append(suggestions["best_model"]["name"])
    
    if "other_options" in suggestions:
        for option in suggestions["other_options"]:
            if "name" in option:
                recommended_models.append(option["name"])
    
    # Determine task type
    target_unique = df[target_column].nunique()
    target_dtype = df[target_column].dtype
    is_classification = (target_unique <= 10) or (target_dtype == 'object') or (target_dtype == 'bool')
    task_type = "classification" if is_classification else "regression"
    
    # Determine data size
    num_rows = len(df)
    if num_rows < 1000:
        data_size = "small"
    elif num_rows < 10000:
        data_size = "medium"
    else:
        data_size = "large"
    
    # Return formatted response
    result = {
        "recommended_models": recommended_models,
        "task_type": task_type,
        "data_size": data_size,
        **suggestions  # Include original suggestions for backward compatibility
    }
    
    return result


def get_fallback_model_suggestions(df: pd.DataFrame, target_column: str) -> dict:
    """Fallback rule-based model selection when LLM fails"""
    print("🔧 Using fallback rule-based model selection...")
    
    target_unique = df[target_column].nunique()
    target_dtype = df[target_column].dtype
    
    # Determine if classification or regression
    is_classification = (target_unique <= 10) or (target_dtype == 'object') or (target_dtype == 'bool')
    
    if is_classification:
        print("📊 Detected classification task")
        if target_unique == 2:
            # Binary classification
            return {
                "best_model": {
                    "name": "XGBClassifier",
                    "description": "XGBoost is highly effective for classification tasks with good performance and speed."
                },
                "other_options": [
                    {
                        "name": "RandomForestClassifier",
                        "description": "Robust ensemble method excellent for binary classification with built-in feature importance and handles overfitting well"
                    },
                    {
                        "name": "GradientBoostingClassifier",
                        "description": "High-performance boosting algorithm that often achieves excellent accuracy on structured data"
                    }
                ]
            }
        else:
            # Multi-class classification
            return {
                "best_model": {
                    "name": "RandomForestClassifier",
                    "description": "Excellent for multi-class classification with natural handling of multiple classes and feature importance"
                },
                "other_options": [
                    {
                        "name": "GradientBoostingClassifier",
                        "description": "Powerful boosting method that handles multi-class problems well with high accuracy potential"
                    }
                ]
            }
    else:
        # Regression
        print("📈 Detected regression task")
        return {
            "best_model": {
                "name": "RandomForestRegressor",
                "description": "Robust ensemble method for regression with good handling of non-linear relationships and feature importance"
            },
            "other_options": [
                {
                    "name": "GradientBoostingRegressor",
                    "description": "High-performance boosting algorithm excellent for regression tasks with complex patterns"
                }
            ]
        }


def select_model(filepath: str, target_column: str) -> dict:
    """
    Main function to select the best and alternative ML models for the dataset.

    Returns a dict:
    {
        "recommended_models": ["model1", "model2", ...],
        "task_type": "classification" or "regression",
        "data_size": "small/medium/large",
        "best_model": {"name": "...", "description": "..."},
        "other_options": [ {"name": "...", "description": "..."}, ... ]
    }
    """
    print(f"🎯 Starting model selection for: {filepath}")
    print(f"🎯 Target column: {target_column}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if target_column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {available_columns}")

        # Basic data info
        target_info = {
            "unique_values": df[target_column].nunique(),
            "data_type": str(df[target_column].dtype),
            "missing_values": df[target_column].isnull().sum(),
            "sample_values": df[target_column].value_counts().head(5).to_dict()
        }
        print(f"🎯 Target column info: {target_info}")

        # Take a sample for the LLM input (first 5 rows)
        sample_df = df.head(5)

        # Try LLM-based model suggestion first
        llm_suggestions = call_llm_model_selector(sample_df, target_column)
        print(f"🔍 LLM suggestions received: {llm_suggestions}")

        if llm_suggestions and "best_model" in llm_suggestions:
            print("✅ Using LLM model suggestions")
            # Convert "other_models" to "other_options" for consistency
            if "other_models" in llm_suggestions:
                llm_suggestions["other_options"] = llm_suggestions.pop("other_models")
            
            # Add the expected frontend fields
            result = _format_response(llm_suggestions, df, target_column)
            print(f"🎯 Final formatted result: {result}")
            return result
        
        # Fallback: Rule-based model selection
        print("⚠️  LLM failed or unavailable, using rule-based model selection")
        fallback_result = get_fallback_model_suggestions(df, target_column)
        return _format_response(fallback_result, df, target_column)
        
    except FileNotFoundError as e:
        error_msg = f"Dataset file not found: {filepath}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error in model selection: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

if __name__ == "__main__":
    import pprint

    # Example usage:
    dataset_path = dataset_path = r"D:\AutoMLL\backend\uploads\diabetes_cleaned.csv"

    target_col = "Outcome"

    try:
        model_info = select_model(dataset_path, target_col)
        print("\n🎯 Model Selection Results:")
        pprint.pprint(model_info)
    except Exception as e:
        print(f"❌ Error: {e}")
