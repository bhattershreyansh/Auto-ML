import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
from typing import Dict, List, Any
from scipy import stats

load_dotenv()
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

PROMPT_TEMPLATE = """
You are an expert data analyst. Here's the dataset summary:

{summary}

Please:
- Briefly describe the dataset structure
- Suggest the most likely target column for ML
- Recommend 3-5 informative graph types (with relevant columns) for data exploration
- Point out any key issues (nulls, imbalance, outliers, data leakage risks)
- Suggest feature engineering opportunities

Respond in this JSON template:
{{
"target_column": "...",
"dataset_description": "...",
"key_issues": ["issue1", "issue2"],
"feature_suggestions": ["suggestion1", "suggestion2"],
"graphs": [{{"type": "...", "columns": ["col1", "col2"]}}, ...]
}}
"""


def call_llm_for_analysis(summary: str) -> dict:
    """
    Calls LLM to get comprehensive dataset analysis.
    Returns a dict with analysis and visualization suggestions.
    """
    if not groq_client:
        logging.warning("No LLM client available. Skipping LLM suggestions.")
        return {}
    
    prompt = PROMPT_TEMPLATE.format(summary=summary)
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        
        text = response.choices[0].message.content
        
        # Safely parse JSON section
        import re
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            logging.warning("LLM did not return proper JSON. Output: %s", text)
            return {}
    
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return {}


def detect_outliers(df: pd.DataFrame) -> Dict[str, int]:
    """
    Detect outliers in numerical columns using IQR method.
    Returns dict of column: outlier_count.
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers[col] = int(outlier_count)
    
    return outliers


def check_class_imbalance(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Check for class imbalance in target column.
    """
    if target_col not in df.columns:
        return {}
    
    value_counts = df[target_col].value_counts()
    total = len(df)
    
    imbalance_info = {
    "class_distribution": value_counts.to_dict(),
    "class_percentages": (value_counts / total * 100).round(2).to_dict(),
    "is_imbalanced": bool((value_counts.max() / value_counts.min()) > 3) if len(value_counts) > 1 else False
}
    
    return imbalance_info


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute correlation matrix for numeric columns.
    Returns high correlations (>0.7 or <-0.7).
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) < 2:
        return {}
    
    corr_matrix = numeric_df.corr()
    
    # Find high correlations (excluding diagonal)
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corrs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": round(float(corr_val), 3)
                })
    
    return {
        "high_correlations": high_corrs,
        "correlation_matrix": corr_matrix.round(3).to_dict()
    }


def generate_enhanced_visualizations(df: pd.DataFrame, target_col: str = None) -> Dict[str, str]:
    """
    Generate comprehensive Plotly visualizations for EDA.
    Returns dict of {viz_name: plotly_json}.
    """
    graphs = {}
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    try:
        # 1. Missing Data Heatmap
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            fig = go.Figure(data=[go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                marker_color='indianred'
            )])
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Columns",
                yaxis_title="Missing Count",
                height=400
            )
            graphs["missing_values"] = pio.to_json(fig)
        
        # 2. Target Distribution
        if target_col and target_col in df.columns:
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
                fig = px.bar(
                    df[target_col].value_counts().reset_index(),
                    x='index',
                    y=target_col,
                    title=f"Target Distribution: {target_col}",
                    labels={'index': target_col, target_col: 'Count'},
                    color=target_col,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
            else:
                fig = px.histogram(
                    df,
                    x=target_col,
                    title=f"Target Distribution: {target_col}",
                    nbins=50,
                    marginal="box"
                )
            graphs["target_distribution"] = pio.to_json(fig)
        
        # 3. Correlation Heatmap
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                labels=dict(color="Correlation")
            )
            fig.update_layout(height=600)
            graphs["correlation_heatmap"] = pio.to_json(fig)
        
        # 4. Numeric Feature Distributions
        if len(numeric_cols) > 0:
            # Take first 4 numeric columns
            cols_to_plot = numeric_cols[:4]
            fig = go.Figure()
            
            for col in cols_to_plot:
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title="Numeric Feature Distributions (Box Plots)",
                yaxis_title="Value",
                height=500,
                showlegend=True
            )
            graphs["numeric_distributions"] = pio.to_json(fig)
        
        # 5. Pairplot/Scatter Matrix (first 3-4 numeric features)
        if len(numeric_cols) >= 2:
            cols_for_scatter = numeric_cols[:min(4, len(numeric_cols))]
            fig = px.scatter_matrix(
                df,
                dimensions=cols_for_scatter,
                title="Feature Relationships (Scatter Matrix)",
                height=700
            )
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            graphs["scatter_matrix"] = pio.to_json(fig)
        
        # 6. Categorical Feature Analysis
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 15:
                value_counts = df[cat_col].value_counts().head(10)
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {cat_col}",
                    hole=0.3
                )
                graphs[f"categorical_{cat_col}"] = pio.to_json(fig)
        
        # 7. Feature vs Target (if target is categorical)
        if target_col and target_col in categorical_cols and len(numeric_cols) > 0:
            num_col = numeric_cols[0]
            fig = px.box(
                df,
                x=target_col,
                y=num_col,
                title=f"{num_col} by {target_col}",
                color=target_col,
                points="outliers"
            )
            graphs["target_vs_feature"] = pio.to_json(fig)
    
    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")
    
    return graphs


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    quality_report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_percentage": round(float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2)
    }
    
    # Columns with missing values
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    quality_report["columns_with_missing"] = {
        str(col): int(count) for col, count in missing_cols.items()
    }
    
    # Constant columns (zero variance)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    quality_report["constant_columns"] = constant_cols
    
    # High cardinality columns (potential ID columns)
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > 0.9 * len(df)]
    quality_report["high_cardinality_columns"] = high_card_cols
    
    return quality_report


def analyze_dataset(filepath: str) -> dict:
    """
    Comprehensive dataset analysis with enhanced EDA and visualizations.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        return {"error": str(e)}
    
    logging.info(f"📊 Analyzing dataset: {df.shape}")
    
    # Basic statistics
    basic_stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "unique_counts": df.nunique().to_dict(),
        "sample_data": df.head(5).to_dict('records')
    }
    
    # Data quality report
    quality_report = generate_data_quality_report(df)
    
    # Statistical summary
    summary_stats = df.describe(include='all', percentiles=[.25, .5, .75]).to_dict()
    
    # Call LLM for intelligent analysis
    llm_summary_text = df.describe(include='all').transpose().to_string()
    llm_output = call_llm_for_analysis(llm_summary_text)
    
    # Determine target column
    if llm_output.get("target_column") and llm_output["target_column"] in df.columns:
        target = llm_output["target_column"]
    else:
        # Fallback heuristic
        candidates = [c for c in df.columns if df[c].dtype == 'object' and 2 <= df[c].nunique() <= 20]
        if not candidates:
            candidates = [c for c in df.columns if 2 <= df[c].nunique() <= 20]
        target = candidates[0] if candidates else df.columns[-1]
    
    # Outlier detection
    outliers = detect_outliers(df)
    
    # Class imbalance check
    imbalance_info = check_class_imbalance(df, target)
    
    # Correlation analysis
    correlation_info = correlation_analysis(df)
    
    # Generate visualizations
    visualizations = generate_enhanced_visualizations(df, target)
    
    # Compile final report
    analysis_report = {
        "basic_statistics": basic_stats,
        "data_quality": quality_report,
        "suggested_target": target,
        "outliers": outliers,
        "class_imbalance": imbalance_info,
        "correlations": correlation_info,
        "llm_insights": {
            "description": llm_output.get("dataset_description", ""),
            "key_issues": llm_output.get("key_issues", []),
            "feature_suggestions": llm_output.get("feature_suggestions", [])
        },
        "visualizations": visualizations,
        "summary_statistics": summary_stats
    }
    
    logging.info(f"✅ Analysis complete. Generated {len(visualizations)} visualizations.")
    
    return analysis_report


if __name__ == "__main__":
    # Test the analyzer
    test_file = input("Enter CSV file path: ").strip()
    result = analyze_dataset(test_file)
    
    print("\n📊 Analysis Summary:")
    print(f"- Rows: {result['data_quality']['total_rows']}")
    print(f"- Columns: {result['data_quality']['total_columns']}")
    print(f"- Missing: {result['data_quality']['total_missing']}")
    print(f"- Suggested Target: {result['suggested_target']}")
    print(f"- Visualizations: {len(result['visualizations'])}")
