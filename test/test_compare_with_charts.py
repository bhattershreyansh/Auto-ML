#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_comparator import compare_models

def test_compare_with_charts():
    print("🧪 Testing compare_models with chart data...")
    
    try:
        result = compare_models(
            filepath="uploads/diabetes_cleaned.csv",
            target_column="Outcome",
            model_names=["XGBClassifier", "RandomForestClassifier"],  # Test with just 2 models
            test_size=0.2,
            tune_hyperparams=False,
            cv_folds=3
        )
        
        print("✅ Compare models completed!")
        print(f"📊 Task type: {result.get('task_type')}")
        print(f"🏆 Models in leaderboard: {len(result.get('leaderboard', []))}")
        print(f"📈 Chart data entries: {len(result.get('chart_data', []))}")
        print(f"🔗 Correlation data keys: {list(result.get('correlation_data', {}).keys())}")
        
        if 'correlation_data' in result and 'correlation_matrix' in result['correlation_data']:
            corr_entries = len(result['correlation_data']['correlation_matrix'])
            print(f"🔗 Correlation matrix entries: {corr_entries}")
            print(f"🔗 Columns: {result['correlation_data']['columns']}")
            print(f"🎯 Target correlations: {len(result['correlation_data']['target_correlations'])}")
        
        if 'chart_data' in result:
            print("📊 Chart data sample:")
            for i, entry in enumerate(result['chart_data'][:2]):
                print(f"   {i+1}. {entry}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_compare_with_charts()