#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_comparator import compare_models

# Test the model comparison
try:
    result = compare_models(
        filepath="uploads/diabetes_cleaned.csv",
        target_column="Outcome",
        model_names=["RandomForestClassifier", "GradientBoostingClassifier"],
        test_size=0.2,
        tune_hyperparams=False,
        cv_folds=3
    )
    print("✅ Model comparison working!")
    print("Result structure:")
    print(f"- Task type: {result.get('task_type')}")
    print(f"- Leaderboard entries: {len(result.get('leaderboard', []))}")
    print(f"- Models tried: {result.get('models_tried')}")
    
    if result.get('leaderboard'):
        print("\nLeaderboard:")
        for i, model in enumerate(result['leaderboard']):
            print(f"  {i+1}. {model['model_name']}: {model['score']:.4f} (time: {model['train_time']:.2f}s)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()