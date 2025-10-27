#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_selector import select_model

# Test the model selector
try:
    result = select_model("uploads/diabetes_cleaned.csv", "Outcome")
    print("✅ Model selector working!")
    print("Result:", result)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()