import os
import sys
# Add the current directory and its parent to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'services'))

from services.trainer import train_model

def test_shap():
    dataset = "uploads/diabetes.csv"
    target = "Outcome"
    model = "RandomForestClassifier"
    
    print(f"🧪 Testing SHAP on {dataset}...")
    try:
        metrics, pipeline, encoder = train_model(
            dataset, target, model, tune_hyperparams=False
        )
        
        print("\n✅ Training metrics:")
        print(f"Accuracy: {metrics.get('accuracy')}")
        
        print("\n🧠 SHAP Insights (Top 5):")
        insights = metrics.get('insights', [])
        for i, item in enumerate(insights[:5]):
            print(f"{i+1}. {item['feature']}: {item['importance']:.4f}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_shap()
