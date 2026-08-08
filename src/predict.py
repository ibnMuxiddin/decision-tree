import os
import argparse
import joblib
import numpy as np
import pandas as pd
from preprocessing import load_preprocessor, DrugDataPreprocessor

def predict_drug(age: int, sex: str, bp: str, cholesterol: str, na_to_k: float, model_type: str = 'dt') -> tuple[str, dict[str, float]]:
    """Predicts recommended drug for a given patient profile."""
    # Find paths relative to current script or cwd
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    model_path = os.path.join(base_dir, "models", "DecisionTree.pkl" if model_type.lower() == 'dt' else "RandomForest.pkl")
    
    if not os.path.exists(prep_path):
        prep_path = "models/preprocessor.pkl"
        model_path = f"models/{'DecisionTree.pkl' if model_type.lower() == 'dt' else 'RandomForest.pkl'}"
        
    preprocessor = load_preprocessor(prep_path)
    model = joblib.load(model_path)
    
    features = preprocessor.transform_sample(age, sex, bp, cholesterol, na_to_k)
    
    pred_class = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prob_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
    
    return pred_class, prob_dict

def main():
    parser = argparse.ArgumentParser(description="Patient Drug Recommendation System CLI")
    parser.add_argument("--age", type=int, help="Patient age (e.g. 47)")
    parser.add_argument("--sex", type=str, choices=['F', 'M', 'f', 'm'], help="Patient sex (F or M)")
    parser.add_argument("--bp", type=str, choices=['LOW', 'NORMAL', 'HIGH', 'low', 'normal', 'high'], help="Blood pressure level")
    parser.add_argument("--cholesterol", type=str, choices=['NORMAL', 'HIGH', 'normal', 'high'], help="Cholesterol level")
    parser.add_argument("--na_to_k", type=float, help="Sodium to potassium ratio (e.g. 13.093)")
    parser.add_argument("--model", type=str, default="dt", choices=["dt", "rf"], help="Model type: 'dt' (Decision Tree) or 'rf' (Random Forest)")

    args = parser.parse_args()

    if None in (args.age, args.sex, args.bp, args.cholesterol, args.na_to_k):
        print("Interactive Mode - Please enter patient details:")
        try:
            age = int(input("Age (e.g., 47): "))
            sex = input("Sex (F/M): ").strip().upper()
            bp = input("Blood Pressure (LOW/NORMAL/HIGH): ").strip().upper()
            cholesterol = input("Cholesterol (NORMAL/HIGH): ").strip().upper()
            na_to_k = float(input("Na to K Ratio (e.g., 15.2): "))
        except Exception as e:
            print(f"Error reading inputs: {e}")
            return
    else:
        age, sex, bp, cholesterol, na_to_k = args.age, args.sex, args.bp, args.cholesterol, args.na_to_k

    recommended_drug, probabilities = predict_drug(age, sex, bp, cholesterol, na_to_k, model_type=args.model)

    print("\n" + "=" * 45)
    print("        PREDICTION RESULTS")
    print("=" * 45)
    print(f"Patient Profile : Age={age}, Sex={sex.upper()}, BP={bp.upper()}, Cholesterol={cholesterol.upper()}, Na_to_K={na_to_k}")
    print(f"Model Used      : {'Decision Tree' if args.model=='dt' else 'Random Forest'}")
    print(f"RECOMMENDED DRUG: >>> {recommended_drug} <<<")
    print("-" * 45)
    print("Class Probabilities:")
    for drug, prob in probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"  {drug:<8}: {prob*100:6.2f}% {bar}")
    print("=" * 45)

if __name__ == "__main__":
    main()
