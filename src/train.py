import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocessing import DrugDataPreprocessor, save_preprocessor, FEATURE_COLS

def main():
    print("=" * 50)
    print("      DRUG RECOMMENDATION MODEL TRAINING      ")
    print("=" * 50)
    
    # 1. Load raw data
    raw_path = "data/raw/drug200.csv"
    if not os.path.exists(raw_path):
        raw_path = "../data/raw/drug200.csv"
        
    print(f"[1/5] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Data shape: {df.shape}")
    
    # 2. Preprocess data
    print("[2/5] Preprocessing features...")
    preprocessor = DrugDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Save transformed data
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    df_transformed = df.copy()
    df_transformed['Sex'] = X[:, 1]
    df_transformed['BP'] = X[:, 2]
    df_transformed['Cholesterol'] = X[:, 3]
    df_transformed.to_csv(os.path.join(processed_dir, "drug200_transformed.csv"), index=False)
    print(f"Saved processed dataset to {os.path.join(processed_dir, 'drug200_transformed.csv')}")
    
    # Save preprocessor
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    save_preprocessor(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    print(f"Saved preprocessor to {os.path.join(models_dir, 'preprocessor.pkl')}")
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 4. Train Models
    print("[3/5] Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_f1 = f1_score(y_test, dt_preds, average='weighted')
    print(f"Decision Tree Test Accuracy: {dt_acc * 100:.2f}% | F1-Score: {dt_f1:.4f}")
    
    print("[4/5] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}% | F1-Score: {rf_f1:.4f}")
    
    # Save Models
    joblib.dump(dt_model, os.path.join(models_dir, "DecisionTree.pkl"))
    joblib.dump(rf_model, os.path.join(models_dir, "RandomForest.pkl"))
    print(f"Saved models to {models_dir}/")
    
    # 5. Generate Figures and Reports
    print("[5/5] Generating figures and reports...")
    figures_dir = "reports/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Figure 1: Decision Tree Structure
    plt.figure(figsize=(16, 10))
    plot_tree(
        dt_model, 
        feature_names=FEATURE_COLS, 
        class_names=dt_model.classes_, 
        filled=True, 
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Model Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    tree_fig_path = os.path.join(figures_dir, "decision_tree.png")
    plt.savefig(tree_fig_path, dpi=300)
    plt.close()
    
    # Figure 2: Confusion Matrix
    cm = confusion_matrix(y_test, dt_preds, labels=dt_model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
    plt.title("Decision Tree Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Drug")
    plt.ylabel("Actual Drug")
    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    
    # Figure 3: Feature Importances
    plt.figure(figsize=(9, 5))
    importances = pd.Series(dt_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    importances.plot(kind='barh', color='#2b5c8f')
    plt.title("Feature Importance (Decision Tree)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    fi_fig_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(fi_fig_path, dpi=300)
    plt.close()
    
    # Save Report Markdown
    report_content = f"""# Model Performance Report

## Overview
- **Dataset**: `drug200.csv` (200 records, 5 features)
- **Task**: Multi-class Classification (Drug Recommendation: `drugA`, `drugB`, `drugC`, `drugX`, `drugY`)
- **Algorithms**: Decision Tree Classifier, Random Forest Classifier

## Model Performance Summary

| Model | Accuracy | Weighted F1-Score | CV Accuracy (5-Fold) |
|---|---|---|---|
| **Decision Tree** | **{dt_acc*100:.2f}%** | **{dt_f1:.4f}** | **{np.mean(cross_val_score(dt_model, X, y, cv=5))*100:.2f}%** |
| **Random Forest** | **{rf_acc*100:.2f}%** | **{rf_f1:.4f}** | **{np.mean(cross_val_score(rf_model, X, y, cv=5))*100:.2f}%** |

## Detailed Decision Tree Classification Report
```
{classification_report(y_test, dt_preds)}
```

## Generated Visualizations
- Decision Tree Structure: `reports/figures/decision_tree.png`
- Confusion Matrix: `reports/figures/confusion_matrix.png`
- Feature Importance: `reports/figures/feature_importance.png`
"""
    report_path = "reports/model_performance.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Successfully generated model performance report at {report_path}")
    print("=" * 50)
    print("TRAINING & EVALUATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
