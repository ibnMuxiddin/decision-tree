# Model Performance Report

## Overview
- **Dataset**: `drug200.csv` (200 records, 5 features)
- **Task**: Multi-class Classification (Drug Recommendation: `drugA`, `drugB`, `drugC`, `drugX`, `drugY`)
- **Algorithms**: Decision Tree Classifier, Random Forest Classifier

## Model Performance Summary

| Model | Accuracy | Weighted F1-Score | CV Accuracy (5-Fold) |
|---|---|---|---|
| **Decision Tree** | **98.33%** | **0.9830** | **98.50%** |
| **Random Forest** | **98.33%** | **0.9830** | **98.50%** |

## Detailed Decision Tree Classification Report
```
              precision    recall  f1-score   support

       drugA       0.88      1.00      0.93         7
       drugB       1.00      0.80      0.89         5
       drugC       1.00      1.00      1.00         5
       drugX       1.00      1.00      1.00        16
       drugY       1.00      1.00      1.00        27

    accuracy                           0.98        60
   macro avg       0.97      0.96      0.96        60
weighted avg       0.99      0.98      0.98        60

```

## Generated Visualizations
- Decision Tree Structure: `reports/figures/decision_tree.png`
- Confusion Matrix: `reports/figures/confusion_matrix.png`
- Feature Importance: `reports/figures/feature_importance.png`
