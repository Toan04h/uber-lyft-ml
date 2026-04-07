# Uber & Lyft Cab Price Classification

Trained a Random Forest Classifier to classify Uber and Lyft 
cab rides into Low, Medium, and High price tiers using 637,976 
ride records.

## Results
- Accuracy: 91.34% (95% CI: 91.20% – 91.49%)
- F1 Score: 91.32% (95% CI: 91.17% – 91.46%)

## Tech Stack
Python, scikit-learn, pandas, numpy, matplotlib, seaborn

## Files
- CSCE478_randomforestModel.py — full pipeline
- confusion_matrix.png — normalized confusion matrix
- feature_importance.png — top 10 feature importances

## Key Finding
Distance is the strongest predictor of price tier, followed 
by ride tier (Black SUV, Lux Black XL).
