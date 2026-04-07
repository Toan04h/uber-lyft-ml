#!/usr/bin/env python
# coding: utf-8

# In[9]:


# CSCE 478 - Introduction to Machine Learning
# Project: Uber & Lyft Cab Price Prediction
# Model: Random Forest Classifier
# Dataset: cab_rides.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# In[2]:


# Import Uber and Lyft cab rides dataset
cabrides = pd.read_csv("cab_rides.csv").dropna()
# Drop unusable columns in the data set
cabrides = cabrides.drop(columns=["id", "time_stamp", "product_id"])
# Handle categorical columns like cab types, name, sources, and destinations (One-hot encode)
cabrides = pd.get_dummies(cabrides)


# In[3]:


# Select input and output features
X = cabrides.drop(columns=["price"])
y = pd.qcut(cabrides["price"], q=3, labels=["Low", "Medium", "High"])

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit a random forest regressor
rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
rfModel.fit(X_train_scaled, y_train)


# In[4]:


# Evaluate Model Performance
predtest = rfModel.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predtest)
print(f"Accuracy: {accuracy:.4f}")
print()
print(classification_report(y_test, predtest,
      target_names=["Low", "Medium", "High"]))


# In[5]:


# Normalized Confusion Matrix

# normalize='true' gives row-wise percentages (each row sums to 1.0)
cm = confusion_matrix(y_test, predtest, normalize='true')

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# In[6]:


# Bootstrapping

# Resample test set 1000 times with replacement to estimate metric stability
# same indices applied to both y_test and predtest to preserve pairing
bootstrapAccuracy = []
bootstrapF1score = []

for _ in range(1000):
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_sample = y_test.iloc[indices]
    pred_sample = predtest[indices]

    accuracy = accuracy_score(y_sample, pred_sample)
    f1 = f1_score(y_sample, pred_sample, average='weighted')

    bootstrapAccuracy.append(accuracy)
    bootstrapF1score.append(f1)

print("Bootstrapped 95% Confidence Intervals")
print(f"Accuracy: {np.mean(bootstrapAccuracy):.4f} (95% CI: {np.percentile(bootstrapAccuracy, 2.5):.4f} - {np.percentile(bootstrapAccuracy, 97.5):.4f})")
print(f"F1 score: {np.mean(bootstrapF1score):.4f} (95% CI: {np.percentile(bootstrapF1score, 2.5):.4f} - {np.percentile(bootstrapF1score, 97.5):.4f})")


# In[7]:


# Create a dataframe with features and their importances
importance_df = pd.DataFrame(
    data={
        "feature": X.columns,
        "importance": rfModel.feature_importances_,
    }
).sort_values("importance", ascending=False)

top10 = importance_df.head(10)


# In[8]:


# Feature importance plot
plt.figure(figsize=(10, 6))

sns.barplot(data=top10, x='importance', y='feature')

plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

