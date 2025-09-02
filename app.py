# logistic_regression_spam_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize

# Load data
df = pd.read_csv('spambase.csv')

# Drop duplicates
df = df.drop_duplicates()

# Top 20 correlated features with target
corr = df.corr()['spam'].abs().sort_values(ascending=False)[1:21]
top_20 = list(corr.index)
df_top = df[top_20 + ['spam']]

# SMOTE for class balance
X = df_top.drop('spam', axis=1)
y = df_top['spam']
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
df_balanced = pd.DataFrame(X_res, columns=X.columns)
df_balanced['spam'] = y_res

# Winsorization for outliers
for feature in ["word_freq_remove", "word_freq_money", "capital_run_length_total",
                "char_freq_$", "word_freq_000", "word_freq_receive",
                "word_freq_hp", "word_freq_order", "word_freq_email"]:
    df_balanced[feature] = winsorize(df_balanced[feature], limits=[0, 0.05])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced.drop('spam', axis=1), df_balanced['spam'], 
    test_size=0.2, stratify=df_balanced['spam'], random_state=42
)

# Logistic Regression Model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
importance = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_[0]})
importance = importance.sort_values('Coefficient', ascending=False)
sns.barplot(data=importance, y='Feature', x='Coefficient', palette='coolwarm')
plt.title("Feature Importance")
plt.show()
