import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # Helps balance classes

# Load the dataset and remove the 'LoanID' column if present.
df = pd.read_csv("loan_default.csv")
df = df.drop(columns=['LoanID'], errors='ignore')

# Encode any categorical columns using LabelEncoder.
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features (X) and target (y).
TARGET = 'Default'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split into train and test sets. We use stratify to keep class proportions similar.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using SMOTE on the training data.
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale the features to have mean 0 and standard deviation 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Quick check of class distributions.
print("Original Class Distribution:", Counter(y))
print("Train Class Distribution:", Counter(y_train))
print("Test Class Distribution:", Counter(y_test))

# Train three models: Decision Tree, Random Forest, and Gradient Boosting.
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Compute accuracy and ROC AUC for each model.
dt_acc = accuracy_score(y_test, y_pred_dt)
rf_acc = accuracy_score(y_test, y_pred_rf)
gb_acc = accuracy_score(y_test, y_pred_gb)

dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:,1])
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:,1])

# Check if there's a big difference between training and testing accuracy (potential overfitting).
train_acc_dt = accuracy_score(y_train, dt_model.predict(X_train))
train_acc_rf = accuracy_score(y_train, rf_model.predict(X_train))
train_acc_gb = accuracy_score(y_train, gb_model.predict(X_train))

overfitting_check = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Gradient Boosting"],
    "Train Accuracy": [train_acc_dt, train_acc_rf, train_acc_gb],
    "Test Accuracy": [dt_acc, rf_acc, gb_acc],
    "Difference": [
        abs(train_acc_dt - dt_acc),
        abs(train_acc_rf - rf_acc),
        abs(train_acc_gb - gb_acc)
    ]
})
print(overfitting_check)

# Summarize final results.
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Gradient Boosting"],
    "Accuracy": [dt_acc, rf_acc, gb_acc],
    "ROC AUC Score": [dt_auc, rf_auc, gb_auc]
})
print(results)

# Show the confusion matrix for the Gradient Boosting model.
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()
