import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load data
df = pd.read_csv('heloc.csv')

# 2. Target mapping ('Bad' -> 0, 'Good' -> 1)
df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})

print(df['RiskPerformance'].value_counts())
print("percentage of negative class: ", df['RiskPerformance'].value_counts()[0] / len(df['RiskPerformance'])*100) ## 52.19 %

# 3. Handle Special Values
# FICO HELOC uses -9, -8, -7 for missing or special statuses.
# We replace values < 0 with NaN for proper continuous handling.
X = df.drop(columns=['RiskPerformance'])
y = df['RiskPerformance']

X_replace = X.copy()
X_replace[X_replace < 0] = np.nan

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_replace, y, test_size=0.2, random_state=42)

# 5. Impute and Scale
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# 6. Train Linear Model (Logistic Regression)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

# 7. Train Non-Linear Models
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_preds = gb.predict(X_test_scaled)
gb_probs = gb.predict_proba(X_test_scaled)[:, 1]

# 8. Print Results
print(f"LR Accuracy: {accuracy_score(y_test, lr_preds):.4f}, AUC: {roc_auc_score(y_test, lr_probs):.4f}")
print(f"RF Accuracy: {accuracy_score(y_test, rf_preds):.4f}, AUC: {roc_auc_score(y_test, rf_probs):.4f}")
print(f"GB Accuracy: {accuracy_score(y_test, gb_preds):.4f}, AUC: {roc_auc_score(y_test, gb_probs):.4f}")