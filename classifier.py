import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import datetime

# ---- CONFIG ----
DATA_PATH = "weather.csv"   # <-- change to your CSV path
TARGET_COL = "RainTomorrow" # expect values 'Yes'/'No' or 1/0
DATE_COL = "Date"           # optional: if present, used to create temporal features
RANDOM_STATE = 42

# ---- LOAD ----
df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
# normalize target to 0/1
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = df[TARGET_COL].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
else:
    df[TARGET_COL] = df[TARGET_COL].astype(int)

# optional: parse date and add day of year, month
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df['day_of_year'] = df[DATE_COL].dt.dayofyear
    df['month'] = df[DATE_COL].dt.month
else:
    df['day_of_year'] = 0
    df['month'] = 0

# ---- BASIC FEATURE ENGINEERING ----
# Example common numeric columns (modify according to your dataset)
numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c != TARGET_COL]
# drop a few that are not predictive or too sparse if present
for c in ['Evaporation','Sunshine']:
    if c in numeric_cols: numeric_cols.remove(c)

# Example categorical columns
categorical_cols = [c for c in df.columns if df[c].dtype == object and c != TARGET_COL and c != DATE_COL]
# If dataset has column 'RainToday' convert to binary numeric
if 'RainToday' in df.columns:
    df['RainToday'] = df['RainToday'].map(lambda x: 1 if str(x).strip().lower() in ("yes","y","1","true","t") else 0)
    if 'RainToday' not in numeric_cols: numeric_cols.append('RainToday')
    if 'RainToday' in categorical_cols: categorical_cols.remove('RainToday')

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# LAG FEATURES (yesterday's rainfall)
df = df.sort_values(by=DATE_COL) if DATE_COL in df.columns else df
if 'RainToday' in df.columns:
    df['RainYesterday'] = df['RainToday'].shift(1).fillna(0)
    if 'RainYesterday' not in numeric_cols:
        numeric_cols.append('RainYesterday')

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COL])
print("After dropping NA target:", df.shape)

# Split (time-aware if date present, else random stratified)
if DATE_COL in df.columns and df[DATE_COL].notna().all():
    # use the latest 20% as test set (temporal split)
    df = df.sort_values(DATE_COL)
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
else:
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=RANDOM_STATE)

X_train = train_df[numeric_cols + categorical_cols]
y_train = train_df[TARGET_COL]
X_test = test_df[numeric_cols + categorical_cols]
y_test = test_df[TARGET_COL]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Model pipeline
clf = Pipeline(steps=[
    ('preproc', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
])

# Quick baseline fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Classification report (baseline):")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Precision-Recall AUC
prec, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, prec)
print("PR AUC:", pr_auc)

# Save model
joblib.dump(clf, "rainfall_classifier.joblib")
print("Saved model to rainfall_classifier.joblib")
