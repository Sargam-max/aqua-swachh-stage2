"""
Aqua-Swachh Machine Learning Model Training Pipeline
====================================================
Implements ML best practices for Stage 2 filter condition classification:
1. Data loading and schema validation
2. Strict featurization ordering (Train/Test split before scaler fitting)
3. Model baseline comparison:
   - Baseline 1: Dummy (Majority Class)
   - Baseline 2: Logistic Regression (linear baseline)
   - Primary: Random Forest Classifier (ensemble)
4. Cross-validation & comprehensive evaluation (F1, Precision, Recall, Confusion Matrix)
5. Model export with joblib
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from model.features import extract_features, FEATURE_COLUMNS

def train_and_evaluate(
    data_path: str = "data/sensor_data.csv",
    model_output_dir: str = "model/saved_models",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    print("=" * 60, flush=True)
    print("Aqua-Swachh: Filter Health ML Training Pipeline", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please generate or provide sensor CSV first.")
        
    df = pd.read_csv(data_path)
    print(f"\n[1/5] Loaded dataset from {data_path} with {len(df)} rows.", flush=True)
    
    # 2. Data Cleaning & Validation (Per ML Best Practices)
    target_col = 'filter_status'
    required_cols = ['tds_in', 'tds_out', 'temperature', 'flow_rate', target_col]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
    # Check nulls
    null_counts = df[required_cols].isnull().sum()
    print("Null value counts per feature:", flush=True)
    for col, count in null_counts.items():
        print(f"  - {col}: {count}", flush=True)
    if null_counts.sum() > 0:
        print("Dropping rows with missing values...", flush=True)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        
    # Check physical validity (sanity rules)
    valid_mask = (df['tds_in'] > 0) & (df['tds_out'] >= 0) & (df['temperature'] > 0) & (df['flow_rate'] >= 0)
    invalid_rows = (~valid_mask).sum()
    if invalid_rows > 0:
        print(f"Filtering out {invalid_rows} physically invalid sensor rows.", flush=True)
        df = df[valid_mask].reset_index(drop=True)
        
    print(f"Class distribution in clean dataset:\n{df[target_col].value_counts()}", flush=True)
    
    # 3. Physics Feature Extraction
    print("\n[2/5] Extracting physics-informed features...", flush=True)
    X_df = extract_features(df)
    y = df[target_col].values
    
    classes = ['Normal', 'Monitor', 'Service Required']
    
    # 4. Strict Train/Test Split BEFORE Scaling/Fitting
    print(f"\n[3/5] Splitting data: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test (Stratified)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}", flush=True)
    
    # 5. Baseline & Model Comparison
    print("\n[4/5] Training & Evaluating Model Candidates:", flush=True)
    
    models = {
        "Dummy (Majority Class Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DummyClassifier(strategy='most_frequent'))
        ]),
        "Logistic Regression (Linear Baseline)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state))
        ]),
        "Random Forest Classifier (Primary Model)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_split=4,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=1
            ))
        ])
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    comparison_results = {}
    
    for name, pipeline in models.items():
        print(f"Evaluating {name}...", flush=True)
        # Cross validation on Training set only
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
        
        # Fit on entire Training set
        pipeline.fit(X_train, y_train)
        
        # Evaluate on unseen Test set
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        
        comparison_results[name] = {
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'test_accuracy': float(test_acc),
            'test_f1_macro': float(test_f1),
            'pipeline': pipeline,
            'y_pred': y_pred
        }
        print(f"  -> 5-Fold CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})", flush=True)
        print(f"  -> Test Accuracy:     {test_acc:.4f}", flush=True)
        print(f"  -> Test Macro F1:      {test_f1:.4f}", flush=True)
        
    best_model_name = "Random Forest Classifier (Primary Model)"
    best_pipeline = comparison_results[best_model_name]['pipeline']
    best_y_pred = comparison_results[best_model_name]['y_pred']
    
    print("\n" + "=" * 60, flush=True)
    print(f"Detailed Classification Report for {best_model_name}:", flush=True)
    print("=" * 60, flush=True)
    cls_report = classification_report(y_test, best_y_pred, target_names=classes)
    print(cls_report, flush=True)
    
    cm = confusion_matrix(y_test, best_y_pred, labels=classes)
    print("Confusion Matrix (Rows: Actual, Columns: Predicted):", flush=True)
    print(f"Labels: {classes}", flush=True)
    print(cm, flush=True)
    
    # Feature Importances from Random Forest
    rf_clf = best_pipeline.named_steps['clf']
    importances = rf_clf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances (Random Forest):", flush=True)
    for feat, imp in feat_imp:
        print(f"  - {feat:22s}: {imp:.4f}", flush=True)
        
    # 6. Save Model Artifacts
    print("\n[5/5] Exporting model and artifacts...", flush=True)
    os.makedirs(model_output_dir, exist_ok=True)
    model_file = os.path.join(model_output_dir, "random_forest_filter_model.joblib")
    joblib.dump(best_pipeline, model_file)
    print(f"Saved trained pipeline to: {model_file}", flush=True)
    
    metrics_summary = {
        'best_model': best_model_name,
        'feature_columns': FEATURE_COLUMNS,
        'classes': classes,
        'test_accuracy': comparison_results[best_model_name]['test_accuracy'],
        'test_macro_f1': comparison_results[best_model_name]['test_f1_macro'],
        'confusion_matrix': cm.tolist(),
        'feature_importances': {feat: float(imp) for feat, imp in feat_imp},
        'comparison': {
            k: {
                'cv_f1_mean': v['cv_f1_mean'],
                'cv_f1_std': v['cv_f1_std'],
                'test_accuracy': v['test_accuracy'],
                'test_f1_macro': v['test_f1_macro']
            } for k, v in comparison_results.items()
        }
    }
    
    metrics_path = os.path.join(model_output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}", flush=True)
    
    return metrics_summary

if __name__ == "__main__":
    train_and_evaluate()
