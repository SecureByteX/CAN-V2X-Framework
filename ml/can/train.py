import numpy as np
import pandas as pd
import joblib
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from ml.preprocessing import DataPreprocessor
from ml.feature_engineering import FeatureExtractor
from evaluation.can_evaluation import ModelEvaluator

def train_can_models():
    dataset_path = (
        config.CAN_TRAINING_DATA_PATH
        if os.path.exists(config.CAN_TRAINING_DATA_PATH)
        else config.CAN_RAW_DATA_PATH
    )
    print(f"[CAN Training] Loading CAN dataset from: {dataset_path}")
    preprocessor = DataPreprocessor(target_column='label')
    df_raw = preprocessor.load_data(dataset_path)
    df_clean = preprocessor.clean_data(df_raw)
    
    print("[CAN Training] Extracting domain features...")
    X, y, df_features = FeatureExtractor.extract_can_features(df_clean)
    
    # Save processed feature dataset
    os.makedirs(os.path.dirname(config.CAN_PROCESSED_DATA_PATH), exist_ok=True)
    df_features.to_csv(config.CAN_PROCESSED_DATA_PATH, index=False)
    print(f"[CAN Training] Saved processed features to: {config.CAN_PROCESSED_DATA_PATH}")

    # Encode y labels to integer for XGBoost compatibility
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    train_frame, test_frame = preprocessor.chronological_split(df_features, 'timestamp')
    X_train = train_frame[X.columns]
    X_test = test_frame[X.columns]
    y_train = le.transform(train_frame['label'])
    y_test = le.transform(test_frame['label'])
    
    # Define models to train
    models = {
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    else:
        models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    results_list = []
    best_model = None
    best_score = -1.0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n[CAN Training] Fitting {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        normal_label = int(le.transform(['NORMAL'])[0])
        eval_res = ModelEvaluator.evaluate_model(
            model, X_test, y_test, model_name=name, normal_label=normal_label
        )
        ModelEvaluator.print_evaluation_report(eval_res)
        results_list.append(eval_res)
        
        if eval_res['f1_macro'] > best_score:
            best_score = eval_res['f1_macro']
            best_model = model
            best_name = name

    print(f"\n[CAN Training] Best Performing CAN Model: {best_name} (macro F1: {best_score:.4f})")
    
    # Save best model along with label encoder and feature columns metadata
    model_artifact = {
        'model': best_model,
        'model_name': best_name,
        'label_encoder': le,
        'feature_names': list(X.columns),
        'evaluation_results': results_list,
        'dataset_name': 'HCRL Car-Hacking Dataset',
        'dataset_rows': len(df_features),
        'train_rows': len(train_frame),
        'test_rows': len(test_frame),
        'class_distribution': {
            str(label): int(count)
            for label, count in df_features['label'].value_counts().sort_index().items()
        },
        'split_strategy': 'Last 20% by timestamp within each label',
        'selection_metric': 'macro_f1',
    }
    
    os.makedirs(os.path.dirname(config.CAN_MODEL_PATH), exist_ok=True)
    joblib.dump(model_artifact, config.CAN_MODEL_PATH)
    test_frame.to_csv(config.CAN_HCRL_HOLDOUT_PATH, index=False)
    print(f"[CAN Training] Saved model artifact to: {config.CAN_MODEL_PATH}")
    print(f"[CAN Training] Saved held-out class samples to: {config.CAN_HCRL_HOLDOUT_PATH}")
    
    return model_artifact

if __name__ == '__main__':
    train_can_models()
