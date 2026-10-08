import numpy as np
import pandas as pd
import joblib
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
from evaluation.v2x_evaluation import V2XModelEvaluator


REQUIRED_V2X_COLUMNS = {
    'timestamp', 'message_type', 'sender_id', 'receiver_id', 'latitude',
    'longitude', 'speed', 'heading', 'acceleration', 'sequence_number', 'label'
}


def select_training_dataset(raw_path, processed_path):
    for candidate in (raw_path, processed_path):
        if not os.path.exists(candidate):
            continue
        columns = set(pd.read_csv(candidate, nrows=0).columns)
        if REQUIRED_V2X_COLUMNS.issubset(columns):
            return candidate
    raise ValueError(
        'No V2X training CSV has the required labeled message columns: '
        + ', '.join(sorted(REQUIRED_V2X_COLUMNS))
    )


def train_v2x_models():
    dataset_path = select_training_dataset(
        config.V2X_RAW_DATA_PATH,
        config.V2X_PROCESSED_DATA_PATH
    )
    print(f"[V2X Training] Loading schema-compatible dataset: {dataset_path}")
    preprocessor = DataPreprocessor(target_column='label')
    df_raw = preprocessor.load_data(dataset_path)
    df_clean = preprocessor.clean_data(df_raw)
    
    print("[V2X Training] Extracting domain features...")
    X, y, df_features = FeatureExtractor.extract_v2x_features(df_clean)
    
    # Save processed features
    os.makedirs(os.path.dirname(config.V2X_PROCESSED_DATA_PATH), exist_ok=True)
    df_features.to_csv(config.V2X_PROCESSED_DATA_PATH, index=False)
    print(f"[V2X Training] Saved processed features to: {config.V2X_PROCESSED_DATA_PATH}")

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(X, y_encoded)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
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
        print(f"\n[V2X Training] Fitting {name}...")
        model.fit(X_train, y_train)
        
        normal_label = int(le.transform(['NORMAL'])[0])
        eval_res = V2XModelEvaluator.evaluate_model(
            model, X_test, y_test, model_name=name, normal_label=normal_label
        )
        V2XModelEvaluator.print_evaluation_report(eval_res)
        results_list.append(eval_res)
        
        if eval_res['f1_score'] > best_score:
            best_score = eval_res['f1_score']
            best_model = model
            best_name = name

    print(f"\n[V2X Training] Best Performing V2X Model: {best_name} (F1: {best_score:.4f})")
    
    artifact = {
        'model': best_model,
        'model_name': best_name,
        'label_encoder': le,
        'feature_names': list(X.columns),
        'evaluation_results': results_list
    }
    
    os.makedirs(os.path.dirname(config.V2X_MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, config.V2X_MODEL_PATH)
    print(f"[V2X Training] Saved V2X model artifact to: {config.V2X_MODEL_PATH}")
    
    return artifact

if __name__ == '__main__':
    train_v2x_models()
