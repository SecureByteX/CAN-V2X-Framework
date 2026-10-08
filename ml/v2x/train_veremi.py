import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

import config
from evaluation.can_evaluation import ModelEvaluator
from ml.v2x.veremi import (
    FEATURE_COLUMNS,
    RECEIVER_COLUMN,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    chronological_split,
    get_features_and_labels,
    prepare_veremi_frame,
)


def train_veremi_models():
    if not os.path.exists(config.V2X_VEREMI_RAW_DATA_PATH):
        raise FileNotFoundError(config.V2X_VEREMI_RAW_DATA_PATH)

    dtype = {column: 'float32' for column in FEATURE_COLUMNS}
    dtype.update({TIME_COLUMN: 'float64', RECEIVER_COLUMN: 'category', TARGET_COLUMN: 'category'})
    raw = pd.read_csv(
        config.V2X_VEREMI_RAW_DATA_PATH,
        usecols=sorted(REQUIRED_COLUMNS),
        dtype=dtype,
        low_memory=False,
    )
    frame = prepare_veremi_frame(raw)
    if frame.empty:
        raise ValueError('The VeReMi dataset has no valid labeled rows.')

    train, test = chronological_split(frame, test_fraction=0.2)
    X_train, y_train = get_features_and_labels(train)
    X_test, y_test = get_features_and_labels(test)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    normal_label = int(label_encoder.transform(['Benign'])[0])

    models = {
        'Logistic Regression (SGD)': make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss='log_loss',
                max_iter=30,
                tol=1e-3,
                early_stopping=True,
                n_jobs=4,
                random_state=42,
            ),
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=4,
            random_state=42,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=60,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            n_jobs=4,
            random_state=42,
            eval_metric='mlogloss',
        ),
    }

    results = []
    best_model = None
    best_name = None
    best_macro_f1 = -1.0
    for name, model in models.items():
        print(f'[VeReMi Training] Fitting {name} on {len(train):,} rows...')
        model.fit(X_train, y_train_encoded)
        result = ModelEvaluator.evaluate_model(
            model,
            X_test,
            y_test_encoded,
            model_name=name,
            normal_label=normal_label,
        )
        result['test_rows'] = len(test)
        result['macro_f1_score'] = result['f1_macro']
        ModelEvaluator.print_evaluation_report(result)
        results.append(result)
        if result['f1_macro'] > best_macro_f1:
            best_model = model
            best_name = name
            best_macro_f1 = result['f1_macro']

    class_distribution = {
        str(label): int(count)
        for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
    }
    sample_columns = [TIME_COLUMN, RECEIVER_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    samples = (
        test.groupby(TARGET_COLUMN, group_keys=False)
        .sample(n=min(50, test[TARGET_COLUMN].value_counts().min()), random_state=42)
        [sample_columns]
        .sort_values([TARGET_COLUMN, TIME_COLUMN])
    )
    os.makedirs(os.path.dirname(config.V2X_VEREMI_SAMPLE_PATH), exist_ok=True)
    samples.to_csv(config.V2X_VEREMI_SAMPLE_PATH, index=False)

    artifact = {
        'model': best_model,
        'model_name': best_name,
        'label_encoder': label_encoder,
        'feature_names': FEATURE_COLUMNS,
        'classes': list(label_encoder.classes_),
        'class_distribution': class_distribution,
        'source_path': config.V2X_VEREMI_RAW_DATA_PATH,
        'source_rows': len(frame),
        'train_rows': len(train),
        'test_rows': len(test),
        'split_strategy': 'Last 20% by rcvTime within each AttackerType class',
        'results': results,
    }
    os.makedirs(os.path.dirname(config.V2X_VEREMI_MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, config.V2X_VEREMI_MODEL_PATH)
    print(f'[VeReMi Training] Selected {best_name} by macro F1 ({best_macro_f1:.4f}).')
    print(f'[VeReMi Training] Saved artifact to {config.V2X_VEREMI_MODEL_PATH}')
    print(f'[VeReMi Training] Saved held-out class samples to {config.V2X_VEREMI_SAMPLE_PATH}')
    return artifact


if __name__ == '__main__':
    train_veremi_models()