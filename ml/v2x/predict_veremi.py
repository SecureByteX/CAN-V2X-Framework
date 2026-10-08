import joblib
import numpy as np
import os

import config
from ml.prediction_outcomes import classify_prediction
from ml.v2x.veremi import FEATURE_COLUMNS, RECEIVER_COLUMN, TARGET_COLUMN, TIME_COLUMN


def veremi_severity(label):
    if label == 'Benign':
        return 'INFO'
    if label.startswith('DoS') or label == 'Disruptive':
        return 'CRITICAL'
    if 'Sybil' in label:
        return 'HIGH'
    if label in {'DataReplay', 'DelayedMessages'}:
        return 'MEDIUM'
    return 'HIGH'


class VeReMiPredictor:
    def __init__(self, model_path=config.V2X_VEREMI_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'VeReMi model file not found at {model_path}. Train it first.')
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.model_name = artifact['model_name']
        self.label_encoder = artifact['label_encoder']
        self.feature_names = artifact['feature_names']
        self.classes = list(artifact['classes'])
        self.class_distribution = artifact['class_distribution']
        self.source_rows = artifact['source_rows']
        self.train_rows = artifact['train_rows']
        self.test_rows = artifact['test_rows']
        self.evaluation_results = artifact['results']
        self.split_strategy = artifact['split_strategy']

    def predict_records(self, records):
        features = records[self.feature_names]
        predicted_indices = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)

        results = []
        for index, predicted_label in enumerate(predicted_labels):
            row = records.iloc[index]
            actual_label = str(row[TARGET_COLUMN])
            predicted_label = str(predicted_label)
            results.append({
                'timestamp': float(row[TIME_COLUMN]),
                'receiver_id': str(row[RECEIVER_COLUMN]),
                'actual_label': actual_label,
                'attack_type': predicted_label,
                'outcome': classify_prediction(actual_label, predicted_label, normal_label='Benign'),
                'confidence': round(float(np.max(probabilities[index])), 4),
                'severity': veremi_severity(predicted_label),
                'pos_0': float(row['pos_0']),
                'pos_1': float(row['pos_1']),
                'speed_0': float(row['spd_0']),
                'speed_1': float(row['spd_1']),
                'acceleration_0': float(row['acl_0']),
                'acceleration_1': float(row['acl_1']),
            })
        return results