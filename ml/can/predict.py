import json
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from ml.feature_engineering import FeatureExtractor
from ml.prediction_outcomes import classify_prediction
from ml.onnx_model import LabelDecoder, OnnxClassifier

class CANPredictor:
    """
    Real-time CAN IDS Predictor loaded from saved model artifact.
    """
    def __init__(self, model_path=config.CAN_ONNX_MODEL_PATH, metadata_path=config.CAN_MODEL_METADATA_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CAN ONNX model file not found at {model_path}. Export the trained model first.")
        
        with open(metadata_path, encoding='utf-8') as metadata_file:
            artifact = json.load(metadata_file)
        self.model = OnnxClassifier(model_path)
        self.model_name = artifact['model_name']
        self.label_encoder = LabelDecoder(artifact['classes'])
        self.feature_names = artifact['feature_names']
        self.dataset_name = artifact.get('dataset_name', 'CAN')
        self.dataset_rows = artifact.get('dataset_rows')
        self.train_rows = artifact.get('train_rows')
        self.test_rows = artifact.get('test_rows')
        self.class_distribution = artifact.get('class_distribution', {})
        self.evaluation_results = artifact.get('evaluation_results', [])

    def predict_single_record(self, raw_record_df):
        """
        Takes a raw single record or dataframe batch, extracts features,
        and returns prediction label, confidence score, and severity level.
        """
        if all(column in raw_record_df.columns for column in self.feature_names):
            X_feat = raw_record_df[self.feature_names].copy()
        else:
            X_feat, _, _ = FeatureExtractor.extract_can_features(raw_record_df)
        
        # Ensure correct column ordering
        X_feat = X_feat[self.feature_names]
        
        # Predict class index and probabilities
        pred_idx = self.model.predict(X_feat)
        pred_probs = self.model.predict_proba(X_feat)
        
        results = []
        for i in range(len(raw_record_df)):
            idx = pred_idx[i]
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(np.max(pred_probs[i]))
            actual_label = raw_record_df.iloc[i].get('label')
            outcome = classify_prediction(actual_label, label)
            
            # Map severity
            if label == 'NORMAL':
                severity = 'INFO'
            elif label in ['INJECTION', 'REPLAY', 'FUZZY']:
                severity = 'MEDIUM'
            elif label in ['SPOOFING', 'RPM', 'GEAR']:
                severity = 'HIGH'
            elif label in ['DOS']:
                severity = 'CRITICAL'
            else:
                severity = 'HIGH'
                
            results.append({
                'source': 'CAN',
                'timestamp': str(raw_record_df.iloc[i]['timestamp']),
                'can_id': str(raw_record_df.iloc[i]['can_id']),
                'dlc': int(raw_record_df.iloc[i]['dlc']),
                'data': str(raw_record_df.iloc[i]['data']),
                'attack_type': label,
                'actual_label': actual_label,
                'outcome': outcome,
                'confidence': round(confidence, 4),
                'severity': severity,
                'status': 'Detected' if label != 'NORMAL' else 'Normal',
                'model_name': self.model_name,
                'features': {
                    str(name): float(value)
                    for name, value in X_feat.iloc[i].items()
                }
            })
            
        return results

if __name__ == '__main__':
    predictor = CANPredictor()
    sample_df = pd.DataFrame([{
        'timestamp': 10.123456,
        'can_id': '0x0C0',
        'dlc': 8,
        'data': '270FFF0000000000'
    }])
    res = predictor.predict_single_record(sample_df)
    print("CAN Prediction Sample Output:", res)
