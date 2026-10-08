import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from ml.feature_engineering import FeatureExtractor
from ml.prediction_outcomes import classify_prediction

class V2XPredictor:
    """
    Real-time V2X IDS Predictor loaded from saved model artifact.
    """
    def __init__(self, model_path=config.V2X_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"V2X Model file not found at {model_path}. Train model first.")
        
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.model_name = artifact['model_name']
        self.label_encoder = artifact['label_encoder']
        self.feature_names = artifact['feature_names']

    def predict_single_record(self, raw_record_df):
        X_feat, _, _ = FeatureExtractor.extract_v2x_features(raw_record_df)
        X_feat = X_feat[self.feature_names]
        
        pred_idx = self.model.predict(X_feat)
        pred_probs = self.model.predict_proba(X_feat)
        
        results = []
        for i in range(len(raw_record_df)):
            idx = pred_idx[i]
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(np.max(pred_probs[i]))
            actual_label = raw_record_df.iloc[i].get('label')
            outcome = classify_prediction(actual_label, label)
            
            if label == 'NORMAL':
                severity = 'INFO'
            elif label in ['REPLAY']:
                severity = 'MEDIUM'
            elif label in ['SPOOFING', 'FALSE_SAFETY']:
                severity = 'HIGH'
            elif label in ['DOS']:
                severity = 'CRITICAL'
            else:
                severity = 'HIGH'
                
            results.append({
                'source': 'V2X',
                'timestamp': str(raw_record_df.iloc[i]['timestamp']),
                'message_type': str(raw_record_df.iloc[i]['message_type']),
                'sender_id': str(raw_record_df.iloc[i]['sender_id']),
                'receiver_id': str(raw_record_df.iloc[i]['receiver_id']),
                'latitude': float(raw_record_df.iloc[i]['latitude']),
                'longitude': float(raw_record_df.iloc[i]['longitude']),
                'speed': float(raw_record_df.iloc[i]['speed']),
                'heading': float(raw_record_df.iloc[i]['heading']),
                'acceleration': float(raw_record_df.iloc[i]['acceleration']),
                'sequence_number': int(raw_record_df.iloc[i]['sequence_number']),
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
    predictor = V2XPredictor()
    sample_df = pd.DataFrame([{
        'timestamp': 10.123456,
        'message_type': 'BSM',
        'sender_id': 'VEHICLE_001',
        'receiver_id': 'RSU_001',
        'latitude': 12.9716,
        'longitude': 77.5946,
        'speed': 45.2,
        'heading': 90.0,
        'acceleration': 1.2,
        'sequence_number': 105
    }])
    res = predictor.predict_single_record(sample_df)
    print("V2X Prediction Sample Output:", res)
