import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from simulation.can_generator import CANDataGenerator
from simulation.can_attacks import CANAttackSimulator
from ml.feature_engineering import FeatureExtractor
from ml.preprocessing import DataPreprocessor
from ml.can.predict import CANPredictor
from evaluation.can_evaluation import ModelEvaluator

def test_can_generator():
    gen = CANDataGenerator()
    df = gen.generate_normal_stream(duration_sec=1.0)
    assert not df.empty
    assert 'can_id' in df.columns
    assert 'timestamp' in df.columns
    assert 'data' in df.columns

def test_can_attacks():
    att = CANAttackSimulator()
    df_base = att.generator.generate_normal_stream(duration_sec=1.0)
    df_injected = att.inject_injection_attack(df_base, num_attacks=5)
    assert 'INJECTION' in df_injected['label'].values

def test_can_feature_extractor():
    gen = CANDataGenerator()
    df = gen.generate_normal_stream(duration_sec=1.0)
    X, y, _ = FeatureExtractor.extract_can_features(df)
    assert not X.empty
    assert 'inter_arrival_time' in X.columns
    assert 'burst_rate' in X.columns

def test_can_burst_rate_uses_trailing_50ms_window():
    df = pd.DataFrame([
        {'timestamp': 0.00, 'can_id': '0x0C0', 'dlc': 1, 'data': '01'},
        {'timestamp': 0.02, 'can_id': '0x0C0', 'dlc': 1, 'data': '02'},
        {'timestamp': 0.08, 'can_id': '0x0C0', 'dlc': 1, 'data': '03'},
    ])

    features, _, _ = FeatureExtractor.extract_can_features(df)

    assert features['burst_rate'].tolist() == [1, 2, 1]


def test_can_chronological_split_holds_out_later_times_per_label():
    frame = pd.DataFrame({
        'timestamp': [4, 1, 8, 2, 7, 3, 6, 5],
        'label': ['DOS', 'DOS', 'DOS', 'DOS', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL'],
    })

    train, test = DataPreprocessor().chronological_split(frame, 'timestamp')

    assert train.groupby('label')['timestamp'].max().lt(
        test.groupby('label')['timestamp'].min()
    ).all()

def test_false_positive_rate_uses_encoded_normal_label():
    class FixedModel:
        def predict(self, features):
            return np.array([0, 1, 0, 0])

    results = ModelEvaluator.evaluate_model(
        FixedModel(), pd.DataFrame(index=range(4)), np.array([1, 1, 0, 0]),
        normal_label=1
    )

    assert results['false_positive_rate'] == 0.5
    assert results['inference_time_ms'] > 0
    assert results['inference_sample_count'] == 4

def test_can_predictor():
    if os.path.exists(config.CAN_MODEL_PATH):
        predictor = CANPredictor()
        raw_sample = pd.DataFrame([{
            'timestamp': 1.0,
            'can_id': '0x0C0',
            'dlc': 8,
            'data': '0102030405060708'
        }])
        res = predictor.predict_single_record(raw_sample)
        assert len(res) == 1
        assert res[0]['source'] == 'CAN'
        assert 'attack_type' in res[0]
