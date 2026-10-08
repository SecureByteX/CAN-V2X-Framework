import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from simulation.v2x_generator import V2XDataGenerator
from simulation.v2x_attacks import V2XAttackSimulator
from ml.feature_engineering import FeatureExtractor
from ml.v2x.predict import V2XPredictor
from ml.v2x.train import REQUIRED_V2X_COLUMNS, select_training_dataset
from ml.v2x.veremi import (
    FEATURE_COLUMNS as VEREMI_FEATURE_COLUMNS,
    chronological_split,
    get_features_and_labels,
    prepare_veremi_frame,
)


def test_v2x_training_uses_raw_dataset_when_schema_matches(tmp_path):
    raw_path = tmp_path / 'raw.csv'
    processed_path = tmp_path / 'processed.csv'
    raw_path.write_text(','.join(sorted(REQUIRED_V2X_COLUMNS)) + '\n', encoding='utf-8')
    processed_path.write_text('AttackerType\nBenign\n', encoding='utf-8')

    assert select_training_dataset(str(raw_path), str(processed_path)) == str(raw_path)


def test_v2x_training_falls_back_to_compatible_processed_dataset(tmp_path):
    raw_path = tmp_path / 'raw.csv'
    processed_path = tmp_path / 'processed.csv'
    raw_path.write_text('AttackerType\nDoS\n', encoding='utf-8')
    processed_path.write_text(','.join(sorted(REQUIRED_V2X_COLUMNS)) + '\n', encoding='utf-8')

    assert select_training_dataset(str(raw_path), str(processed_path)) == str(processed_path)


def test_v2x_training_rejects_incompatible_datasets(tmp_path):
    raw_path = tmp_path / 'raw.csv'
    processed_path = tmp_path / 'processed.csv'
    raw_path.write_text('AttackerType\nDoS\n', encoding='utf-8')
    processed_path.write_text('label,latitude,longitude\nNORMAL,12.0,77.0\n', encoding='utf-8')

    with pytest.raises(ValueError, match='No V2X training CSV'):
        select_training_dataset(str(raw_path), str(processed_path))


def test_veremi_holdout_uses_later_times_for_each_class():
    frame = pd.DataFrame({
        'rcvTime': [1, 2, 3, 4, 5, 6, 7, 8],
        'AttackerType': ['Benign'] * 4 + ['DoS'] * 4,
        'ReceiverID': ['R1', 'R2'] * 4,
        **{column: np.arange(8, dtype=float) for column in VEREMI_FEATURE_COLUMNS},
    })

    prepared = prepare_veremi_frame(frame)
    train, test = chronological_split(prepared, test_fraction=0.25)
    features, labels = get_features_and_labels(prepared)

    assert len(train) == 6
    assert len(test) == 2
    assert set(train['AttackerType']) == set(test['AttackerType']) == {'Benign', 'DoS'}
    assert train.groupby('AttackerType')['rcvTime'].max().lt(
        test.groupby('AttackerType')['rcvTime'].min()
    ).all()
    assert 'rcvTime' not in features.columns
    assert 'ReceiverID' not in features.columns
    assert labels.tolist() == frame['AttackerType'].tolist()

def test_v2x_generator():
    gen = V2XDataGenerator()
    df = gen.generate_normal_v2x_stream(duration_sec=1.0)
    assert not df.empty
    assert 'sender_id' in df.columns
    assert 'speed' in df.columns

def test_v2x_attacks():
    att = V2XAttackSimulator()
    df_base = att.generator.generate_normal_v2x_stream(duration_sec=1.0)
    df_spoofed = att.inject_spoofing_attack(df_base, num_attacks=5)
    assert 'SPOOFING' in df_spoofed['label'].values

def test_v2x_feature_extractor():
    gen = V2XDataGenerator()
    df = gen.generate_normal_v2x_stream(duration_sec=1.0)
    X, y, _ = FeatureExtractor.extract_v2x_features(df)
    assert not X.empty
    assert 'sequence_consistency' in X.columns
    assert 'speed_consistency' in X.columns

def test_v2x_predictor():
    if os.path.exists(config.V2X_MODEL_PATH):
        predictor = V2XPredictor()
        raw_sample = pd.DataFrame([{
            'timestamp': 1.0,
            'message_type': 'BSM',
            'sender_id': 'VEHICLE_001',
            'receiver_id': 'RSU_001',
            'latitude': 12.9716,
            'longitude': 77.5946,
            'speed': 45.0,
            'heading': 90.0,
            'acceleration': 0.5,
            'sequence_number': 101
        }])
        res = predictor.predict_single_record(raw_sample)
        assert len(res) == 1
        assert res[0]['source'] == 'V2X'
        assert 'attack_type' in res[0]
