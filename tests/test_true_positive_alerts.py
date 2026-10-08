import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as flask_app
import config
from database.db import get_db_connection, init_db
from database.models import EventDAO
from ml.prediction_outcomes import classify_prediction


@pytest.mark.parametrize(
    ('actual_label', 'predicted_label', 'expected'),
    [
        ('INJECTION', 'INJECTION', 'TRUE_POSITIVE'),
        ('NORMAL', 'NORMAL', 'TRUE_NEGATIVE'),
        ('NORMAL', 'DOS', 'FALSE_POSITIVE'),
        ('DOS', 'NORMAL', 'FALSE_NEGATIVE'),
        ('DOS', 'REPLAY', 'MISCLASSIFIED_ATTACK'),
        (None, 'DOS', 'UNVERIFIED'),
    ],
)
def test_prediction_outcome_classification(actual_label, predicted_label, expected):
    assert classify_prediction(actual_label, predicted_label) == expected


def test_veremi_benign_class_is_treated_as_normal():
    assert classify_prediction('Benign', 'Benign', normal_label='Benign') == 'TRUE_NEGATIVE'
    assert classify_prediction('Benign', 'DoS', normal_label='Benign') == 'FALSE_POSITIVE'
    assert classify_prediction('DoS', 'Benign', normal_label='Benign') == 'FALSE_NEGATIVE'


@pytest.mark.parametrize(
    ('domain', 'mode', 'generator_name', 'generator_method', 'attack_name', 'attack_method', 'predictor_name', 'save_method'),
    [
        ('can', 'INJECTION', 'can_gen', 'generate_normal_stream', 'can_att', 'inject_injection_attack', 'can_predictor', 'save_can_event'),
        ('v2x', 'SPOOFING', 'v2x_gen', 'generate_normal_v2x_stream', 'v2x_att', 'inject_spoofing_attack', 'v2x_predictor', 'save_v2x_event'),
    ],
)
def test_simulate_batch_returns_only_true_positives(
    monkeypatch, domain, mode, generator_name, generator_method,
    attack_name, attack_method, predictor_name, save_method
):
    raw_records = pd.DataFrame({'label': ['NORMAL', mode]})
    predictions = [
        {'actual_label': 'NORMAL', 'attack_type': mode, 'outcome': 'FALSE_POSITIVE'},
        {'actual_label': mode, 'attack_type': mode, 'outcome': 'TRUE_POSITIVE'},
        {'actual_label': 'NORMAL', 'attack_type': 'NORMAL', 'outcome': 'TRUE_NEGATIVE'},
        {'actual_label': mode, 'attack_type': 'NORMAL', 'outcome': 'FALSE_NEGATIVE'},
        {'actual_label': mode, 'attack_type': 'OTHER', 'outcome': 'MISCLASSIFIED_ATTACK'},
    ]
    monkeypatch.setattr(
        flask_app, generator_name,
        SimpleNamespace(**{generator_method: lambda **kwargs: raw_records})
    )
    monkeypatch.setattr(
        flask_app, attack_name,
        SimpleNamespace(**{attack_method: lambda base, **kwargs: raw_records})
    )
    monkeypatch.setattr(
        flask_app, predictor_name,
        SimpleNamespace(
            label_encoder=SimpleNamespace(classes_=['NORMAL', mode]),
            predict_single_record=lambda records: predictions
        )
    )
    monkeypatch.setattr(EventDAO, save_method, lambda event: None)

    response = flask_app.app.test_client().post(
        '/api/simulate-batch', json={'domain': domain, 'mode': mode}
    )

    assert response.status_code == 200
    assert response.json['count'] == 5
    assert response.json['true_positive_count'] == 1
    assert response.json['true_negative_count'] == 1
    assert {record['outcome'] for record in response.json['records']} == {
        'TRUE_POSITIVE', 'TRUE_NEGATIVE'
    }


def test_only_true_positives_are_saved_as_can_alerts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'true-positives.db'))
    init_db()

    base_event = {
        'timestamp': '1.0',
        'can_id': '0x0C0',
        'dlc': 8,
        'data': '0000000000000000',
        'attack_type': 'INJECTION',
        'confidence': 0.99,
        'severity': 'MEDIUM',
    }
    EventDAO.save_can_event({
        **base_event, 'actual_label': 'NORMAL', 'outcome': 'FALSE_POSITIVE'
    })
    EventDAO.save_can_event({
        **base_event, 'actual_label': 'INJECTION', 'outcome': 'TRUE_POSITIVE'
    })

    connection = get_db_connection()
    events = connection.execute('SELECT COUNT(*) FROM can_events').fetchone()[0]
    alerts = connection.execute('SELECT COUNT(*) FROM can_alerts').fetchone()[0]
    alert = connection.execute('SELECT outcome FROM can_alerts').fetchone()
    connection.close()

    assert events == 2
    assert alerts == 1
    assert alert['outcome'] == 'TRUE_POSITIVE'


def test_only_true_positives_are_saved_as_v2x_alerts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'v2x-true-positives.db'))
    init_db()
    base_event = {
        'timestamp': '1.0',
        'message_type': 'BSM',
        'sender_id': 'VEHICLE_001',
        'receiver_id': 'RSU_001',
        'latitude': 12.97,
        'longitude': 77.59,
        'speed': 40.0,
        'heading': 90.0,
        'acceleration': 0.0,
        'sequence_number': 101,
        'attack_type': 'SPOOFING',
        'confidence': 0.95,
        'severity': 'HIGH',
    }
    EventDAO.save_v2x_event({
        **base_event, 'actual_label': 'NORMAL', 'outcome': 'FALSE_POSITIVE'
    })
    EventDAO.save_v2x_event({
        **base_event, 'actual_label': 'SPOOFING', 'outcome': 'TRUE_POSITIVE'
    })

    connection = get_db_connection()
    alerts = connection.execute('SELECT COUNT(*) FROM v2x_alerts').fetchone()[0]
    connection.close()

    assert alerts == 1


def test_statistics_count_true_negatives_and_only_true_positives(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'verified-stats.db'))
    init_db()
    base_event = {
        'timestamp': '1.0',
        'can_id': '0x0C0',
        'dlc': 8,
        'data': '0000000000000000',
        'attack_type': 'INJECTION',
        'confidence': 0.9,
        'severity': 'MEDIUM',
    }
    EventDAO.save_can_event({
        **base_event, 'actual_label': 'NORMAL', 'outcome': 'TRUE_NEGATIVE',
        'attack_type': 'NORMAL', 'severity': 'INFO'
    })
    EventDAO.save_can_event({
        **base_event, 'actual_label': 'NORMAL', 'outcome': 'FALSE_POSITIVE'
    })
    EventDAO.save_can_event({
        **base_event, 'actual_label': 'INJECTION', 'outcome': 'TRUE_POSITIVE'
    })
    EventDAO.save_can_event({**base_event, 'actual_label': None, 'outcome': 'UNVERIFIED'})

    response = flask_app.app.test_client().get('/api/statistics')

    assert response.status_code == 200
    assert response.json['can']['total'] == 3
    assert response.json['can']['normal'] == 1
    assert response.json['can']['attacks'] == 1
    assert response.json['can']['distribution'] == {'INJECTION': 1}