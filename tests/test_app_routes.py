import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as flask_app


def test_root_and_legacy_login_redirect_to_dashboard():
    client = flask_app.app.test_client()

    assert client.get('/').headers['Location'].endswith('/dashboard')
    assert client.get('/login').headers['Location'].endswith('/dashboard')


def test_dashboard_is_available_without_login():
    client = flask_app.app.test_client()

    response = client.get('/dashboard')

    assert response.status_code == 200


def test_api_is_available_without_login():
    client = flask_app.app.test_client()

    response = client.get('/api/statistics')

    assert response.status_code == 200
    assert 'summary' in response.json


def test_veremi_batch_uses_veremi_persistence_adapter(monkeypatch):
    saved = []

    def predict_records(records):
        return [{
            'timestamp': 1.0,
            'receiver_id': 'receiver-1',
            'actual_label': 'DoSRandom',
            'attack_type': 'DoSRandom',
            'outcome': 'TRUE_POSITIVE',
            'confidence': 0.9,
            'severity': 'HIGH',
            'pos_0': 10.0,
            'pos_1': 20.0,
            'speed_0': 1.0,
            'speed_1': 2.0,
            'acceleration_0': 0.1,
            'acceleration_1': 0.2,
        }]

    monkeypatch.setattr(
        flask_app,
        'veremi_predictor',
        SimpleNamespace(
            label_encoder=SimpleNamespace(classes_=['Benign', 'DoSRandom']),
            predict_records=predict_records,
        ),
    )
    monkeypatch.setattr(flask_app.pd, 'read_csv', lambda path: flask_app.pd.DataFrame({'AttackerType': ['DoSRandom']}))
    monkeypatch.setattr(flask_app.EventDAO, 'save_veremi_events', lambda events: saved.extend(events))

    response = flask_app.app.test_client().post('/api/dataset-batch', json={
        'dataset': 'v2x_veremi', 'label': 'DoSRandom', 'save_to_db': True,
    })

    assert response.status_code == 200
    assert response.json['saved_to_db'] is True
    assert len(saved) == 1


def test_can_batch_accepts_legacy_ground_truth_column(monkeypatch):
    def predict(records):
        assert 'label' in records.columns
        return [{
            'timestamp': 1.0, 'can_id': '0x1', 'dlc': 8, 'data': '00',
            'actual_label': 'DOS', 'attack_type': 'DOS',
            'outcome': 'TRUE_POSITIVE', 'confidence': 0.9, 'severity': 'CRITICAL',
        }]

    monkeypatch.setattr(
        flask_app,
        'can_predictor',
        SimpleNamespace(
            label_encoder=SimpleNamespace(classes_=['DOS']),
            predict_single_record=predict,
        ),
    )
    monkeypatch.setattr(flask_app.pd, 'read_csv', lambda path: flask_app.pd.DataFrame({
        'timestamp': [1.0], 'can_id': ['0x1'], 'dlc': [8], 'data': ['00'],
        '\ufeffAttackType': ['DOS'],
    }))

    response = flask_app.app.test_client().post('/api/dataset-batch', json={
        'dataset': 'can_hcrl', 'label': 'DOS',
    })

    assert response.status_code == 200
    assert response.json['processed_count'] == 1


def test_health_reports_loaded_models(monkeypatch):
    model = SimpleNamespace(
        model_name='Test Model',
        label_encoder=SimpleNamespace(classes_=['NORMAL', 'DOS'])
    )
    monkeypatch.setattr(flask_app, 'can_predictor', model)
    monkeypatch.setattr(flask_app, 'v2x_predictor', None)

    response = flask_app.app.test_client().get('/api/health')

    assert response.status_code == 200
    assert response.json['can'] == {
        'ready': True,
        'model': 'Test Model',
        'trained_classes': ['NORMAL', 'DOS']
    }
    assert response.json['legacy_synthetic_v2x']['ready'] is False


def test_readiness_endpoint_reports_degraded_service(monkeypatch):
    monkeypatch.setattr(flask_app, 'v2x_predictor', None)

    response = flask_app.app.test_client().get('/api/ready')

    assert response.status_code == 503
    assert response.json['status'] == 'degraded'


def test_simulation_rejects_invalid_domain_and_mode():
    client = flask_app.app.test_client()

    invalid_domain = client.post('/api/simulate-batch', json={'domain': 'other'})
    invalid_mode = client.post('/api/simulate-batch', json={'domain': 'can', 'mode': 'UNKNOWN'})

    assert invalid_domain.status_code == 400
    assert invalid_mode.status_code == 400


def test_simulation_reports_unavailable_model(monkeypatch):
    monkeypatch.setattr(flask_app, 'can_predictor', None)

    response = flask_app.app.test_client().post(
        '/api/simulate-batch', json={'domain': 'can', 'mode': 'NORMAL'}
    )

    assert response.status_code == 503
    assert response.json['error'] == 'CAN prediction model is unavailable.'


def test_scenario_pages_have_individual_buttons_without_live_controls():
    client = flask_app.app.test_client()
    can_page = client.get('/can').get_data(as_text=True)
    v2x_page = client.get('/v2x').get_data(as_text=True)

    assert 'HCRL CAR-HACKING CLASSES' in can_page
    assert 'loadCanScenarios()' in can_page
    assert 'CAN Injection' not in can_page
    assert 'CAN Spoofing' not in can_page
    assert 'CAN Replay' not in can_page
    assert 'RAW VEREMI ATTACKER TYPES (20 CLASSES)' in v2x_page
    assert 'loadVeReMiScenarios()' in v2x_page
    assert 'False Safety Message' not in v2x_page
    for page in (can_page, v2x_page):
        assert 'Start Live Stream' not in page
        assert 'Stop Live Stream' not in page
        assert '<select' not in page


def test_dataset_class_api_returns_trained_classes(monkeypatch):
    model = SimpleNamespace(
        model_name='VeReMi Test Model',
        label_encoder=SimpleNamespace(classes_=['Benign', 'DoS']),
        class_distribution={'Benign': 100, 'DoS': 100},
    )
    monkeypatch.setattr(flask_app, 'veremi_predictor', model)

    response = flask_app.app.test_client().get('/api/dataset-classes?dataset=v2x_veremi')

    assert response.status_code == 200
    assert response.json['classes'] == ['Benign', 'DoS']
    assert response.json['normal_label'] == 'Benign'


def test_dataset_batch_filters_rows_using_holdout_truth(monkeypatch, tmp_path):
    holdout_path = tmp_path / 'can_holdout.csv'
    holdout_path.write_text('label\nNORMAL\nDOS\n', encoding='utf-8')
    monkeypatch.setattr(flask_app.config, 'CAN_HCRL_HOLDOUT_PATH', str(holdout_path))

    def predict(records):
        return [
            {
                'actual_label': row['label'],
                'attack_type': row['label'],
                'outcome': 'TRUE_NEGATIVE' if row['label'] == 'NORMAL' else 'TRUE_POSITIVE',
            }
            for _, row in records.iterrows()
        ]

    monkeypatch.setattr(
        flask_app,
        'can_predictor',
        SimpleNamespace(
            label_encoder=SimpleNamespace(classes_=['DOS', 'NORMAL']),
            predict_single_record=predict,
        ),
    )
    client = flask_app.app.test_client()

    normal = client.post('/api/dataset-batch', json={
        'dataset': 'can_hcrl', 'label': 'NORMAL', 'limit': 10
    })
    attack = client.post('/api/dataset-batch', json={
        'dataset': 'can_hcrl', 'label': 'DOS', 'limit': 10
    })

    assert normal.status_code == 200
    assert normal.json['true_negative_count'] == 1
    assert [record['outcome'] for record in normal.json['records']] == ['TRUE_NEGATIVE']
    assert attack.status_code == 200
    assert attack.json['true_positive_count'] == 1
    assert [record['actual_label'] for record in attack.json['records']] == ['DOS']
