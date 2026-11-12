import json
import os
import sys
import types
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app as flask_app
import config
from alert_normalizer import AlertValidationError, normalize_alert
from database.db import get_db_connection, init_db
from database.models import EventDAO
from explainability import explain_prediction
from llm_investigator import InvestigationUnavailable, _validate_result, investigate
from mitre_mapper import map_attack
from risk_engine import score_alert, score_incident
from threat_correlation import correlate


@pytest.fixture
def incident_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'incidents.sqlite'))
    monkeypatch.setattr(config, 'DATABASE_URL', '')
    init_db()
    return flask_app.app.test_client()


def can_event(**extra):
    return {
        'timestamp': '1710000000.0', 'can_id': '0x123', 'dlc': 8,
        'data': '0011223344556677', 'attack_type': 'DOS',
        'actual_label': 'DOS', 'outcome': 'TRUE_POSITIVE',
        'confidence': 0.91, 'severity': 'CRITICAL',
        'model_name': 'CAN test model', 'features': {'burst_rate': 3.0},
        **extra,
    }


def v2x_event(**extra):
    return {
        'timestamp': '1710000001.0', 'message_type': 'BSM',
        'sender_id': 'sender-1', 'receiver_id': 'receiver-1',
        'latitude': 12.9, 'longitude': 77.5, 'speed': 21.0,
        'heading': 90.0, 'acceleration': 0.2, 'sequence_number': 10,
        'attack_type': 'DOS', 'actual_label': 'DOS', 'outcome': 'TRUE_POSITIVE',
        'confidence': 0.88, 'severity': 'CRITICAL', 'model_name': 'V2X test model',
        'features': {'speed_delta': 3.0}, **extra,
    }


def test_normalizes_can_and_veremi_without_inventing_ids():
    can = normalize_alert(can_event(), 'CAN')
    veremi = normalize_alert({
        'timestamp': 1710000000.0, 'receiver_id': 'receiver-9',
        'attack_type': 'DoSRandom', 'confidence': 0.8, 'severity': 'CRITICAL',
        'outcome': 'TRUE_POSITIVE', 'features': {'pos_0': 4.5}, 'model_name': 'VeReMi model',
    }, 'V2X')

    assert can['source'] == 'CAN'
    assert can['can_id'] == '0x123'
    assert can['sender_id'] is None
    assert veremi['source'] == 'V2X'
    assert veremi['receiver_id'] == 'receiver-9'
    assert veremi['sender_id'] is None
    assert veremi['vehicle_id'] is None
    assert veremi['relevant_features'] == {'pos_0': 4.5}
    with pytest.raises(AlertValidationError):
        normalize_alert({'timestamp': 'bad', 'attack_type': 'DOS'}, 'CAN')


def test_correlation_requires_time_category_and_reliable_identity():
    base = {'alert_id': 'a', 'source': 'CAN', 'timestamp': '2024-03-09T16:00:00+00:00', 'attack_type': 'DOS', 'can_id': '0x123'}
    same = {**base, 'alert_id': 'b', 'timestamp': '2024-03-09T16:00:20+00:00'}
    other_id = {**same, 'can_id': '0x456'}
    cross = {**same, 'source': 'V2X', 'sender_id': 'sender-1', 'can_id': None}
    explicit_cross = {**cross, 'vehicle_id': 'vehicle-1'}
    explicit_can = {**base, 'vehicle_id': 'vehicle-1'}

    assert correlate(base, same)['strength'] > 0
    assert correlate(base, other_id) is None
    assert correlate(base, cross) is None
    assert correlate(explicit_can, explicit_cross)['strength'] == 1.0
    assert correlate(base, {**same, 'timestamp': '2024-03-09T16:02:00+00:00'}) is None


def test_risk_formula_distinguishes_false_positive_and_correlated_evidence():
    confirmed = score_alert({'severity': 'HIGH', 'confidence': 0.4, 'prediction_outcome': 'TRUE_POSITIVE'})
    false_positive = score_alert({'severity': 'HIGH', 'confidence': 0.99, 'prediction_outcome': 'FALSE_POSITIVE'})
    assert confirmed['risk_score'] > false_positive['risk_score']
    assert confirmed['factors']['confidence_weight'] == 0.3
    combined = score_incident([confirmed, confirmed], 0.85)
    assert combined['risk_score'] > confirmed['risk_score']


def test_database_deduplicates_alert_and_creates_can_only_incident(incident_db):
    event = can_event()
    EventDAO.save_can_event(event)
    EventDAO.save_can_event(event)

    alerts = incident_db.get('/api/normalized-alerts?source=CAN')
    incidents = incident_db.get('/api/incidents')
    assert alerts.status_code == 200
    assert len(alerts.json['alerts']) == 1
    assert incidents.status_code == 200
    assert len(incidents.json['incidents']) == 1
    item = incidents.json['incidents'][0]
    detail = incident_db.get(f"/api/incidents/{item['incident_id']}")
    assert detail.status_code == 200
    assert detail.json['alerts'][0]['can_id'] == '0x123'
    assert {item['technique_id'] for item in detail.json['mitre']} == {'T1498', 'T0814'}


def test_cross_domain_incident_requires_explicit_shared_vehicle_mapping(incident_db):
    EventDAO.save_can_event(can_event())
    EventDAO.save_v2x_event(v2x_event())
    unlinked = incident_db.get('/api/incidents').json['incidents']
    assert len(unlinked) == 2

    EventDAO.save_can_event(can_event(timestamp='1710000100.0', can_id='0x777', vehicle_id='vehicle-42'))
    EventDAO.save_v2x_event(v2x_event(timestamp='1710000101.0', vehicle_id='vehicle-42'))
    all_incidents = incident_db.get('/api/incidents').json['incidents']
    cross_domain = [item for item in all_incidents if item['alert_count'] == 2]
    assert len(cross_domain) == 1
    detail = incident_db.get(f"/api/incidents/{cross_domain[0]['incident_id']}").json
    assert {alert['source'] for alert in detail['alerts']} == {'CAN', 'V2X'}
    assert 'explicitly supplied vehicle identifier' in detail['correlation_reason']


def test_status_validation_history_and_api_errors(incident_db):
    EventDAO.save_can_event(can_event())
    incident_id = incident_db.get('/api/incidents').json['incidents'][0]['incident_id']
    invalid = incident_db.post(f'/api/incidents/{incident_id}/status', json={'status': 'CLOSED'})
    changed = incident_db.post(f'/api/incidents/{incident_id}/status', json={
        'status': 'INVESTIGATING', 'changed_by': 'analyst', 'note': 'Reviewing evidence',
        'assigned_analyst': 'analyst',
    })
    assert invalid.status_code == 400
    assert changed.status_code == 200
    assert changed.json['status'] == 'INVESTIGATING'
    assert changed.json['history'][-1]['note'] == 'Reviewing evidence'
    assert incident_db.get('/api/incidents?status=INVALID').status_code == 400
    assert incident_db.get('/api/incidents/missing').status_code == 404


def test_mitre_mapping_is_conservative_and_explicit():
    mappings = map_attack('DoSRandom')
    assert {item['technique_id'] for item in mappings} == {'T1498', 'T0814'}
    assert all(item['requires_review'] for item in mappings)
    assert map_attack('RandomSpeed') == []


def test_shap_explanation_uses_active_predictor_output(monkeypatch, tmp_path):
    sample_path = tmp_path / 'background.csv'
    pd.DataFrame({'f1': [0.0, 1.0], 'f2': [1.0, 0.0]}).to_csv(sample_path, index=False)

    class FakeExplainer:
        expected_value = np.array([0.25])

        def __init__(self, fn, background):
            self.fn = fn

        def shap_values(self, row, nsamples):
            assert nsamples == 100
            return np.array([[0.1, -0.05]])

    monkeypatch.setitem(sys.modules, 'shap', types.SimpleNamespace(KernelExplainer=FakeExplainer))
    predictor = SimpleNamespace(
        feature_names=['f1', 'f2'],
        label_encoder=SimpleNamespace(classes_=['NORMAL', 'DOS']),
        model=SimpleNamespace(predict_proba=lambda frame: np.array([[0.2, 0.8]])),
    )
    result = explain_prediction({
        'alert_id': 'alert-1', 'predicted_class': 'DOS',
        'relevant_features': {'f1': 0.4, 'f2': 0.3},
    }, predictor, str(sample_path))
    assert result['output_value'] == 0.8
    assert result['top_features'][0]['feature'] == 'f1'
    assert result['top_features'][0]['contribution'] == 0.1


def test_llm_output_schema_rejects_malformed_data_and_threshold_missing_key(monkeypatch):
    valid = {
        'incident_summary': 'Summary', 'suspected_attack': 'Possible DoS',
        'evidence': ['Model labeled a DoS prediction'], 'possible_impact': 'Possible communication loss',
        'mitre_techniques': ['T1498 requires review'], 'recommended_investigation': ['Check packet volume'],
        'recommended_actions': ['Review gateway filtering'], 'confidence_and_limitations': 'Model output is not confirmation.',
    }
    assert _validate_result(valid) == valid
    with pytest.raises(InvestigationUnavailable):
        _validate_result({'incident_summary': 'only'})
    with pytest.raises(InvestigationUnavailable):
        investigate({'incident_id': 'x'})


def test_llm_investigation_success_and_provider_timeout(monkeypatch):
    valid = {
        'incident_summary': 'Summary', 'suspected_attack': 'Possible DoS',
        'evidence': ['Model labeled a DoS prediction'], 'possible_impact': 'Possible communication loss',
        'mitre_techniques': ['T1498 requires review'], 'recommended_investigation': ['Check packet volume'],
        'recommended_actions': ['Review gateway filtering'], 'confidence_and_limitations': 'Not confirmed.',
    }
    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_MODEL', 'test-model')

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({'choices': [{'message': {'content': json.dumps(valid)}}]}).encode()

    monkeypatch.setattr('urllib.request.urlopen', lambda *_args, **_kwargs: FakeResponse())
    result, model, _url = investigate({'incident_id': 'incident-1', 'alerts': []})
    assert result == valid
    assert model == 'test-model'

    def timeout(*_args, **_kwargs):
        raise TimeoutError('simulated timeout')

    monkeypatch.setattr('urllib.request.urlopen', timeout)
    with pytest.raises(InvestigationUnavailable, match='could not be reached'):
        investigate({'incident_id': 'incident-1', 'alerts': []})


def test_incident_investigation_endpoint_keeps_configuration_unavailable_clear(incident_db, monkeypatch):
    EventDAO.save_can_event(can_event())
    incident_id = incident_db.get('/api/incidents').json['incidents'][0]['incident_id']
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    response = incident_db.post(f'/api/incidents/{incident_id}/investigate', json={})
    assert response.status_code == 503
    assert response.json['available'] is False

