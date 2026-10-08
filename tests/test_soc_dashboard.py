import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as flask_app
import config
from database.db import get_db_connection, init_db
from database.models import EventDAO


def test_soc_api_returns_real_verified_alerts_and_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'soc.db'))
    init_db()

    EventDAO.save_can_event({
        'timestamp': '1710000000.0',
        'can_id': '0x123',
        'dlc': 8,
        'data': '0011223344556677',
        'attack_type': 'DOS',
        'actual_label': 'DOS',
        'outcome': 'TRUE_POSITIVE',
        'confidence': 0.91,
        'severity': 'CRITICAL',
        'model_name': 'CAN test model',
        'features': {'inter_arrival_ms': 2.0},
    })
    EventDAO.save_can_event({
        'timestamp': '1710000001.0',
        'can_id': '0x456',
        'dlc': 8,
        'data': '0011223344556677',
        'attack_type': 'DOS',
        'actual_label': 'NORMAL',
        'outcome': 'FALSE_POSITIVE',
        'confidence': 0.88,
        'severity': 'CRITICAL',
    })
    EventDAO.save_v2x_event({
        'timestamp': '1710000002.0',
        'message_type': 'BSM',
        'sender_id': 'VEHICLE-42',
        'receiver_id': 'RSU-1',
        'latitude': 12.97,
        'longitude': 77.59,
        'speed': 38.0,
        'heading': 90.0,
        'acceleration': 0.2,
        'sequence_number': 21,
        'attack_type': 'SPOOFING',
        'actual_label': 'SPOOFING',
        'outcome': 'TRUE_POSITIVE',
        'confidence': 0.84,
        'severity': 'HIGH',
        'model_name': 'V2X test model',
        'features': {'speed_consistency': 0.7},
    })

    client = flask_app.app.test_client()
    response = client.get('/api/soc?range=all')

    assert response.status_code == 200
    payload = response.json
    assert payload['summary']['can_threats'] == 1
    assert payload['summary']['v2x_threats'] == 1
    assert payload['summary']['critical_alerts'] == 2
    assert payload['summary']['distribution'] == {
        'CAN': {'DOS': 1},
        'V2X': {'SPOOFING': 1},
    }
    assert payload['alert_count'] == 2
    assert {alert['source'] for alert in payload['alerts']} == {'CAN', 'V2X'}
    can_alert = next(alert for alert in payload['alerts'] if alert['source'] == 'CAN')
    assert can_alert['model'] == 'CAN test model'
    assert can_alert['features'] == {'inter_arrival_ms': 2.0}
    assert can_alert['origin'] == 'Labeled simulator event'

    filtered = client.get('/api/soc?range=all&source=V2X&severity=HIGH&attack_type=SPOOFING&q=VEHICLE')
    assert filtered.status_code == 200
    assert filtered.json['alert_count'] == 1
    assert filtered.json['alerts'][0]['source'] == 'V2X'

    connection = get_db_connection()
    outcomes = connection.execute(
        'SELECT outcome FROM can_events ORDER BY id'
    ).fetchall()
    connection.close()
    assert [row['outcome'] for row in outcomes] == ['TRUE_POSITIVE', 'FALSE_POSITIVE']


def test_soc_api_rejects_invalid_filters():
    client = flask_app.app.test_client()

    assert client.get('/api/soc?range=forever').status_code == 400
    assert client.get('/api/soc?source=OTHER').status_code == 400
    assert client.get('/api/soc?severity=EXTREME').status_code == 400


def test_soc_api_returns_empty_alerts_when_database_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'empty.db'))
    init_db()

    response = flask_app.app.test_client().get('/api/soc?range=all')

    assert response.status_code == 200
    assert response.json['alerts'] == []
    assert response.json['alert_count'] == 0
    assert response.json['summary']['can_threats'] == 0
    assert response.json['summary']['v2x_threats'] == 0


def test_overview_uses_soc_page_and_relative_backend_endpoints():
    client = flask_app.app.test_client()
    page = client.get('/dashboard').get_data(as_text=True)
    script = client.get('/static/js/dashboard.js').get_data(as_text=True)

    assert 'Vehicular Security Operations Center' in page
    assert 'No live vehicle feed is configured.' in page
    assert 'Recorded security alerts' in page
    assert 'alertDetails' in page
    assert 'Model performance' in page
    assert '/api/soc?' in script
    assert '/api/model-performance' in script
    assert 'http://localhost' not in script
    assert 'http://127.0.0.1' not in script
    assert 'Math.random' not in script
    assert 'No verified attack records in this period.' in script
