import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as flask_app
import config
from database.db import get_db_connection, init_db
from ml.can.parser import CANMessageParser
from ml.can.predict import CANPredictor
from ml.can.stream_processor import CANStreamProcessor
from simulation.can_attacks import CANAttackSimulator
from simulation.can_generator import CANDataGenerator


def test_can_parser_hcrl_text():
    line = 'Timestamp: 1478198376.389427        ID: 0316    000    DLC: 8    05 21 68 09 21 21 00 6f'
    parsed = CANMessageParser.parse_line(line)
    assert parsed is not None
    assert parsed['timestamp'] == 1478198376.389427
    assert parsed['can_id'] == '0x316'
    assert parsed['dlc'] == 8
    assert parsed['data'] == '052168092121006F'
    assert parsed['label'] == 'NORMAL'


def test_can_parser_hcrl_csv():
    line = '1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R'
    parsed = CANMessageParser.parse_line(line)
    assert parsed is not None
    assert parsed['timestamp'] == 1478198376.389427
    assert parsed['can_id'] == '0x316'
    assert parsed['dlc'] == 8
    assert parsed['data'] == '052168092121006F'
    assert parsed['label'] == 'NORMAL'

    attack_line = '1478198377.100000,0000,8,00,00,00,00,00,00,00,00,T'
    attack_parsed = CANMessageParser.parse_line(attack_line)
    assert attack_parsed['label'] == 'ATTACK'


def test_can_parser_candump_formats():
    compact = '(1478198376.500000) can0 120#1687FF0000000000'
    parsed1 = CANMessageParser.parse_line(compact)
    assert parsed1 is not None
    assert parsed1['timestamp'] == 1478198376.5
    assert parsed1['can_id'] == '0x120'
    assert parsed1['data'] == '1687FF0000000000'

    standard = 'can0  0C0   [8]  0A 66 01 20 00 00 00 00'
    parsed2 = CANMessageParser.parse_line(standard, default_timestamp=10.0)
    assert parsed2 is not None
    assert parsed2['timestamp'] == 10.0
    assert parsed2['can_id'] == '0x0C0'
    assert parsed2['data'] == '0A66012000000000'


def test_can_parser_batch():
    multi_line = """
    # Sample comment
    can0  316   [8]  05 21 68 09 21 21 00 6F
    can0  000   [8]  00 00 00 00 00 00 00 00
    """
    records = CANMessageParser.parse_batch(multi_line)
    assert len(records) == 2
    assert records[0]['can_id'] == '0x316'
    assert records[1]['can_id'] == '0x000'


def test_can_stream_processor():
    sp = CANStreamProcessor()
    lines = [
        {'timestamp': 1.000, 'can_id': '0x316', 'dlc': 8, 'data': '052168092121006F'},
        {'timestamp': 1.001, 'can_id': '0x000', 'dlc': 8, 'data': '0000000000000000'},
    ]
    results = sp.process_records(lines)
    assert len(results) == 2
    assert results[0]['can_id'] == '0x316'
    assert results[1]['can_id'] == '0x000'
    assert 'attack_type' in results[1]
    assert 'confidence' in results[1]
    assert 'severity' in results[1]
    assert 'inter_arrival_time' in results[1]['features']
    assert results[1]['features']['inter_arrival_time'] == pytest.approx(0.001, rel=1e-3)


def test_can_attack_simulator_hcrl_attacks():
    gen = CANDataGenerator()
    att = CANAttackSimulator()
    base = gen.generate_normal_stream(duration_sec=0.2)

    fuzzy = att.inject_fuzzy_attack(base, num_attacks=5)
    assert 'FUZZY' in fuzzy['label'].values

    gear = att.inject_gear_attack(base, num_attacks=5)
    assert 'GEAR' in gear['label'].values

    rpm = att.inject_rpm_attack(base, num_attacks=5)
    assert 'RPM' in rpm['label'].values


def test_api_can_ingest_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'ingest.db'))
    init_db()

    client = flask_app.app.test_client()
    raw_text = 'can0 316 [8] 05 21 68 09 21 21 00 6F\ncan0 000 [8] 00 00 00 00 00 00 00 00'
    response = client.post('/api/can/ingest', json={'payload': raw_text, 'save_to_db': True})

    assert response.status_code == 200
    payload = response.json
    assert payload['status'] == 'success'
    assert payload['processed_count'] == 2
    assert payload['saved_to_db'] is True
    assert len(payload['records']) == 2

    # Check database
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM can_events').fetchone()[0]
    conn.close()
    assert count == 2


def test_api_can_stream_endpoint():
    client = flask_app.app.test_client()
    response = client.post('/api/can/stream', json={'label': 'DOS', 'limit': 5, 'save_to_db': False})

    assert response.status_code == 200
    data = response.json
    assert data['status'] == 'success'
    assert data['label'] == 'DOS'
    assert data['processed_count'] == 5
    assert data['true_positive_count'] == 5


def test_api_simulate_batch_supports_hcrl_modes():
    client = flask_app.app.test_client()
    for mode in ['FUZZY', 'GEAR', 'RPM']:
        resp = client.post('/api/simulate-batch', json={'domain': 'can', 'mode': mode})
        assert resp.status_code == 200
        assert resp.json['status'] == 'success'


def test_dataset_batch_save_to_db_option(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'DATABASE_PATH', str(tmp_path / 'batch_save.db'))
    init_db()

    client = flask_app.app.test_client()
    resp = client.post('/api/dataset-batch', json={
        'dataset': 'can_hcrl',
        'label': 'DOS',
        'limit': 5,
        'save_to_db': True
    })
    assert resp.status_code == 200
    assert resp.json['saved_to_db'] is True

    conn = get_db_connection()
    alerts = conn.execute('SELECT COUNT(*) FROM can_alerts').fetchone()[0]
    conn.close()
    assert alerts == 5

