"""Persistence and lifecycle operations for SOC alerts and incidents."""

import json
import uuid
from datetime import datetime, timezone

from alert_normalizer import normalize_alert
from database.db import get_db_connection
from mitre_mapper import map_attack
from risk_engine import score_alert, score_incident
from threat_correlation import correlate


INCIDENT_STATUSES = {'NEW', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE'}
OPEN_STATUSES = {'NEW', 'INVESTIGATING'}
CORRELATION_WINDOW_SECONDS = 30
SEVERITY_ORDER = {'UNKNOWN': 0, 'INFO': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _row_dict(row):
    return dict(row) if row is not None else None


def _is_attack_alert(alert):
    predicted = ''.join(ch for ch in alert['predicted_class'].upper() if ch.isalnum())
    return predicted not in {'NORMAL', 'BENIGN'} and alert['prediction_outcome'] not in {'TRUE_NEGATIVE', 'FALSE_NEGATIVE'}


def _latest_assessment(cursor, alert_id):
    row = cursor.execute(
        'SELECT risk_score, risk_level, factors_json, explanation FROM risk_assessments WHERE alert_id = ? ORDER BY assessment_id DESC LIMIT 1',
        (alert_id,),
    ).fetchone()
    return {
        'risk_score': float(row['risk_score']), 'risk_level': row['risk_level'],
        'factors': json.loads(row['factors_json']), 'explanation': row['explanation'],
    } if row else None


def _latest_incident_assessment(cursor, incident_id):
    row = cursor.execute(
        'SELECT risk_score, risk_level, factors_json, explanation FROM risk_assessments WHERE incident_id = ? ORDER BY assessment_id DESC LIMIT 1',
        (incident_id,),
    ).fetchone()
    return {
        'risk_score': float(row['risk_score']), 'risk_level': row['risk_level'],
        'factors': json.loads(row['factors_json']), 'explanation': row['explanation'],
    } if row else None


def _alert_record(row):
    value = dict(row)
    try:
        value['relevant_features'] = json.loads(value.pop('relevant_features_json') or '{}')
    except (TypeError, ValueError):
        value['relevant_features'] = {}
    return value


def _insert_risk(cursor, alert, assessment):
    factors = assessment['factors']
    explanation = (
        f"Risk {assessment['risk_score']}/100 ({assessment['risk_level']}) uses "
        f"{alert['severity']} severity and {alert['prediction_outcome']} evidence; "
        'confidence contributes 30% and is not treated as proof.'
    )
    cursor.execute(
        '''INSERT INTO risk_assessments (alert_id, risk_score, risk_level, factors_json, explanation)
           VALUES (?, ?, ?, ?, ?)''',
        (alert['alert_id'], assessment['risk_score'], assessment['risk_level'], _json(factors), explanation),
    )


def record_predictions(records, source):
    """Normalize, de-duplicate, score and associate prediction records."""
    inserted = []
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        for record in records or []:
            alert = normalize_alert(record, source=source, event_id=record.get('id'))
            existing = cursor.execute(
                'SELECT alert_id FROM normalized_alerts WHERE fingerprint = ?', (alert['fingerprint'],)
            ).fetchone()
            if existing:
                inserted.append(existing['alert_id'])
                continue
            cursor.execute(
                '''INSERT INTO normalized_alerts (
                    alert_id, fingerprint, event_id, source, timestamp, attack_type,
                    predicted_class, confidence, severity, prediction_outcome,
                    vehicle_id, can_id, sender_id, receiver_id, relevant_features_json, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    alert['alert_id'], alert['fingerprint'], alert['event_id'], alert['source'],
                    alert['timestamp'], alert['attack_type'], alert['predicted_class'],
                    alert['confidence'], alert['severity'], alert['prediction_outcome'],
                    alert['vehicle_id'], alert['can_id'], alert['sender_id'], alert['receiver_id'],
                    _json(alert['relevant_features']) if alert['relevant_features'] is not None else None,
                    alert['model_name'],
                ),
            )
            assessment = score_alert(alert)
            _insert_risk(cursor, alert, assessment)
            if _is_attack_alert(alert):
                _attach_to_incident(cursor, alert)
            inserted.append(alert['alert_id'])
        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _attach_to_incident(cursor, alert):
    active = cursor.execute(
        '''SELECT a.*, m.incident_id FROM normalized_alerts a
           JOIN incident_alert_mapping m ON m.alert_id = a.alert_id
           JOIN correlated_incidents i ON i.incident_id = m.incident_id
           WHERE i.status IN ('NEW', 'INVESTIGATING')
           ORDER BY a.timestamp DESC LIMIT 500'''
    ).fetchall()
    best = None
    for row in active:
        candidate = dict(row)
        match = correlate(alert, candidate, CORRELATION_WINDOW_SECONDS)
        if match and (best is None or match['strength'] > best['match']['strength']):
            best = {'incident_id': candidate['incident_id'], 'match': match}

    if best:
        incident_id = best['incident_id']
        reason = best['match']['reason']
        strength = best['match']['strength']
        cursor.execute(
            'UPDATE correlated_incidents SET correlation_reason = ?, correlation_strength = CASE WHEN correlation_strength < ? THEN ? ELSE correlation_strength END, updated_at = CURRENT_TIMESTAMP, last_alert_at = ? WHERE incident_id = ?',
            (reason, strength, strength, alert['timestamp'], incident_id),
        )
    else:
        incident_id = str(uuid.uuid4())
        title = f"{alert['source']} {alert['attack_type']} activity"
        reason = 'Single alert; no reliable related alert was found within the configured time window.'
        cursor.execute(
            '''INSERT INTO correlated_incidents (
                incident_id, title, status, severity, correlation_reason, created_at, updated_at, last_alert_at
            ) VALUES (?, ?, 'NEW', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)''',
            (incident_id, title, alert['severity'], reason, alert['timestamp']),
        )
        cursor.execute(
            "INSERT INTO incident_history (incident_id, previous_status, new_status, note, changed_by) VALUES (?, NULL, 'NEW', 'Incident created from a normalized model prediction.', 'system')",
            (incident_id,),
        )

    cursor.execute(
        'INSERT INTO incident_alert_mapping (incident_id, alert_id) VALUES (?, ?)',
        (incident_id, alert['alert_id']),
    )
    _recalculate_incident(cursor, incident_id)
    for mapping in map_attack(alert['attack_type']):
        cursor.execute(
            '''INSERT INTO mitre_mappings (
                incident_id, technique_id, framework, technique_name,
                justification, reference_url, requires_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (incident_id, technique_id, framework) DO NOTHING''',
            (incident_id, mapping['technique_id'], mapping['framework'], mapping['technique_name'],
             mapping['justification'], mapping['reference_url'], int(mapping['requires_review'])),
        )


def _recalculate_incident(cursor, incident_id):
    rows = cursor.execute(
        '''SELECT a.alert_id, a.severity, r.risk_score, r.risk_level, r.factors_json, r.explanation
           FROM incident_alert_mapping m
           JOIN normalized_alerts a ON a.alert_id = m.alert_id
           JOIN risk_assessments r ON r.alert_id = a.alert_id
           WHERE m.incident_id = ?
           ORDER BY r.assessment_id DESC''',
        (incident_id,),
    ).fetchall()
    unique = {}
    for row in rows:
        unique.setdefault(row['alert_id'], row)
    values = list(unique.values())
    existing = cursor.execute(
        'SELECT correlation_strength FROM correlated_incidents WHERE incident_id = ?', (incident_id,)
    ).fetchone()
    aggregate = score_incident(
        [{'risk_score': float(row['risk_score'])} for row in values],
        float(existing['correlation_strength']) if existing else 0.0,
    )
    severity = max((row['severity'] for row in values), key=lambda item: SEVERITY_ORDER.get(item or 'UNKNOWN', 0))
    explanation = f"Incident score {aggregate['risk_score']}/100: highest alert risk {aggregate['factors']['highest_alert_score']}, {len(values)} related alert(s), plus corroboration/correlation adjustments."
    cursor.execute(
        'UPDATE correlated_incidents SET risk_score = ?, risk_level = ?, severity = ?, updated_at = CURRENT_TIMESTAMP WHERE incident_id = ?',
        (aggregate['risk_score'], aggregate['risk_level'], severity, incident_id),
    )
    cursor.execute(
        'INSERT INTO risk_assessments (incident_id, risk_score, risk_level, factors_json, explanation) VALUES (?, ?, ?, ?, ?)',
        (incident_id, aggregate['risk_score'], aggregate['risk_level'], _json(aggregate['factors']), explanation),
    )


def list_alerts(limit=100, source=None, outcome=None):
    conn = get_db_connection()
    try:
        conditions, params = [], []
        if source:
            conditions.append('a.source = ?')
            params.append(source.upper())
        if outcome:
            conditions.append('a.prediction_outcome = ?')
            params.append(outcome.upper())
        where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''
        rows = conn.execute(
            f'''SELECT a.*, r.risk_score, r.risk_level, r.factors_json, r.explanation AS risk_explanation
                FROM normalized_alerts a LEFT JOIN risk_assessments r ON r.assessment_id = (
                    SELECT MAX(r2.assessment_id) FROM risk_assessments r2 WHERE r2.alert_id = a.alert_id
                ) {where} ORDER BY a.timestamp DESC LIMIT ?''',
            params + [limit],
        ).fetchall()
        result = []
        for row in rows:
            item = _alert_record(row)
            if item.get('factors_json'):
                item['risk_factors'] = json.loads(item.pop('factors_json'))
            result.append(item)
        return result
    finally:
        conn.close()


def list_incidents(status=None, limit=100):
    conn = get_db_connection()
    try:
        params = []
        where = ''
        if status:
            where = 'WHERE i.status = ?'
            params.append(status.upper())
        rows = conn.execute(
            f'''SELECT i.*, COUNT(m.alert_id) AS alert_count
                FROM correlated_incidents i LEFT JOIN incident_alert_mapping m ON m.incident_id = i.incident_id
                {where} GROUP BY i.incident_id ORDER BY i.updated_at DESC LIMIT ?''',
            params + [limit],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_incident(incident_id):
    conn = get_db_connection()
    try:
        incident = conn.execute('SELECT * FROM correlated_incidents WHERE incident_id = ?', (incident_id,)).fetchone()
        if not incident:
            return None
        result = dict(incident)
        alerts = conn.execute(
            '''SELECT a.*, m.linked_at FROM normalized_alerts a
               JOIN incident_alert_mapping m ON m.alert_id = a.alert_id
               WHERE m.incident_id = ? ORDER BY a.timestamp''',
            (incident_id,),
        ).fetchall()
        result['alerts'] = [_alert_record(row) for row in alerts]
        result['history'] = [dict(row) for row in conn.execute(
            'SELECT * FROM incident_history WHERE incident_id = ? ORDER BY history_id', (incident_id,)
        ).fetchall()]
        result['risk'] = _latest_incident_assessment(conn.cursor(), incident_id)
        result['mitre'] = [dict(row) for row in conn.execute(
            'SELECT technique_id, framework, technique_name, justification, reference_url, requires_review FROM mitre_mappings WHERE incident_id = ?',
            (incident_id,),
        ).fetchall()]
        investigation = conn.execute(
            'SELECT * FROM llm_investigations WHERE incident_id = ? ORDER BY created_at DESC LIMIT 1',
            (incident_id,),
        ).fetchone()
        result['investigation'] = dict(investigation) if investigation else None
        if result['investigation'] and result['investigation'].get('result_json'):
            result['investigation']['result'] = json.loads(result['investigation'].pop('result_json'))
        return result
    finally:
        conn.close()


def update_incident_status(incident_id, status, changed_by=None, note=None, assigned_analyst=None):
    normalized = str(status or '').strip().upper()
    if normalized not in INCIDENT_STATUSES:
        raise ValueError('Status must be NEW, INVESTIGATING, RESOLVED, or FALSE_POSITIVE.')
    if note is not None and len(str(note)) > 2000:
        raise ValueError('Investigation note cannot exceed 2000 characters.')
    if changed_by is not None and len(str(changed_by)) > 120:
        raise ValueError('Analyst name cannot exceed 120 characters.')
    if assigned_analyst is not None and len(str(assigned_analyst)) > 120:
        raise ValueError('Assigned analyst cannot exceed 120 characters.')
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        incident = cursor.execute('SELECT status FROM correlated_incidents WHERE incident_id = ?', (incident_id,)).fetchone()
        if not incident:
            return None
        previous = incident['status']
        cursor.execute(
            'UPDATE correlated_incidents SET status = ?, assigned_analyst = COALESCE(?, assigned_analyst), updated_at = CURRENT_TIMESTAMP WHERE incident_id = ?',
            (normalized, assigned_analyst, incident_id),
        )
        cursor.execute(
            'INSERT INTO incident_history (incident_id, previous_status, new_status, note, changed_by) VALUES (?, ?, ?, ?, ?)',
            (incident_id, previous, normalized, note, changed_by),
        )
        conn.commit()
        return get_incident(incident_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
