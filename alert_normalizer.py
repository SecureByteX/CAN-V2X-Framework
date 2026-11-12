"""Convert the existing CAN and V2X prediction dictionaries to SOC alerts."""

import hashlib
import json
from datetime import datetime, timezone


class AlertValidationError(ValueError):
    pass


def _first_present(record, *names):
    for name in names:
        value = record.get(name)
        if value is not None and value != '':
            return value
    return None


def _normalize_timestamp(value):
    if value is None or value == '':
        raise AlertValidationError('Alert timestamp is required.')
    text = str(value).strip()
    if not text:
        raise AlertValidationError('Alert timestamp is required.')
    try:
        numeric = float(text)
        if not (numeric == numeric and abs(numeric) != float('inf')):
            raise ValueError
        return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError as exc:
        raise AlertValidationError('Alert timestamp must be epoch seconds or ISO-8601.') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def normalize_alert(record, source=None, event_id=None):
    if not isinstance(record, dict):
        raise AlertValidationError('Alert record must be an object.')
    normalized_source = str(source or record.get('source') or '').strip().upper()
    if normalized_source in {'V2X/VEREMI', 'VEREMI'}:
        normalized_source = 'V2X'
    if normalized_source not in {'CAN', 'V2X'}:
        raise AlertValidationError('Alert source must be CAN or V2X.')

    attack_type = _first_present(record, 'attack_type', 'predicted_class', 'prediction')
    if attack_type is None or not str(attack_type).strip():
        raise AlertValidationError('Predicted attack type is required.')
    predicted = str(attack_type).strip()
    actual = _first_present(record, 'actual_label')
    confidence = _first_present(record, 'confidence')
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise AlertValidationError('Confidence must be numeric.') from exc
        if not 0 <= confidence <= 1:
            raise AlertValidationError('Confidence must be between 0 and 1.')

    outcome = str(_first_present(record, 'outcome', 'prediction_outcome') or 'UNVERIFIED').strip().upper()
    features = _first_present(record, 'features', 'relevant_features')
    if features is not None and not isinstance(features, dict):
        raise AlertValidationError('Relevant features must be an object when provided.')
    alert = {
        'event_id': str(event_id) if event_id is not None else None,
        'source': normalized_source,
        'timestamp': _normalize_timestamp(_first_present(record, 'timestamp', 'rcvTime')),
        'attack_type': predicted,
        'predicted_class': predicted,
        'confidence': confidence,
        'severity': str(record.get('severity') or 'UNKNOWN').strip().upper(),
        'prediction_outcome': outcome,
        'vehicle_id': _first_present(record, 'vehicle_id', 'vehicleId'),
        'can_id': _first_present(record, 'can_id') if normalized_source == 'CAN' else None,
        'sender_id': _first_present(record, 'sender_id') if normalized_source == 'V2X' else None,
        'receiver_id': _first_present(record, 'receiver_id') if normalized_source == 'V2X' else None,
        'relevant_features': features,
        'model_name': _first_present(record, 'model_name', 'model'),
        'actual_label': str(actual).strip() if actual is not None else None,
    }
    identity = {
        key: value for key, value in alert.items()
        if key not in {'event_id'}
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), default=str)
    alert['fingerprint'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    alert['alert_id'] = hashlib.sha256(('v-soc:' + canonical).encode('utf-8')).hexdigest()[:32]
    return alert
