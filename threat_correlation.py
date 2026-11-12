"""Conservative time-window correlation for normalized alerts."""

from datetime import datetime, timezone


def _timestamp(value):
    text = str(value or '').strip()
    try:
        numeric = float(text)
        return datetime.fromtimestamp(numeric, timezone.utc).timestamp()
    except (ValueError, OverflowError, OSError):
        try:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


def attack_category(value):
    label = ''.join(ch for ch in str(value or '').upper() if ch.isalnum())
    if label.startswith('DOS'):
        return 'DOS'
    if 'SPOOF' in label or 'SYBIL' in label:
        return 'IDENTITY_SPOOFING'
    if 'REPLAY' in label or 'DELAYED' in label:
        return 'REPLAY_OR_DELAY'
    if label in {'FUZZY', 'INJECTION'}:
        return 'MESSAGE_MANIPULATION'
    return label or 'UNKNOWN'


def correlate(incoming, candidate, window_seconds=30):
    """Return a reason/strength only when time, category and identity agree."""
    if incoming.get('alert_id') == candidate.get('alert_id'):
        return None
    left_time = _timestamp(incoming.get('timestamp'))
    right_time = _timestamp(candidate.get('timestamp'))
    if left_time is None or right_time is None or abs(left_time - right_time) > window_seconds:
        return None
    if attack_category(incoming.get('attack_type')) != attack_category(candidate.get('attack_type')):
        return None

    left_source = str(incoming.get('source') or '').upper()
    right_source = str(candidate.get('source') or '').upper()
    if left_source == right_source == 'CAN':
        identifier = incoming.get('can_id')
        if identifier and identifier == candidate.get('can_id'):
            return {'strength': 0.75, 'reason': 'CAN alerts share a CAN message identifier and attack category within the configured time window.'}
        return None
    if left_source == right_source == 'V2X':
        left_identity = incoming.get('sender_id') or incoming.get('receiver_id')
        right_identity = candidate.get('sender_id') or candidate.get('receiver_id')
        if left_identity and left_identity == right_identity:
            return {'strength': 0.85, 'reason': 'V2X alerts share an explicit sender/receiver identifier and attack category within the configured time window.'}
        return None
    left_vehicle = incoming.get('vehicle_id')
    right_vehicle = candidate.get('vehicle_id')
    if left_source != right_source and left_vehicle and left_vehicle == right_vehicle:
        return {'strength': 1.0, 'reason': 'CAN and V2X alerts share the same explicitly supplied vehicle identifier, attack category, and time window.'}
    return None
