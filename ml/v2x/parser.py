import json
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ml.v2x.veremi import FEATURE_COLUMNS, RECEIVER_COLUMN, TARGET_COLUMN, TIME_COLUMN


class V2XMessageParser:
    """
    Parser for heterogeneous V2X and Basic Safety Message (BSM) communication.
    Supports:
    - Structured JSON/dictionary messages (VeReMi 16-feature or BSM kinematic format)
    - VeReMi raw CSV lines
    - Standard BSM text/JSON lines
    """

    @classmethod
    def parse_dict(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes a V2X dictionary record."""
        # Check if record has VeReMi format
        has_veremi_features = any(k in record for k in FEATURE_COLUMNS)
        if has_veremi_features:
            parsed = {
                'rcvTime': float(record.get('rcvTime', record.get('timestamp', 0.0))),
                'ReceiverID': str(record.get('ReceiverID', record.get('receiver_id', 'RSU_1'))),
                'AttackerType': record.get('AttackerType', record.get('actual_label', record.get('label'))),
            }
            for col in FEATURE_COLUMNS:
                parsed[col] = float(record.get(col, 0.0))
            return parsed

        # Otherwise, treat as BSM kinematic format
        return {
            'timestamp': float(record.get('timestamp', record.get('rcvTime', 0.0))),
            'message_type': str(record.get('message_type', 'BSM')),
            'sender_id': str(record.get('sender_id', 'VEHICLE_001')),
            'receiver_id': str(record.get('receiver_id', 'RSU_001')),
            'latitude': float(record.get('latitude', 12.9716)),
            'longitude': float(record.get('longitude', 77.5946)),
            'speed': float(record.get('speed', 50.0)),
            'heading': float(record.get('heading', 0.0)),
            'acceleration': float(record.get('acceleration', 0.0)),
            'sequence_number': int(record.get('sequence_number', 100)),
            'label': record.get('label', record.get('AttackerType'))
        }

    @classmethod
    def parse_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parses a single line of text or JSON."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        # Try JSON format first
        if line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                return cls.parse_dict(data)
            except json.JSONDecodeError:
                pass

        # Try comma-separated values (CSV)
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            # VeReMi CSV check: at least 19 fields (rcvTime, ReceiverID, 16 features, AttackerType)
            if len(parts) >= 19:
                try:
                    parsed = {
                        'rcvTime': float(parts[0]),
                        'ReceiverID': str(parts[1]),
                    }
                    for i, col in enumerate(FEATURE_COLUMNS):
                        parsed[col] = float(parts[2 + i])
                    parsed['AttackerType'] = parts[2 + len(FEATURE_COLUMNS)] if len(parts) > 2 + len(FEATURE_COLUMNS) else 'Benign'
                    return parsed
                except (ValueError, IndexError):
                    pass

            # BSM kinematic CSV: timestamp, message_type, sender_id, receiver_id, lat, lon, speed, heading, accel, seq, [label]
            if len(parts) >= 10:
                try:
                    return {
                        'timestamp': float(parts[0]),
                        'message_type': str(parts[1]),
                        'sender_id': str(parts[2]),
                        'receiver_id': str(parts[3]),
                        'latitude': float(parts[4]),
                        'longitude': float(parts[5]),
                        'speed': float(parts[6]),
                        'heading': float(parts[7]),
                        'acceleration': float(parts[8]),
                        'sequence_number': int(parts[9]),
                        'label': parts[10] if len(parts) > 10 else None
                    }
                except (ValueError, IndexError):
                    pass

        raise ValueError(f'Unrecognized V2X message format: {line}')

    @classmethod
    def parse_batch(cls, raw_input: Union[str, List[Union[str, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Parses multi-line text, JSON array, or list of records."""
        results = []
        if isinstance(raw_input, str):
            raw_input = raw_input.strip()
            if raw_input.startswith('[') and raw_input.endswith(']'):
                try:
                    json_items = json.loads(raw_input)
                    return [cls.parse_dict(item) for item in json_items if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass
            for line in raw_input.splitlines():
                parsed = cls.parse_line(line)
                if parsed:
                    results.append(parsed)
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, dict):
                    results.append(cls.parse_dict(item))
                elif isinstance(item, str):
                    parsed = cls.parse_line(item)
                    if parsed:
                        results.append(parsed)
        return results

