import re
from typing import Any, Dict, List, Optional, Union


class CANMessageParser:
    """
    Parser for heterogeneous CAN-BUS traffic formats.
    Supports:
    - Structured dictionary / JSON objects
    - HCRL raw text logs ('Timestamp: ... ID: ... DLC: ... DATA...')
    - HCRL raw CSV lines ('1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R')
    - Candump socketCAN compact log ('(1478198376.389427) can0 316#052168092121006F')
    - Candump socketCAN standard log ('can0  316   [8]  05 21 68 09 21 21 00 6F')
    """

    HCRL_TEXT_PATTERN = re.compile(
        r'Timestamp:\s*(\S+)\s+ID:\s*([0-9a-fA-F]+)\s+\S*\s*DLC:\s*(\d+)\s*(.*)'
    )
    CANDUMP_COMPACT_PATTERN = re.compile(
        r'^\s*(?:\((\d+\.?\d*)\)\s+)?\S+\s+([0-9a-fA-F]{3,8})#([0-9a-fA-F]*)\s*$'
    )
    CANDUMP_STANDARD_PATTERN = re.compile(
        r'^\s*(?:\((\d+\.?\d*)\)\s+)?\S+\s+([0-9a-fA-F]{3,8})\s+\[(\d+)\]\s*(.*)$'
    )

    @classmethod
    def parse_dict(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes a structured CAN dictionary."""
        timestamp = float(record.get('timestamp', 0.0))
        raw_id = str(record.get('can_id', '')).strip()
        if raw_id.lower().startswith('0x'):
            can_id_int = int(raw_id, 16)
        else:
            can_id_int = int(raw_id, 16) if any(c in raw_id.lower() for c in 'abcdef') else int(raw_id)
        
        can_id = f'0x{can_id_int:03X}'
        dlc = int(record.get('dlc', 8))
        if not 0 <= dlc <= 8:
            raise ValueError(f'Invalid DLC value: {dlc}')

        raw_data = str(record.get('data', '')).strip().replace(' ', '').upper()
        # Ensure 16-character hex representation for 8 bytes
        expected_len = dlc * 2
        if len(raw_data) < expected_len:
            raw_data = raw_data.ljust(expected_len, '0')
        elif len(raw_data) > expected_len:
            raw_data = raw_data[:expected_len]

        return {
            'timestamp': timestamp,
            'can_id': can_id,
            'dlc': dlc,
            'data': raw_data,
            'label': record.get('label')
        }

    @classmethod
    def parse_line(cls, line: str, default_timestamp: float = 0.0) -> Optional[Dict[str, Any]]:
        """Parses a single log line into a standardized CAN message dict."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        # 1. HCRL raw text format
        hcrl_match = cls.HCRL_TEXT_PATTERN.match(line)
        if hcrl_match:
            ts, cid, dlc, payload_str = hcrl_match.groups()
            payload_bytes = payload_str.strip().split()
            data_hex = ''.join(b.strip().zfill(2) for b in payload_bytes).upper()
            return {
                'timestamp': float(ts),
                'can_id': f'0x{int(cid, 16):03X}',
                'dlc': int(dlc),
                'data': data_hex.ljust(int(dlc) * 2, '0')[:int(dlc) * 2],
                'label': 'NORMAL'
            }

        # 2. HCRL CSV format (e.g. 1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R)
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                try:
                    ts = float(parts[0])
                    cid = f'0x{int(parts[1], 16):03X}'
                    dlc = int(parts[2])
                    # payload bytes start from index 3
                    if len(parts) > 3 + dlc:
                        payload_bytes = parts[3:3 + dlc]
                        label_tag = parts[-1].upper()
                    else:
                        payload_bytes = parts[3:]
                        label_tag = 'NORMAL'
                    
                    data_hex = ''.join(b.zfill(2) for b in payload_bytes).upper()
                    label = 'NORMAL' if label_tag == 'R' else ('ATTACK' if label_tag == 'T' else label_tag)
                    return {
                        'timestamp': ts,
                        'can_id': cid,
                        'dlc': dlc,
                        'data': data_hex.ljust(dlc * 2, '0')[:dlc * 2],
                        'label': label
                    }
                except (ValueError, IndexError):
                    pass

        # 3. Candump compact socketCAN format (can0 316#052168092121006F)
        compact_match = cls.CANDUMP_COMPACT_PATTERN.match(line)
        if compact_match:
            ts_str, cid_str, data_hex = compact_match.groups()
            ts = float(ts_str) if ts_str else default_timestamp
            cid = f'0x{int(cid_str, 16):03X}'
            data_hex = data_hex.upper()
            dlc = len(data_hex) // 2
            return {
                'timestamp': ts,
                'can_id': cid,
                'dlc': dlc,
                'data': data_hex,
                'label': None
            }

        # 4. Candump standard socketCAN format (can0 316 [8] 05 21 68 09 21 21 00 6F)
        std_match = cls.CANDUMP_STANDARD_PATTERN.match(line)
        if std_match:
            ts_str, cid_str, dlc_str, payload_str = std_match.groups()
            ts = float(ts_str) if ts_str else default_timestamp
            cid = f'0x{int(cid_str, 16):03X}'
            dlc = int(dlc_str)
            payload_bytes = payload_str.strip().split()
            data_hex = ''.join(b.zfill(2) for b in payload_bytes).upper()
            return {
                'timestamp': ts,
                'can_id': cid,
                'dlc': dlc,
                'data': data_hex.ljust(dlc * 2, '0')[:dlc * 2],
                'label': None
            }

        raise ValueError(f'Unrecognized CAN message format: {line}')

    @classmethod
    def parse_batch(cls, raw_input: Union[str, List[Union[str, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Parses a multi-line string or mixed list of CAN messages."""
        results = []
        if isinstance(raw_input, str):
            lines = raw_input.strip().splitlines()
            for line in lines:
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

