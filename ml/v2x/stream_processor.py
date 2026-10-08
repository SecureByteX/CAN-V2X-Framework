from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from database.models import EventDAO
from ml.prediction_outcomes import classify_prediction
from ml.v2x.parser import V2XMessageParser
from ml.v2x.predict import V2XPredictor
from ml.v2x.predict_veremi import VeReMiPredictor, veremi_severity
from ml.v2x.veremi import FEATURE_COLUMNS, RECEIVER_COLUMN, TARGET_COLUMN, TIME_COLUMN


class V2XStreamProcessor:
    """
    Real-time streaming processor for V2X BSM wireless messages.
    Supports real 20-class VeReMi models and kinematic feature prediction.
    Optionally persists security alerts to the database.
    """

    def __init__(
        self,
        veremi_predictor: Optional[VeReMiPredictor] = None,
        synthetic_predictor: Optional[V2XPredictor] = None,
        buffer_size: int = 100
    ):
        self.veremi_predictor = veremi_predictor or VeReMiPredictor()
        self.synthetic_predictor = synthetic_predictor or (
            V2XPredictor() if synthetic_predictor is None else None
        )
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)

    def reset(self):
        """Clears the internal history queue."""
        self.history.clear()

    def process_records(
        self,
        records: List[Dict[str, Any]],
        save_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """Processes parsed V2X message dictionaries through the IDS pipeline."""
        if not records:
            return []

        df = pd.DataFrame(records)

        # Check if records match VeReMi format (contain pos_0 or FEATURE_COLUMNS)
        if any(col in df.columns for col in FEATURE_COLUMNS):
            # Ensure all required feature columns exist, defaulting to 0.0
            for col in FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            if TIME_COLUMN not in df.columns:
                df[TIME_COLUMN] = df.get('timestamp', 0.0)
            if RECEIVER_COLUMN not in df.columns:
                df[RECEIVER_COLUMN] = df.get('receiver_id', 'RSU_001')
            if TARGET_COLUMN not in df.columns:
                df[TARGET_COLUMN] = df.get('actual_label', df.get('label', 'Unlabeled'))

            raw_predictions = self.veremi_predictor.predict_records(df)

            results = []
            for pred in raw_predictions:
                event_dict = {
                    'source': 'V2X',
                    'timestamp': str(pred['timestamp']),
                    'message_type': 'BSM',
                    'sender_id': str(pred['receiver_id']),  # VeReMi tracks receiver
                    'receiver_id': str(pred['receiver_id']),
                    'latitude': float(pred.get('pos_0', 0.0)),
                    'longitude': float(pred.get('pos_1', 0.0)),
                    'speed': float(pred.get('speed_0', 0.0)),
                    'heading': 0.0,
                    'acceleration': float(pred.get('acceleration_0', 0.0)),
                    'sequence_number': 0,
                    'attack_type': pred['attack_type'],
                    'actual_label': pred['actual_label'],
                    'outcome': pred['outcome'],
                    'confidence': pred['confidence'],
                    'severity': pred['severity'],
                    'status': 'Detected' if pred['attack_type'] != 'Benign' else 'Normal',
                    'model_name': self.veremi_predictor.model_name,
                    'features': {
                        'pos_0': pred.get('pos_0', 0.0),
                        'pos_1': pred.get('pos_1', 0.0),
                        'speed_0': pred.get('speed_0', 0.0),
                        'speed_1': pred.get('speed_1', 0.0),
                        'acceleration_0': pred.get('acceleration_0', 0.0),
                        'acceleration_1': pred.get('acceleration_1', 0.0),
                    }
                }
                results.append(event_dict)

                if save_to_db:
                    EventDAO.save_v2x_event(event_dict)

            return results

        # Otherwise fallback to kinematic BSM model
        if self.synthetic_predictor:
            predictions = self.synthetic_predictor.predict_single_record(df)
            if save_to_db:
                for p in predictions:
                    EventDAO.save_v2x_event(p)
            return predictions

        return []

    def process_raw_input(
        self,
        raw_text_or_lines: Any,
        save_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """Parses raw text or JSON and processes through V2X pipeline."""
        parsed = V2XMessageParser.parse_batch(raw_text_or_lines)
        return self.process_records(parsed, save_to_db=save_to_db)

