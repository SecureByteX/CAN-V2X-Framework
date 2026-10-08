from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from database.models import EventDAO
from ml.can.parser import CANMessageParser
from ml.can.predict import CANPredictor
from ml.feature_engineering import FeatureExtractor


class CANStreamProcessor:
    """
    Real-time streaming processor for CAN-BUS packets.
    Maintains a rolling window buffer to compute temporal and frequency
    features on incoming messages, performs ML inference via CANPredictor,
    and optionally persists security alerts to the database.
    """

    def __init__(self, predictor: Optional[CANPredictor] = None, buffer_size: int = 100):
        self.predictor = predictor or CANPredictor()
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)

    def reset(self):
        """Clears the internal sliding window."""
        self.history.clear()

    def process_records(
        self,
        records: List[Dict[str, Any]],
        save_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of normalized CAN message dictionaries through the IDS pipeline.
        Calculates rolling features, executes model inference, and returns enriched results.
        """
        if not records:
            return []

        # Add records to rolling history for context
        for r in records:
            self.history.append({
                'timestamp': float(r['timestamp']),
                'can_id': str(r['can_id']),
                'dlc': int(r['dlc']),
                'data': str(r['data']),
                'label': r.get('label')
            })

        # Convert recent history to DataFrame to extract windowed features
        history_df = pd.DataFrame(list(self.history))
        X_feat, _, enriched_df = FeatureExtractor.extract_can_features(history_df)

        # Slice the newly arrived records from the end of the extracted features
        num_new = len(records)
        new_features = X_feat.iloc[-num_new:].copy().reset_index(drop=True)
        new_records_df = history_df.iloc[-num_new:].copy().reset_index(drop=True)

        # Prepare DataFrame with feature columns for the predictor
        for col in self.predictor.feature_names:
            if col in new_features.columns:
                new_records_df[col] = new_features[col]

        # Execute ML prediction
        predictions = self.predictor.predict_single_record(new_records_df)

        # Optionally persist security events to SQLite
        if save_to_db:
            for pred in predictions:
                EventDAO.save_can_event(pred)

        return predictions

    def process_raw_input(
        self,
        raw_text_or_lines: Any,
        save_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """Parses raw text/candump lines and processes them through the IDS."""
        parsed = CANMessageParser.parse_batch(raw_text_or_lines)
        return self.process_records(parsed, save_to_db=save_to_db)

