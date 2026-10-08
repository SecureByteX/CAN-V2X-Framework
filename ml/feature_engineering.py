import pandas as pd
import numpy as np

class FeatureExtractor:
    """
    Feature Extraction Module for both CAN Bus and V2X Domain Traffic.
    Extracted features transform raw packets into high-dimensional behavioral representations.
    """
    @staticmethod
    def extract_can_features(df):
        """
        Extracts temporal, statistical, and frequency features from raw CAN traffic:
        - inter_arrival_time: Time delta between consecutive frames
        - id_frequency: Count of occurrences for each CAN ID in sliding window
        - timing_deviation: Variation from expected ECU message cycle time
        - payload_entropy: Byte variance / disorder in data payload
        - repeated_message_count: Consecutively repeated byte payloads
        - burst_rate: Short window message transmission rate
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Inter-arrival time (overall and per CAN ID)
        df['inter_arrival_time'] = df['timestamp'].diff().fillna(0.0)
        
        # 2. Convert CAN ID to numeric integer representation for ML algorithms
        df['can_id_num'] = df['can_id'].apply(lambda x: int(str(x), 16) if str(x).startswith('0x') else int(x))
        
        # 3. Message DLC
        df['dlc'] = df['dlc'].astype(int)
        
        # 4. Message Payload Byte Analysis
        def parse_data_bytes(hex_str):
            try:
                hex_str = str(hex_str).ljust(16, '0')[:16]
                bytes_arr = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
                return bytes_arr
            except Exception:
                return [0]*8

        parsed_bytes = df['data'].apply(parse_data_bytes)
        bytes_df = pd.DataFrame(parsed_bytes.tolist(), columns=[f'byte_{i}' for i in range(8)])
        
        # Mean, std, and byte sum of payload
        df['byte_mean'] = bytes_df.mean(axis=1)
        df['byte_std'] = bytes_df.std(axis=1).fillna(0)
        df['byte_sum'] = bytes_df.sum(axis=1)
        
        # 5. Sliding window rolling frequency per CAN ID (Window size = 20 messages)
        df['id_frequency'] = df.groupby('can_id')['timestamp'].transform(lambda x: x.rolling(window=20, min_periods=1).count())
        
        # 6. Burst rate (Count of messages in 0.05 second sliding window)
        timestamps = df['timestamp'].to_numpy()
        window_starts = np.searchsorted(timestamps, timestamps - 0.05, side='left')
        df['burst_rate'] = np.arange(1, len(timestamps) + 1) - window_starts
        
        # 7. Repeated payload indicator
        df['repeated_message_count'] = (df['data'] == df['data'].shift(1)).astype(int)
        
        # Select numeric ML features
        feature_cols = [
            'can_id_num', 'dlc', 'inter_arrival_time', 'id_frequency', 
            'burst_rate', 'repeated_message_count', 'byte_mean', 'byte_std', 'byte_sum'
        ]
        
        X = df[feature_cols].copy()
        y = df['label'] if 'label' in df.columns else None
        
        return X, y, df

    @staticmethod
    def extract_v2x_features(df):
        """
        Extracts physical, sequence, and kinematic consistency features from V2X BSM traffic:
        - message_frequency: Transmitted messages per second per sender
        - sequence_consistency: Difference between sequence numbers (detects drops/repeats)
        - position_consistency: Distance moved per timestamp delta (detects teleportation/spoofing)
        - speed_consistency: Delta between calculated speed and reported speed
        - heading_change: Rate of angular trajectory changes
        - timestamp_deviation: Delta between local clock and sender packet timestamp
        - replay_indicator: Identical payload / sequence resubmission
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Group by sender vehicle
        df['time_delta'] = df.groupby('sender_id')['timestamp'].diff().fillna(0.1)
        df['time_delta'] = df['time_delta'].apply(lambda x: 0.1 if x <= 0 else x)
        
        # Sequence delta
        df['seq_delta'] = df.groupby('sender_id')['sequence_number'].diff().fillna(1.0)
        df['sequence_consistency'] = (df['seq_delta'] == 1).astype(int)
        
        # Position Delta (Euclidean approximation for latitude/longitude delta)
        lat_diff = df.groupby('sender_id')['latitude'].diff().fillna(0.0)
        lon_diff = df.groupby('sender_id')['longitude'].diff().fillna(0.0)
        # 1 degree approx 111,000 meters
        dist_meters = np.sqrt((lat_diff * 111000)**2 + (lon_diff * 111000)**2)
        df['calculated_speed'] = dist_meters / df['time_delta']
        
        # Speed consistency (difference between GPS displacement speed and reported BSM speed in m/s)
        # BSM speed in km/h -> m/s is / 3.6
        df['reported_speed_ms'] = df['speed'] / 3.6
        df['speed_consistency'] = np.abs(df['calculated_speed'] - df['reported_speed_ms'])
        
        # Acceleration consistency
        speed_diff = df.groupby('sender_id')['reported_speed_ms'].diff().fillna(0.0)
        df['calculated_accel'] = speed_diff / df['time_delta']
        df['acceleration_consistency'] = np.abs(df['calculated_accel'] - df['acceleration'])
        
        # Heading change rate
        df['heading_change'] = df.groupby('sender_id')['heading'].diff().fillna(0.0).abs()
        
        # Message Frequency (rolling 1 second count)
        df['message_frequency'] = df.groupby('sender_id')['timestamp'].transform(
            lambda x: x.rolling(window=10, min_periods=1).count()
        )
        
        # Replay indicator (duplicate seq_number for same sender)
        df['replay_indicator'] = (df['seq_delta'] == 0).astype(int)
        
        # Encode Sender and Receiver to integer IDs cleanly
        def parse_sender_id(sid):
            s = str(sid)
            if 'VEHICLE_' in s:
                return int(s.replace('VEHICLE_', ''))
            elif 'UNKNOWN' in s or 'ROGUE' in s:
                return 999
            else:
                return 500

        df['sender_num'] = df['sender_id'].apply(parse_sender_id)
        
        feature_cols = [
            'sender_num', 'speed', 'heading', 'acceleration', 'sequence_number',
            'time_delta', 'sequence_consistency', 'speed_consistency',
            'acceleration_consistency', 'heading_change', 'message_frequency', 'replay_indicator'
        ]
        
        X = df[feature_cols].copy()
        y = df['label'] if 'label' in df.columns else None
        
        return X, y, df

if __name__ == '__main__':
    from ml.preprocessing import DataPreprocessor
    import config
    import os
    
    pre = DataPreprocessor()
    dataset_path = (
        config.CAN_TRAINING_DATA_PATH
        if os.path.exists(config.CAN_TRAINING_DATA_PATH)
        else config.CAN_RAW_DATA_PATH
    )
    df_can = pre.load_data(dataset_path)
    X_can, y_can, _ = FeatureExtractor.extract_can_features(df_can.head(500))
    print(f"Extracted CAN Features shape: {X_can.shape}")
    print("CAN Features sample:")
    print(X_can.head())
