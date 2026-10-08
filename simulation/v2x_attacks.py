import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.v2x_generator import V2XDataGenerator
import config

class V2XAttackSimulator:
    """
    Simulates V2X wireless cyberattacks locally:
    1. V2X Spoofing: Impersonating vehicle IDs with physically impossible position/speed jumps
    2. V2X Replay: Re-broadcasting old valid BSM messages with duplicate sequence numbers
    3. V2X DoS/Flooding: High-frequency flooding of bogus BSMs from rogue senders
    4. False Safety Message: Injecting fake emergency brake / collision warnings with extreme acceleration/speed drops
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.generator = V2XDataGenerator(seed=seed)

    def inject_spoofing_attack(self, df_base, num_attacks=40):
        """Impersonates a vehicle with telepathic/impossible position jumps (Ghost Vehicle)."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        
        for ts in timestamps:
            attacks.append({
                'timestamp': round(ts, 6),
                'message_type': 'BSM',
                'sender_id': 'VEHICLE_001',
                'receiver_id': 'RSU_001',
                'latitude': 12.9716 + np.random.uniform(0.05, 0.1), # ~10 km teleportation jump
                'longitude': 77.5946 + np.random.uniform(0.05, 0.1),
                'speed': 180.0, # Excessive speed
                'heading': 45.0,
                'acceleration': 8.5,
                'sequence_number': int(np.random.randint(500, 999)),
                'label': 'SPOOFING'
            })
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_replay_attack(self, df_base, num_attacks=40):
        """Re-sends an old observed BSM message with stale sequence numbers."""
        sample_bsm = df_base[df_base['sender_id'] == 'VEHICLE_002'].iloc[0]
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        
        for ts in timestamps:
            attacks.append({
                'timestamp': round(ts, 6),
                'message_type': 'BSM',
                'sender_id': sample_bsm['sender_id'],
                'receiver_id': sample_bsm['receiver_id'],
                'latitude': sample_bsm['latitude'],
                'longitude': sample_bsm['longitude'],
                'speed': sample_bsm['speed'],
                'heading': sample_bsm['heading'],
                'acceleration': sample_bsm['acceleration'],
                'sequence_number': sample_bsm['sequence_number'], # Duplicate sequence number
                'label': 'REPLAY'
            })
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_dos_attack(self, df_base, start_time=3.0, duration=1.0, burst_freq_sec=0.01):
        """Floods DSRC wireless channel with high-rate BSM traffic from rogue sender IDs."""
        attacks = []
        current_time = start_time
        end_time = start_time + duration
        
        seq = 1000
        while current_time < end_time:
            attacks.append({
                'timestamp': round(current_time, 6),
                'message_type': 'BSM',
                'sender_id': 'UNKNOWN_ROGUE',
                'receiver_id': 'RSU_001',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'speed': 0.0,
                'heading': 0.0,
                'acceleration': 0.0,
                'sequence_number': seq,
                'label': 'DOS'
            })
            current_time += burst_freq_sec
            seq += 1
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_false_safety_attack(self, df_base, num_attacks=30):
        """Injects false emergency brake warnings with extreme deceleration & invalid heading spikes."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        
        for ts in timestamps:
            attacks.append({
                'timestamp': round(ts, 6),
                'message_type': 'BSM_EMERGENCY',
                'sender_id': 'VEHICLE_003',
                'receiver_id': 'RSU_002',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'speed': 5.0,
                'heading': 270.0,
                'acceleration': -14.5, # Severe illegal braking deceleration (-14.5 m/s^2)
                'sequence_number': int(np.random.randint(100, 300)),
                'label': 'FALSE_SAFETY'
            })
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def generate_complete_v2x_dataset(self, duration_sec=30, output_path=None):
        print(f"[V2X Simulator] Generating {duration_sec}s base normal V2X BSM traffic...")
        df = self.generator.generate_normal_v2x_stream(duration_sec=duration_sec)
        
        print("[V2X Simulator] Injecting V2X SPOOFING attacks...")
        df = self.inject_spoofing_attack(df, num_attacks=int(duration_sec * 3))
        
        print("[V2X Simulator] Injecting V2X REPLAY attacks...")
        df = self.inject_replay_attack(df, num_attacks=int(duration_sec * 3))
        
        print("[V2X Simulator] Injecting V2X DOS flooding bursts...")
        df = self.inject_dos_attack(df, start_time=4.0, duration=1.0, burst_freq_sec=0.005)
        df = self.inject_dos_attack(df, start_time=15.0, duration=1.0, burst_freq_sec=0.005)
        
        print("[V2X Simulator] Injecting FALSE SAFETY WARNING attacks...")
        df = self.inject_false_safety_attack(df, num_attacks=int(duration_sec * 2.5))
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[V2X Simulator] Saved V2X dataset to: {output_path}")
            
        return df

if __name__ == '__main__':
    simulator = V2XAttackSimulator()
    df_v2x_all = simulator.generate_complete_v2x_dataset(duration_sec=10, output_path=config.V2X_RAW_DATA_PATH)
    print("\nV2X Attack dataset distribution:")
    print(df_v2x_all['label'].value_counts())
