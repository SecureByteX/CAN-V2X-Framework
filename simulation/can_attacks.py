import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.can_generator import CANDataGenerator
import config

class CANAttackSimulator:
    """
    Simulates CAN Bus cyberattacks locally:
    1. Injection: Malicious frames inserted into stream with abnormal payload
    2. Spoofing: Impersonation of critical ECU ID (e.g. 0x0C0 Engine) with high frequency
    3. DoS/Flooding: Burst flooding of high-priority CAN ID (0x000) at zero inter-arrival time
    4. Replay: Re-injecting old observed message payloads with current timestamps
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.generator = CANDataGenerator(seed=seed)

    def inject_injection_attack(self, df_base, num_attacks=50):
        """Injects unknown/unauthorized CAN IDs into the stream."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        
        injection_ids = ['0x7E0', '0x7E8', '0x1AA', '0x5FF']
        for ts in timestamps:
            cid = np.random.choice(injection_ids)
            dlc = 8
            data = "".join([f"{b:02X}" for b in np.random.randint(200, 256, size=8)])
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': cid,
                'dlc': dlc,
                'data': data,
                'label': 'INJECTION'
            })
        
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_spoofing_attack(self, df_base, target_id='0x0C0', num_attacks=50):
        """Impersonates a valid ECU ID (0x0C0) but sends spoofed malicious values."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        
        for ts in timestamps:
            # Malicious payload: extreme RPM values (e.g. 9999 RPM)
            spoofed_bytes = [0x27, 0x0F, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]
            data = "".join([f"{b:02X}" for b in spoofed_bytes])
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': target_id,
                'dlc': 8,
                'data': data,
                'label': 'SPOOFING'
            })
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_dos_attack(self, df_base, start_time=2.0, duration=0.5, burst_freq_ms=0.5):
        """Floods the bus with 0x000 (highest priority ID) at extremely high frequency."""
        attacks = []
        current_time = start_time
        end_time = start_time + duration
        
        while current_time < end_time:
            attacks.append({
                'timestamp': round(current_time, 6),
                'can_id': '0x000',
                'dlc': 8,
                'data': '0000000000000000',
                'label': 'DOS'
            })
            current_time += (burst_freq_ms / 1000.0)
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_replay_attack(self, df_base, num_attacks=40):
        """Captures a legitimate prior message payload and replays it repeatedly."""
        # Grab legitimate payload sample
        sample_msg = df_base[df_base['can_id'] == '0x200'].iloc[0]
        replay_payload = sample_msg['data']
        
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        for ts in timestamps:
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': '0x200',
                'dlc': sample_msg['dlc'],
                'data': replay_payload,
                'label': 'REPLAY'
            })
            
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_fuzzy_attack(self, df_base, num_attacks=50):
        """Injects messages with completely randomized arbitration IDs and payloads (HCRL Fuzzy)."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        for ts in timestamps:
            rand_id = f"0x{np.random.randint(0, 0x800):03X}"
            data = "".join([f"{b:02X}" for b in np.random.randint(0, 256, size=8)])
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': rand_id,
                'dlc': 8,
                'data': data,
                'label': 'FUZZY'
            })
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_gear_attack(self, df_base, target_id='0x43F', num_attacks=50):
        """Injects spoofed transmission gear control frames (HCRL Gear)."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        for ts in timestamps:
            # Malicious gear shift payload
            gear_bytes = [0x00, 0x50, 0x7F, 0x00, 0x00, 0x20, 0x00, 0x00]
            data = "".join([f"{b:02X}" for b in gear_bytes])
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': target_id,
                'dlc': 8,
                'data': data,
                'label': 'GEAR'
            })
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def inject_rpm_attack(self, df_base, target_id='0x316', num_attacks=50):
        """Injects anomalous engine RPM messages at high frequency (HCRL RPM)."""
        attacks = []
        timestamps = np.sort(np.random.uniform(df_base['timestamp'].min(), df_base['timestamp'].max(), num_attacks))
        for ts in timestamps:
            rpm_bytes = [0x05, 0x27, 0x0F, 0x09, 0x21, 0x21, 0x00, 0x6F]
            data = "".join([f"{b:02X}" for b in rpm_bytes])
            attacks.append({
                'timestamp': round(ts, 6),
                'can_id': target_id,
                'dlc': 8,
                'data': data,
                'label': 'RPM'
            })
        df_attack = pd.DataFrame(attacks)
        df_combined = pd.concat([df_base, df_attack]).sort_values('timestamp').reset_index(drop=True)
        return df_combined

    def generate_complete_dataset(self, duration_sec=30, output_path=None):
        """Generates a complete dataset with Normal + all 4 attack types."""
        print(f"[CAN Simulator] Generating {duration_sec}s base normal CAN traffic...")
        df = self.generator.generate_normal_stream(duration_sec=duration_sec)
        
        print("[CAN Simulator] Injecting INJECTION attacks...")
        df = self.inject_injection_attack(df, num_attacks=int(duration_sec * 5))
        
        print("[CAN Simulator] Injecting SPOOFING attacks...")
        df = self.inject_spoofing_attack(df, num_attacks=int(duration_sec * 5))
        
        print("[CAN Simulator] Injecting DOS flooding bursts...")
        df = self.inject_dos_attack(df, start_time=5.0, duration=1.0, burst_freq_ms=1.0)
        df = self.inject_dos_attack(df, start_time=18.0, duration=1.0, burst_freq_ms=1.0)
        
        print("[CAN Simulator] Injecting REPLAY attacks...")
        df = self.inject_replay_attack(df, num_attacks=int(duration_sec * 4))
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[CAN Simulator] Saved CAN dataset to: {output_path}")
            
        return df

if __name__ == '__main__':
    simulator = CANAttackSimulator()
    df_all = simulator.generate_complete_dataset(duration_sec=10, output_path=config.CAN_RAW_DATA_PATH)
    print("\nCAN Attack dataset distribution:")
    print(df_all['label'].value_counts())
