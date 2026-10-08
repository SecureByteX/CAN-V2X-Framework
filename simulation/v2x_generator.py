import numpy as np
import pandas as pd
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class V2XDataGenerator:
    """
    Generates realistic synthetic V2X Basic Safety Messages (BSM)
    transmitted via DSRC / C-V2X (IEEE 802.11p / 3GPP LTE-V2X).
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.vehicle_ids = [f"VEHICLE_{i:03d}" for i in range(1, 6)]
        self.rsu_ids = ['RSU_001', 'RSU_002']
        
        # Initial positions around Bengaluru ADAS testbed
        self.base_lat = 12.9716
        self.base_lon = 77.5946

    def generate_normal_v2x_stream(self, duration_sec=10, start_time=0.0):
        records = []
        
        # Frequency: BSMs sent at 10 Hz (every 0.1s)
        time_step = 0.1
        current_time = start_time
        end_time = start_time + duration_sec
        
        # Track state for each vehicle
        states = {}
        for vid in self.vehicle_ids:
            states[vid] = {
                'lat': self.base_lat + np.random.uniform(-0.005, 0.005),
                'lon': self.base_lon + np.random.uniform(-0.005, 0.005),
                'speed': np.random.uniform(40.0, 60.0), # km/h
                'heading': np.random.choice([0.0, 90.0, 180.0, 270.0]), # degrees
                'accel': 0.0,
                'seq': 100
            }
            
        while current_time < end_time:
            for vid in self.vehicle_ids:
                st = states[vid]
                
                # Update kinematics cleanly
                st['accel'] = round(float(np.random.normal(0, 0.2)), 2)
                st['speed'] = max(0.0, min(120.0, st['speed'] + (st['accel'] * time_step * 3.6)))
                
                # Heading slight variance
                st['heading'] = round((st['heading'] + np.random.normal(0, 0.5)) % 360, 2)
                
                # Update Lat/Lon displacement (1 deg lat ~ 111,000m)
                speed_ms = st['speed'] / 3.6
                dist_m = speed_ms * time_step
                rad = np.radians(st['heading'])
                st['lat'] += (dist_m * np.cos(rad)) / 111000.0
                st['lon'] += (dist_m * np.sin(rad)) / (111000.0 * np.cos(np.radians(st['lat'])))
                
                st['seq'] += 1
                
                rsu_id = np.random.choice(self.rsu_ids)
                
                records.append({
                    'timestamp': round(current_time + np.random.normal(0, 0.002), 6),
                    'message_type': 'BSM',
                    'sender_id': vid,
                    'receiver_id': rsu_id,
                    'latitude': round(st['lat'], 6),
                    'longitude': round(st['lon'], 6),
                    'speed': round(st['speed'], 2),
                    'heading': st['heading'],
                    'acceleration': st['accel'],
                    'sequence_number': st['seq'],
                    'label': 'NORMAL'
                })
                
            current_time += time_step

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

if __name__ == '__main__':
    gen = V2XDataGenerator()
    df_v2x = gen.generate_normal_v2x_stream(duration_sec=2)
    print("Normal V2X BSM stream sample:")
    print(df_v2x.head())
