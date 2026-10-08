import numpy as np
import pandas as pd
import time
import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CANDataGenerator:
    """
    Generates realistic CAN bus traffic for normal vehicle operations
    such as engine speed, vehicle speed, brake status, steering angle, etc.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        # Normal CAN ID definitions (Standard 11-bit hex IDs)
        self.can_ids = {
            '0x0C0': {'name': 'Engine_Status', 'dlc': 8, 'cycle_ms': 10},     # 10ms cycle
            '0x0D0': {'name': 'Transmission_Status', 'dlc': 8, 'cycle_ms': 20}, # 20ms cycle
            '0x120': {'name': 'Wheel_Speed', 'dlc': 8, 'cycle_ms': 10},        # 10ms cycle
            '0x200': {'name': 'Brake_Status', 'dlc': 4, 'cycle_ms': 50},        # 50ms cycle
            '0x316': {'name': 'Steering_Angle', 'dlc': 6, 'cycle_ms': 20},      # 20ms cycle
            '0x430': {'name': 'Body_Control', 'dlc': 8, 'cycle_ms': 100},       # 100ms cycle
        }

    def generate_normal_message(self, can_id_hex, current_time):
        meta = self.can_ids[can_id_hex]
        dlc = meta['dlc']
        
        # Synthesize realistic payload bytes according to ECU function
        if can_id_hex == '0x0C0': # Engine RPM (0-7000)
            rpm = int(np.random.normal(2500, 300))
            rpm = max(800, min(6500, rpm))
            b0 = (rpm >> 8) & 0xFF
            b1 = rpm & 0xFF
            bytes_list = [b0, b1, 0x01, 0x20, 0x00, 0x00, 0x00, 0x00]
        elif can_id_hex == '0x120': # Speed (0-180 km/h)
            speed = int(np.random.normal(60, 5) * 100) # scaled
            b0 = (speed >> 8) & 0xFF
            b1 = speed & 0xFF
            bytes_list = [b0, b1, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00]
        elif can_id_hex == '0x316': # Steering angle (-360 to +360)
            angle = int(np.random.normal(0, 15) + 500)
            b0 = (angle >> 8) & 0xFF
            b1 = angle & 0xFF
            bytes_list = [b0, b1, 0x00, 0x00, 0x00, 0x00]
        else:
            bytes_list = list(np.random.randint(0, 256, size=dlc, dtype=int))
            
        data_hex = "".join([f"{b:02X}" for b in bytes_list])
        
        return {
            'timestamp': round(current_time, 6),
            'can_id': can_id_hex,
            'dlc': dlc,
            'data': data_hex,
            'label': 'NORMAL'
        }

    def generate_normal_stream(self, duration_sec=10, start_time=0.0):
        records = []
        current_time = start_time
        
        # Schedule messages based on cycle times
        id_next_time = {cid: start_time + np.random.uniform(0, meta['cycle_ms']/1000.0) 
                        for cid, meta in self.can_ids.items()}
        
        end_time = start_time + duration_sec
        while current_time < end_time:
            # Find next scheduled message
            next_cid = min(id_next_time, key=id_next_time.get)
            current_time = id_next_time[next_cid]
            
            if current_time >= end_time:
                break
                
            # Add small timing jitter (+/- 0.5ms)
            jitter = np.random.normal(0, 0.0005)
            actual_timestamp = max(0.0, current_time + jitter)
            
            record = self.generate_normal_message(next_cid, actual_timestamp)
            records.append(record)
            
            # Update next time slot with cycle time (converted to seconds)
            cycle_sec = self.can_ids[next_cid]['cycle_ms'] / 1000.0
            id_next_time[next_cid] += cycle_sec

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

if __name__ == '__main__':
    gen = CANDataGenerator()
    df_normal = gen.generate_normal_stream(duration_sec=2)
    print("Normal CAN stream preview:")
    print(df_normal.head(10))
