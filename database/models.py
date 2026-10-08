import json

from database.db import get_db_connection

class EventDAO:
    @staticmethod
    def save_can_event(event_dict):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO can_events (
                timestamp, can_id, dlc, data, prediction, confidence, severity,
                actual_label, outcome, model_name, features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_dict['timestamp'], event_dict['can_id'], event_dict['dlc'],
            event_dict['data'], event_dict['attack_type'], event_dict['confidence'], event_dict['severity'],
            event_dict.get('actual_label'), event_dict.get('outcome', 'UNVERIFIED'),
            event_dict.get('model_name'),
            json.dumps(event_dict.get('features')) if event_dict.get('features') is not None else None
        ))
        
        # If attack detected, record in alerts table
        if event_dict.get('outcome') == 'TRUE_POSITIVE':
            cursor.execute('''
                INSERT INTO can_alerts (source, timestamp, can_id, attack_type, confidence, severity, status, actual_label, outcome)
                VALUES ('CAN', ?, ?, ?, ?, ?, 'Confirmed true positive', ?, 'TRUE_POSITIVE')
            ''', (
                event_dict['timestamp'], event_dict['can_id'], event_dict['attack_type'],
                event_dict['confidence'], event_dict['severity'], event_dict.get('actual_label')
            ))
            
        conn.commit()
        conn.close()

    @staticmethod
    def save_v2x_event(event_dict):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO v2x_events
            (timestamp, message_type, sender_id, receiver_id, latitude, longitude, speed, heading, acceleration, sequence_number, prediction, confidence, severity, actual_label, outcome, model_name, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_dict['timestamp'], event_dict['message_type'], event_dict['sender_id'], event_dict['receiver_id'],
            event_dict['latitude'], event_dict['longitude'], event_dict['speed'], event_dict['heading'],
            event_dict['acceleration'], event_dict['sequence_number'], event_dict['attack_type'],
            event_dict['confidence'], event_dict['severity'], event_dict.get('actual_label'),
            event_dict.get('outcome', 'UNVERIFIED'), event_dict.get('model_name'),
            json.dumps(event_dict.get('features')) if event_dict.get('features') is not None else None
        ))
        
        if event_dict.get('outcome') == 'TRUE_POSITIVE':
            cursor.execute('''
                INSERT INTO v2x_alerts (source, timestamp, sender_id, attack_type, confidence, severity, status, actual_label, outcome)
                VALUES ('V2X', ?, ?, ?, ?, ?, 'Confirmed true positive', ?, 'TRUE_POSITIVE')
            ''', (
                event_dict['timestamp'], event_dict['sender_id'], event_dict['attack_type'],
                event_dict['confidence'], event_dict['severity'], event_dict.get('actual_label')
            ))
            
        conn.commit()
        conn.close()

    @staticmethod
    def save_model_performance(subsystem, eval_results_list):
        conn = get_db_connection()
        cursor = conn.cursor()
        for res in eval_results_list:
            cursor.execute('''
                INSERT INTO model_performance (subsystem, model_name, accuracy, precision_val, recall_val, f1_score, fpr, inference_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subsystem, res['model_name'], res['accuracy'], res['precision'],
                res['recall'], res['f1_score'], res['false_positive_rate'], res['inference_time_ms']
            ))
        conn.commit()
        conn.close()
