import json

from database.db import get_db_connection

class EventDAO:
    @staticmethod
    def save_can_event(event_dict):
        EventDAO.save_can_events([event_dict])

    @staticmethod
    def save_can_events(events):
        if not events:
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        event_query = '''
            INSERT INTO can_events (
                timestamp, can_id, dlc, data, prediction, confidence, severity,
                actual_label, outcome, model_name, features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(event_query, [(
            event['timestamp'], event['can_id'], event['dlc'],
            event['data'], event['attack_type'], event['confidence'], event['severity'],
            event.get('actual_label'), event.get('outcome', 'UNVERIFIED'),
            event.get('model_name'),
            json.dumps(event.get('features')) if event.get('features') is not None else None
        ) for event in events])

        alert_query = '''
                INSERT INTO can_alerts (source, timestamp, can_id, attack_type, confidence, severity, status, actual_label, outcome)
                VALUES ('CAN', ?, ?, ?, ?, ?, 'Confirmed true positive', ?, 'TRUE_POSITIVE')
            '''
        alerts = [event for event in events if event.get('outcome') == 'TRUE_POSITIVE']
        cursor.executemany(alert_query, [(
            event['timestamp'], event['can_id'], event['attack_type'],
            event['confidence'], event['severity'], event.get('actual_label')
        ) for event in alerts])
        conn.commit()
        conn.close()

    @staticmethod
    def save_v2x_event(event_dict):
        EventDAO.save_v2x_events([event_dict])

    @staticmethod
    def save_v2x_events(events):
        if not events:
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        event_query = '''
            INSERT INTO v2x_events
            (timestamp, message_type, sender_id, receiver_id, latitude, longitude, speed, heading, acceleration, sequence_number, prediction, confidence, severity, actual_label, outcome, model_name, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(event_query, [(
            event['timestamp'], event['message_type'], event['sender_id'], event['receiver_id'],
            event['latitude'], event['longitude'], event['speed'], event['heading'],
            event['acceleration'], event['sequence_number'], event['attack_type'],
            event['confidence'], event['severity'], event.get('actual_label'),
            event.get('outcome', 'UNVERIFIED'), event.get('model_name'),
            json.dumps(event.get('features')) if event.get('features') is not None else None
        ) for event in events])

        alert_query = '''
                INSERT INTO v2x_alerts (source, timestamp, sender_id, attack_type, confidence, severity, status, actual_label, outcome)
                VALUES ('V2X', ?, ?, ?, ?, ?, 'Confirmed true positive', ?, 'TRUE_POSITIVE')
            '''
        alerts = [event for event in events if event.get('outcome') == 'TRUE_POSITIVE']
        cursor.executemany(alert_query, [(
            event['timestamp'], event['sender_id'], event['attack_type'],
            event['confidence'], event['severity'], event.get('actual_label')
        ) for event in alerts])
        conn.commit()
        conn.close()

    @staticmethod
    def save_veremi_events(events):
        """Persist VeReMi predictions while retaining the raw feature vectors.

        VeReMi records are vector-based and do not have the legacy V2X
        message fields.  Store the first vector components in the dashboard's
        common columns and keep the complete prediction in features_json.
        """
        if not events:
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        event_query = '''
            INSERT INTO v2x_events
            (timestamp, message_type, sender_id, receiver_id, latitude, longitude, speed, heading, acceleration, sequence_number, prediction, confidence, severity, actual_label, outcome, model_name, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.executemany(event_query, [(
            event['timestamp'], 'VeReMi', None, event.get('receiver_id'),
            event.get('pos_0'), event.get('pos_1'), event.get('speed_0'),
            event.get('speed_1'), event.get('acceleration_0'), None,
            event['attack_type'], event['confidence'], event['severity'],
            event.get('actual_label'), event.get('outcome', 'UNVERIFIED'),
            'VeReMi XGBoost', json.dumps(event)
        ) for event in events])

        alert_query = '''
            INSERT INTO v2x_alerts (source, timestamp, sender_id, attack_type, confidence, severity, status, actual_label, outcome)
            VALUES ('V2X/VeReMi', ?, ?, ?, ?, ?, 'Confirmed true positive', ?, 'TRUE_POSITIVE')
        '''
        alerts = [event for event in events if event.get('outcome') == 'TRUE_POSITIVE']
        cursor.executemany(alert_query, [(
            event['timestamp'], event.get('receiver_id'), event['attack_type'],
            event['confidence'], event['severity'], event.get('actual_label')
        ) for event in alerts])
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
