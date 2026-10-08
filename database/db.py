import sqlite3
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_db_connection():
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # CAN Events Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS can_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            can_id TEXT,
            dlc INTEGER,
            data TEXT,
            prediction TEXT,
            confidence REAL,
            severity TEXT,
            actual_label TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # V2X Events Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS v2x_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            message_type TEXT,
            sender_id TEXT,
            receiver_id TEXT,
            latitude REAL,
            longitude REAL,
            speed REAL,
            heading REAL,
            acceleration REAL,
            sequence_number INTEGER,
            prediction TEXT,
            confidence REAL,
            severity TEXT,
            actual_label TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # CAN Alerts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS can_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT DEFAULT 'CAN',
            timestamp TEXT,
            can_id TEXT,
            attack_type TEXT,
            confidence REAL,
            severity TEXT,
            status TEXT DEFAULT 'Detected',
            actual_label TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # V2X Alerts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS v2x_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT DEFAULT 'V2X',
            timestamp TEXT,
            sender_id TEXT,
            attack_type TEXT,
            confidence REAL,
            severity TEXT,
            status TEXT DEFAULT 'Detected',
            actual_label TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Model Performance Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subsystem TEXT,
            model_name TEXT,
            accuracy REAL,
            precision_val REAL,
            recall_val REAL,
            f1_score REAL,
            fpr REAL,
            inference_time_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    migrations = {
        'can_events': {
            'actual_label': 'TEXT', 'outcome': 'TEXT',
            'model_name': 'TEXT', 'features_json': 'TEXT'
        },
        'v2x_events': {
            'actual_label': 'TEXT', 'outcome': 'TEXT',
            'model_name': 'TEXT', 'features_json': 'TEXT'
        },
        'can_alerts': {'actual_label': 'TEXT', 'outcome': 'TEXT'},
        'v2x_alerts': {'actual_label': 'TEXT', 'outcome': 'TEXT'},
    }
    for table, columns in migrations.items():
        existing = {row['name'] for row in cursor.execute(f'PRAGMA table_info({table})')}
        for column, column_type in columns.items():
            if column not in existing:
                cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}')

    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_can_events_dashboard ON can_events(outcome, created_at DESC)'
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_v2x_events_dashboard ON v2x_events(outcome, created_at DESC)'
    )
    
    # One-time honest backfill: rows written by older builds may have a NULL
    # outcome. They are historical records of unknown verification state, so
    # mark them UNVERIFIED rather than guessing TRUE/FALSE.
    for table in ('can_events', 'v2x_events', 'can_alerts', 'v2x_alerts'):
        try:
            cursor.execute(
                f"UPDATE {table} SET outcome = 'UNVERIFIED' WHERE outcome IS NULL"
            )
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    print("[Database] Initialized SQLite tables successfully.")

if __name__ == '__main__':
    init_db()
