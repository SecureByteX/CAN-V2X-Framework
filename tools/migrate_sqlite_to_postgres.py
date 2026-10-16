"""Copy local SQLite dashboard data into the PostgreSQL DATABASE_URL target.

Usage from PowerShell:
    $env:DATABASE_URL = 'postgresql://...'
    .venv311/Scripts/python.exe tools/migrate_sqlite_to_postgres.py
"""

import os
import sqlite3

import psycopg

import config
from database.db import init_db


TABLES = {
    'can_events': (
        'timestamp', 'can_id', 'dlc', 'data', 'prediction', 'confidence',
        'severity', 'actual_label', 'outcome', 'created_at', 'model_name',
        'features_json',
    ),
    'v2x_events': (
        'timestamp', 'message_type', 'sender_id', 'receiver_id', 'latitude',
        'longitude', 'speed', 'heading', 'acceleration', 'sequence_number',
        'prediction', 'confidence', 'severity', 'actual_label', 'outcome',
        'created_at', 'model_name', 'features_json',
    ),
    'can_alerts': (
        'source', 'timestamp', 'can_id', 'attack_type', 'confidence',
        'severity', 'status', 'actual_label', 'outcome', 'created_at',
    ),
    'v2x_alerts': (
        'source', 'timestamp', 'sender_id', 'attack_type', 'confidence',
        'severity', 'status', 'actual_label', 'outcome', 'created_at',
    ),
    'model_performance': (
        'subsystem', 'model_name', 'accuracy', 'precision_val', 'recall_val',
        'f1_score', 'fpr', 'inference_time_ms', 'created_at',
    ),
}


def main():
    if not config.DATABASE_URL:
        raise SystemExit('Set DATABASE_URL to the PostgreSQL target before running this script.')

    source_path = config.DATABASE_PATH
    if not source_path or not os.path.isfile(source_path):
        raise SystemExit(f'Local SQLite database was not found: {source_path}')

    init_db()
    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    target = psycopg.connect(config.DATABASE_URL)

    try:
        with target.cursor() as cursor:
            for table, columns in TABLES.items():
                rows = source.execute(
                    f"SELECT {', '.join(columns)} FROM {table}"
                ).fetchall()
                if not rows:
                    continue
                placeholders = ', '.join(['%s'] * len(columns))
                query = (
                    f"INSERT INTO {table} ({', '.join(columns)}) "
                    f"VALUES ({placeholders})"
                )
                cursor.executemany(
                    query,
                    [tuple(row[column] for column in columns) for row in rows],
                )
                print(f'{table}: copied {len(rows)} rows')
        target.commit()
    finally:
        source.close()
        target.close()


if __name__ == '__main__':
    main()
