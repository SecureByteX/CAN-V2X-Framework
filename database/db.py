import os
import sqlite3

import config

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # PostgreSQL is optional for local SQLite development.
    psycopg = None
    dict_row = None


DatabaseError = (sqlite3.Error, psycopg.Error if psycopg else Exception)


def using_postgres():
    return bool(config.DATABASE_URL)


def _adapt_sql(query):
    if not using_postgres():
        return query
    query = query.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'BIGSERIAL PRIMARY KEY')
    query = query.replace("datetime('now', ?)", "(CURRENT_TIMESTAMP + (%s)::interval)")
    return query.replace('?', '%s')


class _Cursor:
    def __init__(self, raw_cursor):
        self._raw = raw_cursor

    def execute(self, query, params=None):
        self._raw.execute(_adapt_sql(query), params or ())
        return self

    def executemany(self, query, params):
        self._raw.executemany(_adapt_sql(query), params)
        return self

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()

    def __iter__(self):
        return iter(self._raw)


class _Connection:
    def __init__(self, raw_connection, postgres):
        self._raw = raw_connection
        self._postgres = postgres

    @property
    def row_factory(self):
        return None if self._postgres else self._raw.row_factory

    @row_factory.setter
    def row_factory(self, value):
        if not self._postgres:
            self._raw.row_factory = value

    def cursor(self):
        if self._postgres:
            return _Cursor(self._raw.cursor(row_factory=dict_row))
        return _Cursor(self._raw.cursor())

    def execute(self, query, params=None):
        return self.cursor().execute(query, params)

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        self._raw.close()


def get_db_connection():
    if using_postgres():
        if psycopg is None:
            raise RuntimeError('DATABASE_URL is set but psycopg is not installed.')
        return _Connection(
            psycopg.connect(config.DATABASE_URL, row_factory=dict_row),
            postgres=True,
        )

    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return _Connection(connection, postgres=False)


def _existing_columns(cursor, table):
    if using_postgres():
        rows = cursor.execute(
            '''SELECT column_name FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = ?''',
            (table,),
        ).fetchall()
        return {row['column_name'] for row in rows}
    return {row['name'] for row in cursor.execute(f'PRAGMA table_info({table})')}


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    table_definitions = [
        '''CREATE TABLE IF NOT EXISTS can_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, can_id TEXT,
            dlc INTEGER, data TEXT, prediction TEXT, confidence REAL,
            severity TEXT, actual_label TEXT, outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT, features_json TEXT)''',
        '''CREATE TABLE IF NOT EXISTS v2x_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            message_type TEXT, sender_id TEXT, receiver_id TEXT,
            latitude REAL, longitude REAL, speed REAL, heading REAL,
            acceleration REAL, sequence_number INTEGER, prediction TEXT,
            confidence REAL, severity TEXT, actual_label TEXT, outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT, features_json TEXT)''',
        '''CREATE TABLE IF NOT EXISTS can_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT DEFAULT 'CAN',
            timestamp TEXT, can_id TEXT, attack_type TEXT, confidence REAL,
            severity TEXT, status TEXT DEFAULT 'Detected', actual_label TEXT,
            outcome TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
        '''CREATE TABLE IF NOT EXISTS v2x_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT DEFAULT 'V2X',
            timestamp TEXT, sender_id TEXT, attack_type TEXT, confidence REAL,
            severity TEXT, status TEXT DEFAULT 'Detected', actual_label TEXT,
            outcome TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
        '''CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subsystem TEXT,
            model_name TEXT, accuracy REAL, precision_val REAL,
            recall_val REAL, f1_score REAL, fpr REAL,
            inference_time_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
    ]
    for definition in table_definitions:
        cursor.execute(definition)

    migrations = {
        'can_events': {'actual_label': 'TEXT', 'outcome': 'TEXT', 'model_name': 'TEXT', 'features_json': 'TEXT'},
        'v2x_events': {'actual_label': 'TEXT', 'outcome': 'TEXT', 'model_name': 'TEXT', 'features_json': 'TEXT'},
        'can_alerts': {'actual_label': 'TEXT', 'outcome': 'TEXT'},
        'v2x_alerts': {'actual_label': 'TEXT', 'outcome': 'TEXT'},
    }
    for table, columns in migrations.items():
        existing = _existing_columns(cursor, table)
        for column, column_type in columns.items():
            if column not in existing:
                cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_can_events_dashboard ON can_events(outcome, created_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_v2x_events_dashboard ON v2x_events(outcome, created_at DESC)')

    for table in ('can_events', 'v2x_events', 'can_alerts', 'v2x_alerts'):
        try:
            cursor.execute(f"UPDATE {table} SET outcome = 'UNVERIFIED' WHERE outcome IS NULL")
        except DatabaseError:
            pass

    conn.commit()
    conn.close()
    backend = 'PostgreSQL' if using_postgres() else 'SQLite'
    print(f'[Database] Initialized {backend} tables successfully.')


if __name__ == '__main__':
    init_db()
