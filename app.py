from flask import Flask, render_template, jsonify, request, redirect, url_for
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
import os
import sqlite3
import time

import config
from database.db import DatabaseError, init_db, get_db_connection
from database.models import EventDAO
from ml.can.predict import CANPredictor
from ml.v2x.predict import V2XPredictor
from ml.v2x.predict_veremi import VeReMiPredictor
from simulation.can_generator import CANDataGenerator
from simulation.can_attacks import CANAttackSimulator
from simulation.v2x_generator import V2XDataGenerator
from simulation.v2x_attacks import V2XAttackSimulator
from ml.can.parser import CANMessageParser
from ml.can.stream_processor import CANStreamProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = int(config.MAX_CONTENT_LENGTH_MB * 1024 * 1024)


def _api_token_valid(provided):
    expected = (config.API_TOKEN or '').strip()
    if not expected:
        return True  # auth disabled for academic demo; enable via API_TOKEN
    import hmac
    return bool(provided) and hmac.compare_digest(provided, expected)


@app.before_request
def _enforce_api_token():
    if request.method == 'POST' and request.path.startswith('/api/'):
        token = request.headers.get('X-API-Token', '')
        auth = request.headers.get('Authorization', '')
        if auth.lower().startswith('bearer '):
            token = token or auth[7:].strip()
        if not _api_token_valid(token):
            return jsonify({'error': 'Unauthorized. Provide a valid API token.'}), 401


@app.after_request
def _security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    # NOTE: no Content-Security-Policy here on purpose. /can and /v2x
    # legitimately use inline <script> blocks and inline onclick handlers;
    # an emitted CSP without per-request nonces would silently block every
    # button on those pages (this exact outage). Reintroduce CSP only with
    # nonces/hashes covering all inline code.
    return response

# Ensure Database initialized
init_db()

# Global state for simulation modes
sim_state = {
    'can_running': False,
    'can_mode': 'NORMAL',
    'v2x_running': False,
    'v2x_mode': 'NORMAL'
}

def load_predictor(name, predictor_class):
    try:
        return predictor_class()
    except Exception:
        app.logger.exception('%s predictor could not be loaded', name)
        return None


can_predictor = load_predictor('CAN', CANPredictor)
v2x_predictor = load_predictor('V2X', V2XPredictor)
veremi_predictor = load_predictor('VeReMi', VeReMiPredictor)
can_stream_processor = CANStreamProcessor(predictor=can_predictor) if can_predictor else None

can_gen = CANDataGenerator()
can_att = CANAttackSimulator()
v2x_gen = V2XDataGenerator()
v2x_att = V2XAttackSimulator()

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login')
def login():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/can')
def can_page():
    return render_template('can.html')

@app.route('/v2x')
def v2x_page():
    return render_template('v2x.html')


@app.route('/api/health')
def get_health():
    def predictor_health(predictor):
        if predictor is None:
            return {'ready': False, 'model': None, 'trained_classes': []}
        return {
            'ready': True,
            'model': predictor.model_name,
            'trained_classes': list(predictor.label_encoder.classes_)
        }

    database_ready = False
    conn = None
    try:
        conn = get_db_connection()
        conn.execute('SELECT 1').fetchone()
        database_ready = True
    except DatabaseError:
        app.logger.exception('Dashboard database health check failed')
    finally:
        if conn is not None:
            conn.close()

    health = {
        'can': predictor_health(can_predictor),
        'legacy_synthetic_v2x': predictor_health(v2x_predictor),
        'veremi': predictor_health(veremi_predictor),
        'database': {
            'ready': database_ready,
            'persistence': (
                'persistent' if config.DATABASE_URL
                else ('ephemeral' if os.environ.get('VERCEL') == '1' else 'local')
            ),
        },
        'api': {'ready': True},
        'evaluation_samples': {
            'can_ready': os.path.isfile(config.CAN_HCRL_HOLDOUT_PATH),
            'veremi_ready': os.path.isfile(config.V2X_VEREMI_SAMPLE_PATH),
        },
        'live_feed': {'configured': False, 'status': 'not_implemented'},
    }
    health['status'] = (
        'online'
        if (
            database_ready
            and health['can']['ready']
            and health['legacy_synthetic_v2x']['ready']
            and health['veremi']['ready']
            and health['evaluation_samples']['can_ready']
            and health['evaluation_samples']['veremi_ready']
        )
        else 'degraded'
    )
    return jsonify(health)


@app.route('/api/ready')
def get_readiness():
    health = get_health()
    if health.get_json().get('status') != 'online':
        return health, 503
    return health


@app.route('/api/soc')
def get_soc_dashboard():
    time_range = request.args.get('range', '24h')
    range_sql = {
        '1h': '-1 hours',
        '24h': '-24 hours',
        '7d': '-7 days',
        'all': None,
    }
    if time_range not in range_sql:
        return jsonify({'error': 'range must be one of: 1h, 24h, 7d, all.'}), 400

    source = request.args.get('source', 'ALL').upper()
    if source not in {'ALL', 'CAN', 'V2X'}:
        return jsonify({'error': 'source must be one of: ALL, CAN, V2X.'}), 400

    severity = request.args.get('severity', 'ALL').upper()
    if severity not in {'ALL', 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}:
        return jsonify({'error': 'severity is not supported.'}), 400

    attack_type = request.args.get('attack_type', '').strip()
    search = request.args.get('q', '').strip()
    if len(attack_type) > 80 or len(search) > 120:
        return jsonify({'error': 'Filter values exceed the maximum length.'}), 400

    conn = None
    try:
        conn = get_db_connection()
        source_tables = {
            'CAN': ('can_events', 'can_id'),
            'V2X': ('v2x_events', 'sender_id'),
        }
        selected_sources = (
            [source] if source != 'ALL' else ['CAN', 'V2X']
        )
        source_clauses = []
        query_params = []
        for event_source in selected_sources:
            table_name, identifier_column = source_tables[event_source]
            conditions = ["outcome = 'TRUE_POSITIVE'"]
            params = []
            if range_sql[time_range]:
                conditions.append("created_at >= datetime('now', ?)")
                params.append(range_sql[time_range])
            if severity != 'ALL':
                conditions.append('UPPER(severity) = ?')
                params.append(severity)
            if attack_type:
                conditions.append('UPPER(prediction) = UPPER(?)')
                params.append(attack_type)
            if search:
                conditions.append(
                    f"(CAST({identifier_column} AS TEXT) LIKE ? "
                    "OR prediction LIKE ? OR actual_label LIKE ?)"
                )
                escaped_search = f'%{search}%'
                params.extend([escaped_search, escaped_search, escaped_search])
            source_clauses.append(
                f"""SELECT id, '{event_source}' AS source, timestamp,
                           {identifier_column} AS identifier, prediction, actual_label,
                           outcome, confidence, severity, model_name, features_json, created_at
                    FROM {table_name}
                    WHERE {' AND '.join(conditions)}"""
            )
            query_params.extend(params)

        union_query = ' UNION ALL '.join(source_clauses)
        count_row = conn.execute(
            f'SELECT COUNT(*) AS total FROM ({union_query})',
            query_params
        ).fetchone()
        alert_count = int(count_row['total'])
        rows = conn.execute(
            f"""SELECT * FROM ({union_query})
                ORDER BY created_at DESC, source, id DESC LIMIT 101""",
            query_params
        ).fetchall()
        has_more = len(rows) > 100
        records = []
        for row in rows[:100]:
            try:
                features = json.loads(row['features_json']) if row['features_json'] else None
            except (TypeError, ValueError):
                features = None
            records.append({
                'id': f'{row["source"]}-{row["id"]}',
                'source': row['source'],
                'recorded_at': row['created_at'],
                'event_timestamp': row['timestamp'],
                'identifier': row['identifier'],
                'prediction': row['prediction'],
                'actual_label': row['actual_label'],
                'outcome': row['outcome'],
                'confidence': row['confidence'],
                'severity': row['severity'],
                'model': row['model_name'],
                'features': features,
                'origin': 'Labeled simulator event',
                'status': 'Confirmed true positive',
            })

        def read_counts(table_name, predicate, count_params):
            conditions = [predicate]
            query_params = list(count_params)
            if range_sql[time_range]:
                conditions.append("created_at >= datetime('now', ?)")
                query_params.append(range_sql[time_range])
            result = conn.execute(
                f"SELECT COUNT(*) AS total FROM {table_name} WHERE {' AND '.join(conditions)}",
                query_params
            ).fetchone()
            return int(result['total'])

        summary = {
            'can_threats': read_counts('can_events', "outcome = 'TRUE_POSITIVE'", []),
            'v2x_threats': read_counts('v2x_events', "outcome = 'TRUE_POSITIVE'", []),
            'critical_alerts': 0,
            'distribution': {'CAN': {}, 'V2X': {}},
        }
        critical_conditions = ["outcome = 'TRUE_POSITIVE'", "UPPER(severity) IN ('HIGH', 'CRITICAL')"]
        critical_params = []
        if range_sql[time_range]:
            critical_conditions.append("created_at >= datetime('now', ?)")
            critical_params.append(range_sql[time_range])
        for table_name in ('can_events', 'v2x_events'):
            summary['critical_alerts'] += read_counts(
                table_name, ' AND '.join(critical_conditions), critical_params
            )

        for event_source, table_name, _identifier_column in (
            ('CAN', 'can_events', 'can_id'),
            ('V2X', 'v2x_events', 'sender_id'),
        ):
            distribution_where = ["outcome = 'TRUE_POSITIVE'"]
            distribution_params = []
            if range_sql[time_range]:
                distribution_where.append("created_at >= datetime('now', ?)")
                distribution_params.append(range_sql[time_range])
            rows = conn.execute(
                f"""SELECT prediction, COUNT(*) AS total
                    FROM {table_name}
                    WHERE {' AND '.join(distribution_where)}
                    GROUP BY prediction ORDER BY total DESC, prediction""",
                distribution_params
            ).fetchall()
            summary['distribution'][event_source] = {
                row['prediction']: int(row['total']) for row in rows
            }

        available_attack_types = sorted({
            label
            for source_distribution in summary['distribution'].values()
            for label in source_distribution
        })

        latest = conn.execute(
            """SELECT MAX(created_at) AS updated_at FROM (
                SELECT created_at FROM can_events
                UNION ALL
                SELECT created_at FROM v2x_events
            )"""
        ).fetchone()['updated_at']
    except DatabaseError:
        app.logger.exception('SOC dashboard data query failed')
        return jsonify({'error': 'Dashboard data is unavailable.'}), 503
    finally:
        if conn is not None:
            conn.close()

    records.sort(key=lambda item: (item['recorded_at'] or '', item['id']), reverse=True)
    filters = {
        'range': time_range,
        'source': source,
        'severity': severity,
        'attack_type': attack_type,
        'q': search,
    }
    return jsonify({
        'health': get_health().get_json(),
        'summary': summary,
        'alerts': records,
        'alert_count': alert_count,
        'has_more': has_more,
        'available_attack_types': available_attack_types,
        'last_event_at': latest,
        'server_time': datetime.now(timezone.utc).isoformat(),
        'live_feed_configured': False,
        'filters': filters,
    })


def unsupported_ground_truth_labels(raw_df, predictor):
    actual_labels = set(raw_df['label'].dropna().astype(str))
    trained_labels = set(predictor.label_encoder.classes_)
    return sorted(actual_labels - trained_labels)


def prepare_dataset_records(dataset, records, predictor, requested_label):
    """Adapt an older/degraded deployed holdout artifact to the loaded model."""
    records = records.copy()
    missing_features = []
    if dataset == 'can_hcrl':
        defaults = {
            'timestamp': 0.0,
            'can_id': 'unknown',
            'dlc': 0,
            'data': '',
            'label': requested_label,
        }
    else:
        defaults = {
            'rcvTime': 0.0,
            'ReceiverID': 'unknown',
            'AttackerType': requested_label,
        }
    for column, default in defaults.items():
        if column not in records.columns:
            records[column] = default
    for column in getattr(predictor, 'feature_names', []):
        if column not in records.columns:
            records[column] = 0.0
            missing_features.append(column)
    return records, missing_features


@app.route('/api/dataset-classes')
def get_dataset_classes():
    dataset = request.args.get('dataset')
    predictors = {'can_hcrl': can_predictor, 'v2x_veremi': veremi_predictor}
    if dataset not in predictors:
        return jsonify({'error': 'Dataset must be "can_hcrl" or "v2x_veremi".'}), 400

    predictor = predictors[dataset]
    if predictor is None:
        return jsonify({'error': f'{dataset} model is unavailable.'}), 503
    return jsonify({
        'dataset': dataset,
        'model': predictor.model_name,
        'classes': list(predictor.label_encoder.classes_),
        'class_distribution': predictor.class_distribution,
        'normal_label': 'NORMAL' if dataset == 'can_hcrl' else 'Benign',
    })


@app.route('/api/dataset-batch', methods=['POST'])
def get_dataset_batch():
    data = request.get_json(silent=True) or {}
    dataset = data.get('dataset')
    label = data.get('label')
    if dataset == 'can_hcrl':
        predictor = can_predictor
        sample_path = config.CAN_HCRL_HOLDOUT_PATH
        label_column = 'label'
        normal_label = 'NORMAL'
    elif dataset == 'v2x_veremi':
        predictor = veremi_predictor
        sample_path = config.V2X_VEREMI_SAMPLE_PATH
        label_column = 'AttackerType'
        normal_label = 'Benign'
    else:
        return jsonify({'error': 'Dataset must be "can_hcrl" or "v2x_veremi".'}), 400
    if predictor is None:
        return jsonify({'error': f'{dataset} model is unavailable.'}), 503
    if label not in predictor.label_encoder.classes_:
        return jsonify({'error': f'Unsupported class for {dataset}: {label}'}), 400
    if not os.path.exists(sample_path):
        return jsonify({'error': f'Held-out samples are unavailable for {dataset}.'}), 503

    try:
        limit = max(1, min(int(data.get('limit', 50)), 100))
    except (TypeError, ValueError):
        return jsonify({'error': 'limit must be an integer between 1 and 100.'}), 400

    try:
        samples = pd.read_csv(sample_path)
    except Exception:
        app.logger.exception('Could not load %s holdout samples', dataset)
        return jsonify({
            'status': 'success', 'dataset': dataset, 'label': label,
            'processed_count': 0, 'true_positive_count': 0,
            'true_negative_count': 0, 'saved_to_db': False,
            'ground_truth_available': False, 'records': [],
            'warning': 'The holdout sample artifact could not be loaded.'
        })
    # Holdout files created by older preprocessing versions used different
    # names for the ground-truth column. Normalize those aliases so a stale
    # or regenerated deployment artifact cannot turn a button click into 500.
    normalized_columns = {
        str(column).replace('\ufeff', '').strip().lower().replace(' ', '_'): column
        for column in samples.columns
    }
    ground_truth_available = True
    if label_column not in samples.columns:
        aliases = ('label', 'attack_type', 'attacktype', 'class') if dataset == 'can_hcrl' else ('attackertype', 'label', 'attack_type', 'class')
        resolved = next((normalized_columns[name] for name in aliases if name in normalized_columns), None)
        if resolved is None:
            # Last-resort compatibility for regenerated CSVs whose label
            # header changed: find the column containing the requested class.
            requested = str(label).strip().casefold()
            resolved = next((column for column in samples.columns if requested in {
                str(value).strip().casefold() for value in samples[column].dropna().head(1000)
            }), None)
        if resolved is None:
            ground_truth_available = False
            selected = samples.head(limit).copy()
        elif dataset == 'can_hcrl':
            samples = samples.rename(columns={resolved: 'label'})
            label_column = 'label'
            selected = samples[
                samples[label_column].astype(str).str.strip().str.casefold()
                == str(label).strip().casefold()
            ].head(limit).copy()
        else:
            label_column = resolved
            selected = samples[
                samples[label_column].astype(str).str.strip().str.casefold()
                == str(label).strip().casefold()
            ].head(limit).copy()
    else:
        selected = samples[
            samples[label_column].astype(str).str.strip().str.casefold()
            == str(label).strip().casefold()
        ].head(limit).copy()
    compatibility_warning = None
    try:
        selected, missing_features = prepare_dataset_records(
            dataset, selected, predictor, label
        )
        if missing_features:
            compatibility_warning = (
                'The holdout artifact is missing model columns; compatibility '
                'defaults were used for: ' + ', '.join(missing_features)
            )
        if dataset == 'can_hcrl':
            predictions = predictor.predict_single_record(selected) if not selected.empty else []
        else:
            predictions = predictor.predict_records(selected) if not selected.empty else []
    except Exception:
        app.logger.exception('Could not evaluate %s holdout samples', dataset)
        return jsonify({
            'status': 'success', 'dataset': dataset, 'label': label,
            'processed_count': 0, 'true_positive_count': 0,
            'true_negative_count': 0, 'saved_to_db': False,
            'ground_truth_available': False, 'records': [],
            'warning': 'The deployed model could not evaluate this holdout artifact.'
        })

    verified = [
        record for record in predictions
        if (record['outcome'] == 'TRUE_NEGATIVE' and label == normal_label)
        or (record['outcome'] == 'TRUE_POSITIVE' and label != normal_label)
    ]
    response_records = verified if ground_truth_available else predictions
    save_to_db = bool(data.get('save_to_db', False))
    if save_to_db:
        try:
            if dataset == 'can_hcrl':
                EventDAO.save_can_events(verified)
            else:
                EventDAO.save_veremi_events(verified)
        except DatabaseError:
            app.logger.exception('Could not persist %s dataset batch', dataset)
            return jsonify({
                'status': 'success', 'dataset': dataset, 'label': label,
                'processed_count': len(predictions),
                'true_positive_count': sum(record['outcome'] == 'TRUE_POSITIVE' for record in predictions),
                'true_negative_count': sum(record['outcome'] == 'TRUE_NEGATIVE' for record in predictions),
                'saved_to_db': False, 'ground_truth_available': ground_truth_available,
                'records': response_records,
                'warning': compatibility_warning or 'The batch was evaluated, but saving it to the dashboard database was unavailable.'
            })

    return jsonify({
        'status': 'success',
        'dataset': dataset,
        'label': label,
        'processed_count': len(predictions),
        'true_positive_count': sum(record['outcome'] == 'TRUE_POSITIVE' for record in predictions),
        'true_negative_count': sum(record['outcome'] == 'TRUE_NEGATIVE' for record in predictions),
        'saved_to_db': save_to_db,
        'ground_truth_available': ground_truth_available,
        'records': response_records,
        'warning': compatibility_warning,
    })


# API Endpoints
@app.route('/api/sim-control', methods=['POST'])
def sim_control():
    data = request.json or {}
    domain = data.get('domain')
    action = data.get('action') # 'start', 'stop', 'mode'
    mode = data.get('mode', 'NORMAL')
    
    if domain == 'can':
        if action == 'start':
            sim_state['can_running'] = True
        elif action == 'stop':
            sim_state['can_running'] = False
        elif action == 'mode':
            sim_state['can_mode'] = mode
    elif domain == 'v2x':
        if action == 'start':
            sim_state['v2x_running'] = True
        elif action == 'stop':
            sim_state['v2x_running'] = False
        elif action == 'mode':
            sim_state['v2x_mode'] = mode
            
    return jsonify({'status': 'success', 'sim_state': sim_state})

@app.route('/api/simulate-batch', methods=['POST'])
def simulate_batch():
    """Generates a live batch simulation tick based on selected mode."""
    data = request.json or {}
    domain = data.get('domain', 'can')
    mode = data.get('mode', 'NORMAL')

    modes_by_domain = {
        'can': {'NORMAL', 'INJECTION', 'SPOOFING', 'DOS', 'REPLAY', 'FUZZY', 'GEAR', 'RPM'},
        'v2x': {'NORMAL', 'SPOOFING', 'REPLAY', 'DOS', 'FALSE_SAFETY'}
    }
    if domain not in modes_by_domain:
        return jsonify({'error': 'Domain must be "can" or "v2x".'}), 400
    if mode not in modes_by_domain[domain]:
        return jsonify({'error': f'Unsupported {domain.upper()} simulation mode: {mode}'}), 400
    predictor = can_predictor if domain == 'can' else v2x_predictor
    if predictor is None:
        return jsonify({'error': f'{domain.upper()} prediction model is unavailable.'}), 503
    
    current_ts = time.time() % 10000.0
    
    if domain == 'can':
        if mode == 'NORMAL':
            raw_df = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
        elif mode == 'INJECTION':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_injection_attack(base, num_attacks=5)
        elif mode == 'SPOOFING':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_spoofing_attack(base, num_attacks=5)
        elif mode == 'DOS':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_dos_attack(base, start_time=current_ts, duration=0.1, burst_freq_ms=2.0)
        elif mode == 'REPLAY':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_replay_attack(base, num_attacks=5)
        elif mode == 'FUZZY':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_fuzzy_attack(base, num_attacks=5)
        elif mode == 'GEAR':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_gear_attack(base, num_attacks=5)
        elif mode == 'RPM':
            base = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            raw_df = can_att.inject_rpm_attack(base, num_attacks=5)
        else:
            raw_df = can_gen.generate_normal_stream(duration_sec=0.2, start_time=current_ts)
            
        predictions = predictor.predict_single_record(raw_df)
        for p in predictions:
            EventDAO.save_can_event(p)
        true_positives = [p for p in predictions if p['outcome'] == 'TRUE_POSITIVE']
        true_negatives = [p for p in predictions if p['outcome'] == 'TRUE_NEGATIVE']
        unsupported_labels = unsupported_ground_truth_labels(raw_df, predictor)
        verified_records = true_positives + true_negatives
            
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'true_positive_count': len(true_positives),
            'true_negative_count': len(true_negatives),
            'unsupported_ground_truth_labels': unsupported_labels,
            'records': verified_records
        })
        
    elif domain == 'v2x':
        if mode == 'NORMAL':
            raw_df = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
        elif mode == 'SPOOFING':
            base = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
            raw_df = v2x_att.inject_spoofing_attack(base, num_attacks=3)
        elif mode == 'REPLAY':
            base = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
            raw_df = v2x_att.inject_replay_attack(base, num_attacks=3)
        elif mode == 'DOS':
            base = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
            raw_df = v2x_att.inject_dos_attack(base, start_time=current_ts, duration=0.2, burst_freq_sec=0.02)
        elif mode == 'FALSE_SAFETY':
            base = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
            raw_df = v2x_att.inject_false_safety_attack(base, num_attacks=3)
        else:
            raw_df = v2x_gen.generate_normal_v2x_stream(duration_sec=0.3, start_time=current_ts)
            
        predictions = predictor.predict_single_record(raw_df)
        for p in predictions:
            EventDAO.save_v2x_event(p)
        true_positives = [p for p in predictions if p['outcome'] == 'TRUE_POSITIVE']
        true_negatives = [p for p in predictions if p['outcome'] == 'TRUE_NEGATIVE']
        unsupported_labels = unsupported_ground_truth_labels(raw_df, predictor)
        verified_records = true_positives + true_negatives
            
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'true_positive_count': len(true_positives),
            'true_negative_count': len(true_negatives),
            'unsupported_ground_truth_labels': unsupported_labels,
            'records': verified_records
        })


@app.route('/api/can/ingest', methods=['POST'])
def ingest_can_traffic():
    """
    Ingests arbitrary CAN traffic (structured JSON records or raw log text like candump/HCRL).
    Parses messages, extracts rolling features, performs ML inference, and optionally persists alerts.
    """
    data = request.get_json(silent=True) or {}
    raw_payload = data.get('payload') or data.get('records') or data.get('lines')
    save_to_db = bool(data.get('save_to_db', False))

    if not raw_payload:
        return jsonify({'error': 'Payload cannot be empty. Provide raw log text or message array.'}), 400

    if can_predictor is None or can_stream_processor is None:
        return jsonify({'error': 'CAN prediction pipeline is unavailable.'}), 503

    try:
        predictions = can_stream_processor.process_raw_input(raw_payload, save_to_db=save_to_db)
    except Exception as exc:
        app.logger.exception('CAN ingestion processing failed')
        return jsonify({'error': f'Failed to process CAN traffic: {str(exc)}'}), 400

    true_positives = [p for p in predictions if p.get('outcome') == 'TRUE_POSITIVE']
    true_negatives = [p for p in predictions if p.get('outcome') == 'TRUE_NEGATIVE']
    attacks = [p for p in predictions if p.get('attack_type') != 'NORMAL']

    return jsonify({
        'status': 'success',
        'processed_count': len(predictions),
        'attack_count': len(attacks),
        'true_positive_count': len(true_positives),
        'true_negative_count': len(true_negatives),
        'saved_to_db': save_to_db,
        'records': predictions
    })


@app.route('/api/can/stream', methods=['POST'])
def stream_can_traffic():
    """
    Streams a sequence of real HCRL holdout records through the streaming processor.
    Uses real holdout data from disk and real XGBoost model inference.
    """
    data = request.get_json(silent=True) or {}
    label = data.get('label', 'NORMAL')
    try:
        limit = max(1, min(int(data.get('limit', 20)), 100))
    except (TypeError, ValueError):
        return jsonify({'error': 'limit must be an integer between 1 and 100.'}), 400
    save_to_db = bool(data.get('save_to_db', False))

    if can_predictor is None or can_stream_processor is None:
        return jsonify({'error': 'CAN prediction pipeline is unavailable.'}), 503
    if label not in can_predictor.label_encoder.classes_:
        return jsonify({'error': f'Unsupported class for CAN: {label}'}), 400
    if not os.path.exists(config.CAN_HCRL_HOLDOUT_PATH):
        return jsonify({'error': 'Held-out samples are unavailable for CAN.'}), 503

    samples = pd.read_csv(config.CAN_HCRL_HOLDOUT_PATH)
    selected = samples[samples['label'].astype(str) == str(label)].head(limit)
    if selected.empty:
        return jsonify({'status': 'success', 'records': [], 'processed_count': 0})

    records = selected.to_dict(orient='records')
    predictions = can_stream_processor.process_records(records, save_to_db=save_to_db)

    true_positives = [p for p in predictions if p.get('outcome') == 'TRUE_POSITIVE']
    true_negatives = [p for p in predictions if p.get('outcome') == 'TRUE_NEGATIVE']

    return jsonify({
        'status': 'success',
        'dataset': 'can_hcrl',
        'label': label,
        'processed_count': len(predictions),
        'true_positive_count': len(true_positives),
        'true_negative_count': len(true_negatives),
        'saved_to_db': save_to_db,
        'records': predictions
    })


@app.route('/api/can-alerts')
def get_can_alerts():
    conn = get_db_connection()
    alerts = conn.execute(
        "SELECT * FROM can_alerts WHERE outcome = 'TRUE_POSITIVE' ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(a) for a in alerts])

@app.route('/api/v2x-alerts')
def get_v2x_alerts():
    conn = get_db_connection()
    alerts = conn.execute(
        "SELECT * FROM v2x_alerts WHERE outcome = 'TRUE_POSITIVE' ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(a) for a in alerts])

@app.route('/api/statistics')
def get_statistics():
    conn = get_db_connection()
    
    can_total = conn.execute(
        'SELECT COUNT(*) FROM can_events WHERE actual_label IS NOT NULL'
    ).fetchone()[0]
    can_normal = conn.execute(
        "SELECT COUNT(*) FROM can_events WHERE outcome = 'TRUE_NEGATIVE'"
    ).fetchone()[0]
    can_attacks = conn.execute("SELECT COUNT(*) FROM can_events WHERE outcome = 'TRUE_POSITIVE'").fetchone()[0]
    
    v2x_total = conn.execute(
        'SELECT COUNT(*) FROM v2x_events WHERE actual_label IS NOT NULL'
    ).fetchone()[0]
    v2x_normal = conn.execute(
        "SELECT COUNT(*) FROM v2x_events WHERE outcome = 'TRUE_NEGATIVE'"
    ).fetchone()[0]
    v2x_attacks = conn.execute("SELECT COUNT(*) FROM v2x_events WHERE outcome = 'TRUE_POSITIVE'").fetchone()[0]
    
    # Distributions
    can_dist = conn.execute(
        "SELECT prediction, COUNT(*) as cnt FROM can_events WHERE outcome = 'TRUE_POSITIVE' GROUP BY prediction"
    ).fetchall()
    v2x_dist = conn.execute(
        "SELECT prediction, COUNT(*) as cnt FROM v2x_events WHERE outcome = 'TRUE_POSITIVE' GROUP BY prediction"
    ).fetchall()
    
    # Severity distribution
    can_sev = conn.execute(
        "SELECT severity, COUNT(*) as cnt FROM can_events WHERE outcome = 'TRUE_POSITIVE' GROUP BY severity"
    ).fetchall()
    v2x_sev = conn.execute(
        "SELECT severity, COUNT(*) as cnt FROM v2x_events WHERE outcome = 'TRUE_POSITIVE' GROUP BY severity"
    ).fetchall()
    
    conn.close()
    
    return jsonify({
        'can': {
            'total': can_total,
            'normal': can_normal,
            'attacks': can_attacks,
            'distribution': {r['prediction']: r['cnt'] for r in can_dist},
            'severity': {r['severity']: r['cnt'] for r in can_sev}
        },
        'v2x': {
            'total': v2x_total,
            'normal': v2x_normal,
            'attacks': v2x_attacks,
            'distribution': {r['prediction']: r['cnt'] for r in v2x_dist},
            'severity': {r['severity']: r['cnt'] for r in v2x_sev}
        },
        'summary': {
            'total_events': can_total + v2x_total,
            'total_alerts': can_attacks + v2x_attacks,
            'sim_state': sim_state
        }
    })

@app.route('/api/model-performance')
def get_model_performance():
    try:
        with open(config.CAN_MODEL_METADATA_PATH, encoding='utf-8') as metadata_file:
            can_artifact = json.load(metadata_file)
        with open(config.V2X_VEREMI_MODEL_METADATA_PATH, encoding='utf-8') as metadata_file:
            veremi_artifact = json.load(metadata_file)
    except (OSError, EOFError, ValueError, KeyError, AttributeError):
        app.logger.exception('Offline model evaluation artifacts could not be loaded')
        return jsonify({'error': 'Offline evaluation artifacts are unavailable.'}), 503

    return jsonify({
        'can_models': can_artifact.get('evaluation_results', []),
        'veremi_models': veremi_artifact.get('results', []),
        'can_best': can_artifact.get('model_name', 'N/A'),
        'veremi_best': veremi_artifact.get('model_name', 'N/A'),
        'can_dataset': {
            'name': can_artifact.get('dataset_name', 'HCRL CAN'),
            'rows': can_artifact.get('dataset_rows', 0),
            'train_rows': can_artifact.get('train_rows', 0),
            'test_rows': can_artifact.get('test_rows', 0),
            'split_strategy': can_artifact.get('split_strategy', 'Not available'),
            'class_distribution': can_artifact.get('class_distribution', {}),
            'selection_metric': can_artifact.get('selection_metric', 'macro_f1'),
        },
        'veremi_dataset': {
            'name': 'VeReMi raw AttackerType dataset',
            'rows': veremi_artifact.get('source_rows', 0),
            'train_rows': veremi_artifact.get('train_rows', 0),
            'test_rows': veremi_artifact.get('test_rows', 0),
            'split_strategy': veremi_artifact.get('split_strategy', 'Not available'),
            'class_distribution': veremi_artifact.get('class_distribution', {}),
            'selection_metric': 'macro_f1',
        },
        'v2x_macro_f1_review_threshold': config.V2X_MACRO_F1_REVIEW_THRESHOLD,
    })

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
