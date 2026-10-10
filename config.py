import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32).hex()

# Optional API token. When set, all POST /api/* endpoints require it via
# the X-API-Token header (or "Authorization: Bearer <token>").
# Empty by default so the academic demo dashboard keeps working out of the box.
API_TOKEN = os.environ.get('API_TOKEN', '').strip()

# Server binding (production must use HOST/PORT env vars, never localhost literals).
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '5000'))
DEBUG = os.environ.get('FLASK_DEBUG', '').lower() == 'true'

# Reject oversized ingestion payloads (10 MB default).
MAX_CONTENT_LENGTH_MB = float(os.environ.get('MAX_CONTENT_LENGTH_MB', '10'))


# Database
DATABASE_PATH = os.environ.get('DATABASE_PATH')
if not DATABASE_PATH:
    if os.environ.get('VERCEL') == '1':
        # Vercel's deployed filesystem is read-only except for /tmp. This is
        # scratch storage only; event history needs an external database to persist.
        DATABASE_PATH = os.path.join('/tmp', 'ids_system.db')
    else:
        DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'ids_system.db')
SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'

# Data Paths
CAN_RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'can', 'raw', 'can_traffic_synthetic.csv')
CAN_PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'can', 'processed', 'can_features.csv')
CAN_TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'data', 'can', 'processed', 'can_hcrl_training.csv')
CAN_HCRL_HOLDOUT_PATH = os.path.join(BASE_DIR, 'data', 'can', 'processed', 'can_hcrl_holdout_samples.csv')

V2X_RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'v2x', 'raw', 'v2x_traffic_synthetic.csv')
V2X_PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'v2x', 'processed', 'v2x_features.csv')
V2X_VEREMI_RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'v2x', 'raw', 'Preprocessed VeReMi for V2.csv')
V2X_VEREMI_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'veremi_model.pkl')
V2X_VEREMI_ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'veremi_model.onnx')
V2X_VEREMI_MODEL_METADATA_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'veremi_model.json')
V2X_VEREMI_SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'v2x', 'processed', 'v2x_veremi_holdout_samples.csv')
V2X_MACRO_F1_REVIEW_THRESHOLD = float(os.environ.get('V2X_MACRO_F1_REVIEW_THRESHOLD', '0.70'))
if not 0.0 <= V2X_MACRO_F1_REVIEW_THRESHOLD <= 1.0:
    raise ValueError('V2X_MACRO_F1_REVIEW_THRESHOLD must be between 0 and 1.')

# Model Paths
CAN_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'can', 'model.pkl')
V2X_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'model.pkl')
CAN_ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'can', 'model.onnx')
CAN_MODEL_METADATA_PATH = os.path.join(BASE_DIR, 'ml', 'can', 'model.json')
V2X_ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'model.onnx')
V2X_MODEL_METADATA_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'model.json')
