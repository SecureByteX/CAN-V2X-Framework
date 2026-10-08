import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32).hex()

# Database
DATABASE_PATH = os.environ.get('DATABASE_PATH') or os.path.join(
    BASE_DIR, 'database', 'ids_system.db'
)
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
V2X_VEREMI_SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'v2x', 'processed', 'v2x_veremi_holdout_samples.csv')
V2X_MACRO_F1_REVIEW_THRESHOLD = float(os.environ.get('V2X_MACRO_F1_REVIEW_THRESHOLD', '0.70'))
if not 0.0 <= V2X_MACRO_F1_REVIEW_THRESHOLD <= 1.0:
    raise ValueError('V2X_MACRO_F1_REVIEW_THRESHOLD must be between 0 and 1.')

# Model Paths
CAN_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'can', 'model.pkl')
V2X_MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'v2x', 'model.pkl')
