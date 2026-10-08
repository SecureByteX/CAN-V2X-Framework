import pandas as pd


TIME_COLUMN = 'rcvTime'
TARGET_COLUMN = 'AttackerType'
RECEIVER_COLUMN = 'ReceiverID'
FEATURE_COLUMNS = [
    'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
    'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
    'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
    'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1',
]
REQUIRED_COLUMNS = {TIME_COLUMN, TARGET_COLUMN, RECEIVER_COLUMN, *FEATURE_COLUMNS}


def prepare_veremi_frame(frame):
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'VeReMi data is missing required columns: {", ".join(sorted(missing))}')

    selected = frame[[TIME_COLUMN, TARGET_COLUMN, RECEIVER_COLUMN, *FEATURE_COLUMNS]].copy()
    selected[TIME_COLUMN] = pd.to_numeric(selected[TIME_COLUMN], errors='coerce')
    for column in FEATURE_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors='coerce')
    selected[TARGET_COLUMN] = selected[TARGET_COLUMN].astype('string').str.strip()
    selected = selected.dropna(subset=[TIME_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS])
    selected = selected[selected[TARGET_COLUMN] != ''].reset_index(drop=True)
    return selected


def chronological_split(frame, test_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between 0 and 1')

    train_indices = []
    test_indices = []
    for _, group in frame.groupby(TARGET_COLUMN, sort=True):
        ordered_indices = group.sort_values(TIME_COLUMN, kind='stable').index.tolist()
        split_at = int(len(ordered_indices) * (1 - test_fraction))
        if split_at < 1 or split_at >= len(ordered_indices):
            raise ValueError('Each VeReMi class needs at least two rows for a holdout split')
        train_indices.extend(ordered_indices[:split_at])
        test_indices.extend(ordered_indices[split_at:])

    train = frame.loc[train_indices].sort_index()
    test = frame.loc[test_indices].sort_index()
    return train, test


def get_features_and_labels(frame):
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(str).copy()