import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


ATTACK_FILES = {
    'DoS_dataset.csv': 'DOS',
    'Fuzzy_dataset.csv': 'FUZZY',
    'gear_dataset.csv': 'GEAR',
    'RPM_dataset.csv': 'RPM',
}
NORMAL_LINE = re.compile(
    r'^Timestamp:\s*(\S+)\s+ID:\s*([0-9a-fA-F]+)\s+\S+\s+DLC:\s*(\d+)\s*(.*)$'
)


def normalize_record(timestamp, can_id, dlc, payload, label):
    dlc = int(dlc)
    payload = [byte.strip() for byte in payload]
    if not 0 <= dlc <= 8 or len(payload) != dlc:
        raise ValueError('DLC does not match the number of payload bytes')
    if any(not re.fullmatch(r'[0-9a-fA-F]{2}', byte) for byte in payload):
        raise ValueError('Payload contains a value that is not a byte')
    return [
        float(timestamp),
        f'0x{int(can_id, 16):03X}',
        dlc,
        ''.join(payload).upper(),
        label,
    ]


def add_to_reservoir(reservoirs, seen, row, sample_limit, rng):
    label = row[-1]
    seen[label] += 1
    reservoir = reservoirs[label]
    if len(reservoir) < sample_limit:
        reservoir.append(row)
        return
    replacement = rng.randrange(seen[label])
    if replacement < sample_limit:
        reservoir[replacement] = row


def prepare_dataset(source_dir, output_path, sample_limit=100000, seed=42):
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if sample_limit < 1:
        raise ValueError('sample_limit must be at least 1')

    reservoirs = {label: [] for label in ['NORMAL', *ATTACK_FILES.values()]}
    seen = Counter()
    skipped = Counter()
    rng = random.Random(seed)

    normal_files = sorted((source_dir / 'normal_run_data').rglob('*.txt'))
    if not normal_files:
        raise FileNotFoundError(f'No normal-run .txt files found under {source_dir}')

    for path in normal_files:
        with path.open('r', encoding='utf-8', errors='replace') as source:
            for line in source:
                match = NORMAL_LINE.match(line.strip())
                if not match:
                    skipped['normal'] += 1
                    continue
                timestamp, can_id, dlc, payload = match.groups()
                try:
                    row = normalize_record(timestamp, can_id, dlc, payload.split(), 'NORMAL')
                except ValueError:
                    skipped['normal'] += 1
                    continue
                add_to_reservoir(reservoirs, seen, row, sample_limit, rng)

    for filename, attack_label in ATTACK_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Required HCRL capture not found: {path}')
        with path.open('r', encoding='utf-8', errors='replace', newline='') as source:
            for line in source:
                fields = line.strip().split(',')
                if len(fields) < 4:
                    skipped[attack_label] += 1
                    continue
                source_label = fields[-1].strip().upper()
                if source_label == 'R':
                    skipped['redundant_normal'] += 1
                    continue
                if source_label != 'T':
                    skipped[attack_label] += 1
                    continue
                try:
                    row = normalize_record(
                        fields[0], fields[1], fields[2], fields[3:-1], attack_label
                    )
                except ValueError:
                    skipped[attack_label] += 1
                    continue
                add_to_reservoir(reservoirs, seen, row, sample_limit, rng)

    missing_classes = [label for label, count in seen.items() if count == 0]
    if missing_classes or any(not reservoirs[label] for label in reservoirs):
        raise ValueError(f'No valid training records found for classes: {missing_classes}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['timestamp', 'can_id', 'dlc', 'data', 'label'])
        for label in reservoirs:
            writer.writerows(reservoirs[label])

    print(f'Wrote curated HCRL CAN dataset: {output_path}')
    print('Rows seen by class:')
    for label in reservoirs:
        print(f'  {label}: {seen[label]} available, {len(reservoirs[label])} sampled')
    print(f'Discarded redundant normal attack-capture rows: {skipped["redundant_normal"]}')
    print(f'Skipped malformed/unrecognized rows: {sum(skipped.values()) - skipped["redundant_normal"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a balanced HCRL CAN training CSV.')
    parser.add_argument('--source-dir', required=True, help='Folder containing HCRL CSVs and normal_run_data/')
    parser.add_argument('--output', required=True, help='Destination CSV for curated records')
    parser.add_argument('--samples-per-class', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.output, args.samples_per_class, args.seed)