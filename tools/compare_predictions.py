#!/usr/bin/env python3
"""Compare a fresh original-Test inference result with one selected submission."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['qa_id', 'prediction']:
            raise ValueError(f'{path}: expected columns qa_id,prediction')
        rows = list(reader)
    if len({r['qa_id'] for r in rows}) != len(rows):
        raise ValueError(f'{path}: duplicate qa_id')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', type=Path)
    parser.add_argument('--submission', choices=['56239668', '56258762'], default='56239668')
    args = parser.parse_args()
    slot = {'56239668': 1, '56258762': 2}[args.submission]
    reference = Path(__file__).resolve().parents[1] / f'reproduction/reference/final_submission_{slot}.csv'
    expected, actual = read(reference), read(args.prediction)
    expected_map = {r['qa_id']: r['prediction'] for r in expected}
    actual_map = {r['qa_id']: r['prediction'] for r in actual}
    missing = sorted(expected_map.keys() - actual_map.keys())
    extra = sorted(actual_map.keys() - expected_map.keys())
    changed = [i for i in expected_map.keys() & actual_map.keys() if expected_map[i] != actual_map[i]]
    same_order = [r['qa_id'] for r in expected] == [r['qa_id'] for r in actual]
    exact = reference.read_bytes() == args.prediction.read_bytes()
    print(json.dumps(dict(submission=args.submission, rows=len(actual),
                         different_answers=len(changed), missing_ids=missing,
                         extra_ids=extra, same_order=same_order, exact_csv_bytes=exact), indent=2))
    raise SystemExit(0 if exact else 1)


if __name__ == '__main__':
    main()
