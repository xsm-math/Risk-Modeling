"""Compare an isolated run with published aggregates; never rewrite references."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, run_dir, atol=1e-7):
    # Published native float32 challenger aggregates predate round-trip export.
    # Keep a sub-float32-epsilon absolute tolerance and retain every difference.
    checks = []
    for filename, key in [('test_metrics.csv', 'model'),
                          ('validation_metrics.csv', 'model'),
                          ('model_policies.csv', 'model'),
                          ('split_summary.csv', 'split')]:
        def read(folder):
            with (folder / filename).open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = list(reader)
            indexed = {row[key]: row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f'Duplicate {key} in {folder / filename}')
            return columns, indexed
        cols, expected = read(reference)
        actual_cols, actual = read(run_dir)
        if cols != actual_cols or set(expected) != set(actual):
            raise ValueError(f'Schema or row keys differ: {filename}')
        maximum = 0.0
        differences = []
        for name, row in expected.items():
            for column in cols:
                if column == key:
                    continue
                a, b = float(row[column]), float(actual[name][column])
                if not (math.isfinite(a) and math.isfinite(b)):
                    raise ValueError(f'Nonfinite value: {filename}/{name}/{column}')
                delta = abs(a - b)
                maximum = max(maximum, delta)
                if delta:
                    differences.append({'row': name, 'column': column,
                                        'reference': a, 'reproduced': b, 'absolute_error': delta})
                if delta > atol:
                    raise ValueError(f'Reference mismatch: {filename}/{name}/{column}: {a} vs {b}')
        checks.append({'file': filename, 'rows': len(expected), 'max_absolute_error': maximum,
                       'nonzero_differences': differences,
                       'reference_sha256': hashlib.sha256((reference / filename).read_bytes()).hexdigest()})
    expected = json.loads((reference / 'selection.json').read_text(encoding='utf-8'))
    actual = json.loads((run_dir / 'selection.json').read_text(encoding='utf-8'))
    for key in ['model', 'criterion', 'primary_cost_ratio', 'config_sha256']:
        if expected[key] != actual[key]:
            raise ValueError(f'Selection differs: {key}')
    if abs(expected['threshold'] - actual['threshold']) > atol:
        raise ValueError('Selected threshold differs')
    expected_meta = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
    actual_meta = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    for key in ['source_sha256', 'split_manifest_sha256', 'model_features']:
        if expected_meta[key] != actual_meta[key]:
            raise ValueError(f'Data, split or feature provenance differs: {key}')
    return {'checks': 'passed', 'absolute_tolerance': atol, 'selected': actual['model'],
            'source_sha256': actual_meta['source_sha256'],
            'split_manifest_sha256': actual_meta['split_manifest_sha256'],
            'runtime_packages': actual_meta['runtime_packages'],
            'tables': checks,
            'scope': 'Aggregate reference agreement only. Run scripts/audit.py for row-level reconstruction. This is not new external validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'reports/credit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference.resolve(), args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
