"""A changed model result must not be reported as a successful reproduction."""
import csv
import shutil
import pytest
from scripts.compare_reference import ROOT, compare


def test_published_aggregates_agree_with_themselves():
    assert compare(ROOT / 'reports/credit', ROOT / 'reports/credit')['checks'] == 'passed'


def test_changed_metric_is_rejected(tmp_path):
    reference = ROOT / 'reports/credit'
    for name in ['test_metrics.csv', 'validation_metrics.csv', 'model_policies.csv',
                 'split_summary.csv', 'selection.json', 'manifest.json']:
        shutil.copyfile(reference / name, tmp_path / name)
    path = tmp_path / 'test_metrics.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    rows[0]['auc'] = str(float(rows[0]['auc']) + .01)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='Reference mismatch'):
        compare(reference, tmp_path)
