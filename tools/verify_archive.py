"""Verify copied historical bytes and data arithmetic; does not execute DGEMM."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1] / 'archive'
SHAPES = {(12349, 140, 3040), (24698, 8, 3040), (256, 3040, 3040), (3040, 16, 12349)}


def rows(name):
    with (ARCHIVE / 'data' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        target = (ARCHIVE / entry['path']).resolve()
        assert ARCHIVE.resolve() in target.parents, 'manifest path escapes archive'
        data = target.read_bytes()
        assert len(data) == entry['bytes'], entry['path']
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
    stages, elapsed = rows('stages.csv'), rows('elapsed-summary.csv')
    assert len(stages) == 16 and len(elapsed) == 4
    assert {(int(r['m']), int(r['n']), int(r['k'])) for r in elapsed} == SHAPES
    recalculated = []
    for case in range(1, 5):
        group = [r for r in stages if int(r['case']) == case]
        assert {r['stage'] for r in group} == {'serial_O1', 'openmp_O3', 'neon_packing', 'repeat4_best'}
        baseline = next(Decimal(r['gflops']) for r in group if r['stage'] == 'serial_O1')
        assert baseline > 0
        for row in group:
            measured = Decimal(row['gflops'])
            assert measured > 0
            computed = measured / baseline
            recalculated.append({'case': case, 'stage': row['stage'],
                'computed_ratio': str(computed.quantize(Decimal('.000001'))),
                'reported_minus_computed': str((Decimal(row['reported_speedup']) - computed)
                                               .quantize(Decimal('.000001')))})
    rounded_time_ratios = []
    for row in elapsed:
        before, after = Decimal(row['baseline_seconds']), Decimal(row['optimized_seconds'])
        assert before > 0 and after > 0
        rounded_time_ratios.append({'case': row['case'],
            'rounded_time_ratio': str((before / after).quantize(Decimal('.000001')))})
    print(json.dumps({'status': 'passed', 'archive_files_verified': len(manifest['files']),
        'stage_rows': len(stages), 'elapsed_rows': len(elapsed),
        'hardware_benchmark_run': False, 'ratios': recalculated,
        'rounded_time_ratios': rounded_time_ratios}, indent=2))


if __name__ == '__main__':
    main()
