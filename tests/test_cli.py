"""Command-line workflows using generated mathematical inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class CommandLineWorkflows(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def invoke(self, command, *args, success=True):
        result = subprocess.run(
            [sys.executable, '-m', 'lowradiance.cli', command, *map(str, args)],
            capture_output=True, text=True, timeout=120,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def output(self, name):
        return self.root / (name + '.json')

    def read(self, name):
        return json.loads(self.output(name).read_text())

    def test_help_for_every_command(self):
        for command in ['fit', 'fle', 'reference', 'sparse', 'allocation', 'spatial', 'temporal']:
            with self.subTest(command=command):
                self.assertIn('--output', self.invoke(command, '--help').stdout)

    def test_fit_and_projection(self):
        # Equal endpoint counts within each origin and age define a finite optimum.
        rows = []
        for city in range(12):
            for origin in (0, 1):
                for destination in (0, 1, 2):
                    rows.append(dict(origin=origin, destination=destination,
                                     months=1, death_lower=0, death_upper=1,
                                     age=60 + city, municipality_id=f'city_{city}'))
        frame = pd.DataFrame(rows)
        design = np.column_stack([np.ones(len(frame)), (frame.age - 60) / 10])
        intervals = self.root / 'intervals.csv'
        arrays = self.root / 'design.npz'
        frame.to_csv(intervals, index=False)
        np.savez(arrays, design=design)
        self.invoke('fit', '--intervals', intervals, '--arrays', arrays,
                    '--output', self.output('model'))
        model = self.read('model')
        self.assertTrue(model['converged'])
        self.assertEqual(np.asarray(model['covariance']).shape, (8, 8))
        self.assertEqual(model['n_clusters'], 12)
        projection = self.root / 'projection.npz'
        np.savez(projection, coefficients=model['coefficients'], design=[[1., 0.]])
        self.invoke('fle', '--arrays', projection, '--end-age', 61,
                    '--output', self.output('fle'))
        fle = self.read('fle')
        np.testing.assert_allclose(np.asarray(fle['occupancy']).sum(axis=2), 1., atol=1e-12)
        self.assertLessEqual(np.asarray(fle['functional_years']).sum(), 1.)
        # Invalid inputs must fail before writing an output.
        frame.drop(columns='origin').to_csv(intervals, index=False)
        self.invoke('fit', '--intervals', intervals, '--arrays', arrays,
                    '--output', self.output('bad'), success=False)
        self.assertFalse(self.output('bad').exists())
        frame.to_csv(intervals, index=False)
        for bad_design in [design[:-1], np.full_like(design, np.nan)]:
            np.savez(arrays, design=bad_design)
            self.invoke('fit', '--intervals', intervals, '--arrays', arrays,
                        '--output', self.output('bad'), success=False)
            self.assertFalse(self.output('bad').exists())

    def test_reference_and_sparse(self):
        problem = dict(target_id='target', donor_ids=['target', 'donor'],
                       radiance=[10., 5.], target_radiance=10., services=[[1.], [1.]],
                       target_services=[1.], donor_need_draws=[[1., 1.]] * 8,
                       target_need_draws=[1.] * 8, context_distance=[0., 1.])
        source = self.root / 'problem.json'
        source.write_text(json.dumps(dict(problem=problem,
                          options=dict(objective='radiance', risk_limit=1.))))
        self.invoke('reference', '--input', source, '--output', self.output('reference'))
        self.assertAlmostEqual(self.read('reference')['decision']['achieved_gap'], .5)
        sparse = dict(ids=['target', 'donor'], si=0, radiance=[10., 5.],
                      target_radiance=10., services=[[1.], [1.]], target_services=[1.],
                      distance=[0., 1.], d=[[1., 1.]] * 8, t=[1.] * 8,
                      task='long_30_year', band=0.)
        source.write_text(json.dumps(dict(problem=sparse, arm='FRE_long', k=1, self_cap=1.)))
        self.invoke('sparse', '--input', source, '--output', self.output('sparse'))
        np.testing.assert_allclose(self.read('sparse')['w'], [0., 1.], atol=1e-6)

    def test_allocation_and_output_protection(self):
        source = self.root / 'scores.csv'
        pd.DataFrame(dict(city_id=['a', 'b', 'c'], baseline_mass=[1., 2., 1.],
                          event_mass=[1., 2., 3.], functional=[1., 2., 3.],
                          conventional=[1., 1., 1.])).to_csv(source, index=False)
        args = ['--table', source, '--score-columns', 'functional', 'conventional',
                '--draws', '30', '--output', self.output('allocation')]
        self.invoke('allocation', *args)
        result = self.read('allocation')
        for allocation in result['allocations'].values():
            self.assertAlmostEqual(sum(allocation), 1.)
        before = self.output('allocation').read_bytes()
        self.invoke('allocation', *args, success=False)
        self.assertEqual(before, self.output('allocation').read_bytes())

    def test_spatial_and_temporal(self):
        fields = ['population_total2020', 'population65plus2020',
                  'population65plus2020_coverage', 'population_total2020_coverage',
                  'hospital_drive_minutes2020', 'hospital_drive_minutes2020_coverage',
                  'elder_total_beds_per10002025', 'elder_total_beds_per10002025_coverage',
                  'nearest_bus_stop_m2024', 'tcc_baseline2012_2021',
                  't2m_baseline2012_2021', 'tp_baseline2012_2021', 'sde_baseline2012_2021']
        frame = pd.DataFrame({field: np.ones(8) for field in fields})
        frame['radius_m'] = 1000
        features = self.root / 'cells.csv'
        arrays = self.root / 'cells.npz'
        frame.to_csv(features, index=False)
        radiance = np.tile(np.arange(1., 9.), (14, 1))
        np.savez(arrays, radiance=radiance, coverage=np.ones_like(radiance))
        self.invoke('spatial', '--features', features, '--arrays', arrays,
                    '--output', self.output('spatial'))
        self.assertEqual(self.read('spatial')['matched_pairs'], 2)
        annual = self.root / 'annual.csv'
        weights = self.root / 'weights.csv'
        pd.DataFrame([dict(city_id=city, year=year, radiance=value, valid=True)
                      for year in range(2012, 2026)
                      for city, value in [('target', 10.), ('donor', 5.)]]).to_csv(annual, index=False)
        pd.DataFrame(dict(donor_id=['target', 'donor'], weight=[.5, .5])).to_csv(weights, index=False)
        self.invoke('temporal', '--annual', annual, '--weights', weights,
                    '--target-id', 'target', '--output', self.output('temporal'))
        result = self.read('temporal')
        self.assertTrue(result['summary']['stable'])
        self.assertEqual(len(result['annual']), 14)


if __name__ == '__main__':
    unittest.main()
