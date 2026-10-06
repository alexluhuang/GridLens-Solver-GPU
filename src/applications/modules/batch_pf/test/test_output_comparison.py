#!/usr/bin/env python3
"""A different extreme label is allowed only when both tables prove a tie."""

import csv
from pathlib import Path
import tempfile
import unittest
import sys

from run_ca_test import compare_table, equivalent_voltage_tie


class TableComparisonTests(unittest.TestCase):
    def compare(self, left, right):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            for target, data in ((a, left), (b, right)):
                target.mkdir()
                with (target / "table.csv").open("w") as stream:
                    csv.writer(stream).writerows([["event_idx", "bus", "voltage"], *data])
            errors = []
            compare_table(a, b, "table.csv", 2, 1e-3, errors)
            return errors

    def test_equal_rows_and_small_rounding_pass(self):
        rows = [[1, 2, 1.0], [1, 3, 0.98]]
        self.assertEqual(self.compare(rows, rows), [])
        self.assertEqual(self.compare(rows, [[1, 2, 1.0001], [1, 3, 0.98]]), [])

    def test_changed_voltage_or_element_fails(self):
        self.assertTrue(self.compare([[1, 2, 1.0]], [[1, 2, 1.01]]))
        self.assertTrue(self.compare([[1, 2, 1.0]], [[1, 3, 1.0]]))

    def test_equal_malformed_or_duplicate_rows_fail(self):
        self.assertTrue(self.compare([[1, 2]], [[1, 2]]))
        rows = [[1, 2, 1.0], [1, 2, 1.0]]
        self.assertTrue(self.compare(rows, rows))


class VoltageTieTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.a, self.b = Path(self.directory.name) / "a", Path(self.directory.name) / "b"
        self.a.mkdir()
        self.b.mkdir()
        self.left = dict(contingency="outage", bus_id=1, v_pu=0.4)
        self.right = dict(contingency="outage", bus_id=2, v_pu=0.4)
        self.write(self.a, [(1, 0.4), (2, 0.4)])
        self.write(self.b, [(1, 0.4), (2, 0.4)])

    def write(self, directory, values):
        with (directory / "ca_results_violations.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["contingency", "type", "element", "mva_or_vpu"])
            for bus, value in values:
                writer.writerow(["outage", "voltage", bus, value])

    def tied(self, key="worst_voltage_low"):
        return equivalent_voltage_tie(self.a, self.b, key, self.left, self.right, 1e-3)

    def test_verified_low_and_high_ties_pass(self):
        self.assertTrue(self.tied())
        self.assertTrue(self.tied("worst_voltage_high"))

    def test_missing_candidate_in_either_table_fails(self):
        self.write(self.b, [(1, 0.4)])
        self.assertFalse(self.tied())

    def test_nearby_voltage_outside_physics_tolerance_fails(self):
        self.write(self.b, [(1, 0.4), (2, 0.4001)])
        self.assertFalse(self.tied())

    def test_both_candidates_must_be_the_global_extreme(self):
        self.write(self.a, [(1, 0.4), (2, 0.4), (3, 0.3)])
        self.assertFalse(self.tied())


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    print("No errors detected" if result.wasSuccessful() else "failure detected")
    sys.exit(0 if result.wasSuccessful() else 1)
