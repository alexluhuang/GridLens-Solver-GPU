#!/usr/bin/env python3
"""A different extreme label is allowed only when both tables prove a tie."""

import csv
from pathlib import Path
import tempfile
import unittest
import sys

from run_ca_test import (compare_table, equivalent_voltage_case_tie, equivalent_voltage_tie,
                         reported_state)


class ReportedStateTests(unittest.TestCase):
    def validate(self, outcome=None, convergence=None, shadow=None):
        conv = dict(event_idx="1", status_code="OK", iterations="3", final_tolerance="1e-7")
        report = dict(event_idx="1", path="cpu", health_events="0", reported_status="OK",
                      reported_iterations="3", reported_tolerance="1e-7",
                      final_pv_buses="2", final_pq_buses="10")
        conv.update(convergence or {})
        report.update(outcome or {})
        with tempfile.TemporaryDirectory() as directory:
            tables = {"convergence": conv, "gpu_outcomes": report}
            if shadow:
                tables["gpu_shadow"] = dict(event_idx="1", **shadow)
            for name, row in tables.items():
                with (Path(directory) / ("ca_results_"+name+".csv")).open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                    writer.writerow(row)
            errors = []
            reported_state(directory, errors)
            return errors

    def test_cpu_record_and_shadow_counts_pass(self):
        self.assertEqual(self.validate(shadow=dict(pv_buses_cpu="2", pv_buses_gpu="2",
                                                  pq_buses_cpu="10", pq_buses_gpu="10")), [])

    def test_unsolved_case_does_not_inherit_previous_iterations(self):
        conv = dict(status_code="NO_SLACK", iterations="9")
        outcome = dict(reported_status="NO_SLACK", reported_iterations="", reported_tolerance="")
        self.assertEqual(self.validate(outcome, conv), [])
        outcome["reported_iterations"] = "9"
        self.assertTrue(self.validate(outcome, conv))

    def test_missing_or_changed_solved_record_fails(self):
        self.assertTrue(self.validate(dict(reported_iterations="")))
        self.assertTrue(self.validate(dict(reported_iterations="4")))
        self.assertTrue(self.validate(dict(reported_status="DIVERGED")))

    def test_cleanup_bus_counts_cannot_replace_solved_counts(self):
        self.assertTrue(self.validate(shadow=dict(pv_buses_cpu="1", pv_buses_gpu="1",
                                                 pq_buses_cpu="11", pq_buses_gpu="11")))

    def test_tolerance_uses_published_precision(self):
        conv = dict(final_tolerance="1.6033e-07")
        self.assertEqual(self.validate(dict(reported_tolerance="1.603344e-07"), conv), [])
        self.assertTrue(self.validate(dict(reported_tolerance="1.603400e-07"), conv))

    def test_numerical_failure_requires_health_failures_on_both_paths(self):
        outcome = dict(path="cpu_fallback", health_events="1", reported_status="NUMERICAL_FAILURE")
        conv = dict(status_code="DIVERGED")
        self.assertTrue(self.validate(outcome, conv))  # Ordinary divergence is not a health failure.
        outcome["reported_tolerance"] = conv["final_tolerance"] = "nan"
        self.assertEqual(self.validate(outcome, conv), [])
        outcome["health_events"] = "0"
        self.assertTrue(self.validate(outcome, conv))


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

    def test_one_printed_digit_on_a_rounding_boundary_passes(self):
        self.assertEqual(self.compare([[1, 2, "5.99"]], [[1, 2, "6.00"]]), [])
        self.assertTrue(self.compare([[1, 2, "5.99"]], [[1, 2, "6.01"]]))
        self.assertTrue(self.compare([[1, 2, "1.0"]], [[1, 2, "1.01"]]))

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


class VoltageCaseTieTests(unittest.TestCase):
    """Two outages leaving the same bus at the same extreme voltage"""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.a, self.b = Path(self.directory.name) / "a", Path(self.directory.name) / "b"
        self.a.mkdir()
        self.b.mkdir()
        self.left = dict(contingency="branch", bus_id=7, v_pu=0.7)
        self.right = dict(contingency="generator", bus_id=7, v_pu=0.7)
        for directory in (self.a, self.b):
            self.write(directory, [("branch", 0.7), ("generator", 0.7)])

    def write(self, directory, values):
        with (directory / "ca_results_violations.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["contingency", "type", "element", "mva_or_vpu"])
            for case, value in values:
                writer.writerow([case, "voltage", 7, value])

    def test_both_cases_at_the_voltage_in_both_tables_pass(self):
        self.assertTrue(equivalent_voltage_case_tie(self.a, self.b, self.left, self.right, 1e-3))

    def test_a_case_missing_or_at_another_voltage_fails(self):
        self.write(self.b, [("branch", 0.7)])
        self.assertFalse(equivalent_voltage_case_tie(self.a, self.b, self.left, self.right, 1e-3))
        self.write(self.b, [("branch", 0.7), ("generator", 0.7001)])
        self.assertFalse(equivalent_voltage_case_tie(self.a, self.b, self.left, self.right, 1e-3))


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    print("No errors detected" if result.wasSuccessful() else "failure detected")
    sys.exit(0 if result.wasSuccessful() else 1)
