#!/usr/bin/env python3
"""Verify that the quality gate keeps findings and rejects incomplete analysis."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_quality as audit


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.log = self.root / "clang.log"
        self.xml = self.root / "cppcheck.xml"
        self.xml.write_text('<results><errors/></results>')
        self.output = self.root / "findings.jsonl"

    def warning(self, path, row, message="review this"):
        return f"{path}:{row}:1: warning: {message} [cppcoreguidelines-check]\n"

    def run_audit(self, *extra):
        arguments = ["audit_quality.py", "--clang-log", str(self.log),
                     "--cppcheck-xml", str(self.xml), "--output", str(self.output), *extra]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            return audit.main()

    def test_scope_keeps_new_code_and_only_changed_adapter_lines(self):
        module = "/src/src/applications/modules/batch_pf/host/settings.cpp"
        adapter = "/src/src/applications/modules/powerflow/pf_app_module.cpp"
        self.log.write_text("clang-tidy-18\n" + self.warning(module, 90) +
                            self.warning(adapter, 12) + self.warning(adapter, 50))
        scope = [dict(name=adapter, lines=[[10, 20]])]
        findings = audit.diagnostics([self.log], [self.xml], scope)
        self.assertEqual({(item["file"], item["line"]) for item in findings}, {
            ("src/applications/modules/batch_pf/host/settings.cpp", 90),
            ("src/applications/modules/powerflow/pf_app_module.cpp", 12)})

    def test_moving_a_warning_is_allowed_but_a_second_occurrence_fails(self):
        source = "/src/src/applications/modules/batch_pf/host/settings.cpp"
        self.log.write_text("clang-tidy-18\n" + self.warning(source, 1))
        self.assertEqual(self.run_audit(), 0)
        baseline = self.root / "baseline.jsonl"
        baseline.write_text(self.output.read_text())
        self.log.write_text("clang-tidy-18\n" + self.warning(source, 20))
        self.assertEqual(self.run_audit("--baseline", str(baseline)), 0)
        self.log.write_text(self.log.read_text() + self.warning(source, 30))
        self.assertEqual(self.run_audit("--baseline", str(baseline)), 1)

    def test_compiler_failure_or_missing_invocation_cannot_pass(self):
        for content in ["clang-tidy-18\nFound compiler error(s).\n", "empty analysis\n"]:
            self.log.write_text(content)
            with self.assertRaises(RuntimeError):
                self.run_audit()

    def test_cppcheck_error_cannot_be_baselined(self):
        self.log.write_text("clang-tidy-18\n")
        self.xml.write_text('<results><errors><error severity="error" id="bad" msg="bad">'
                            '<location file="/src/src/applications/modules/batch_pf/core/model.cpp" '
                            'line="1"/></error></errors></results>')
        self.assertEqual(self.run_audit(), 1)

    def test_valid_empty_analysis_passes(self):
        self.log.write_text("clang-tidy-18\n")
        self.assertEqual(self.run_audit(), 0)
        self.assertEqual(self.output.read_text(), "")

    def test_compiler_warnings_are_counted_separately(self):
        self.log.write_text("clang-tidy-18\n")
        build = self.root / "build.log"
        build.write_text('/src/src/applications/modules/batch_pf/core/model.cpp:1:1: '
                         'warning: lost precision [-Wconversion]\n')
        findings = audit.diagnostics([self.log], [self.xml], [], [build])
        self.assertEqual([(x["tool"], x["check"]) for x in findings],
                         [("compiler", "-Wconversion")])

    def test_third_party_preprocessing_failure_is_not_filtered_out(self):
        self.log.write_text("clang-tidy-18\n")
        self.xml.write_text('<results><errors><error severity="error" '
                            'id="preprocessorErrorDirective" msg="cannot parse Boost">'
                            '<location file="/deps/boost/header.hpp" line="1"/>'
                            '</error></errors></results>')
        with self.assertRaises(RuntimeError):
            self.run_audit()

    def test_nvcc_warning_format_is_counted(self):
        self.log.write_text("clang-tidy-18\n")
        build = self.root / "build.log"
        build.write_text('/src/src/applications/modules/batch_pf/core/engine.cu(20): '
                         'warning #177-D: variable unused\n')
        findings = audit.diagnostics([self.log], [self.xml], [], [build])
        self.assertEqual([(x["tool"], x["check"], x["line"]) for x in findings],
                         [("compiler", "nvcc-177", 20)])

    def test_changed_line_ranges_skip_deletions(self):
        diff = ("+++ b/src/applications/modules/powerflow/pf_app_module.cpp\n"
                "@@ -1,2 +1,0 @@\n@@ -30 +28,3 @@\n@@ -50 +50 @@\n")
        with patch.object(audit.subprocess, "check_output", return_value=diff):
            result = audit.line_filter("/src", "base")
        self.assertEqual(result, [dict(
            name="/src/src/applications/modules/powerflow/pf_app_module.cpp",
            lines=[[28, 30], [50, 50]])])


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    print("No errors detected" if result.wasSuccessful() else "failure detected")
    sys.exit(0 if result.wasSuccessful() else 1)
