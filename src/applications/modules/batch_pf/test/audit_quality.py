#!/usr/bin/env python3
"""Record analyzer diagnostics and reject new ones against a reviewed baseline.

This does not suppress diagnostics or approve exceptions. The register and
baseline retain findings, including ones that still need a code review.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def relative(path):
    path = os.path.normpath(path)
    marker = "/src/applications/"
    if marker in path:
        return "src/applications/" + path.split(marker, 1)[1]
    return path


ADAPTERS = (
    "src/applications/components/pf_matrix",
    "src/applications/modules/powerflow",
    "src/applications/contingency_analysis/ca_driver.cpp",
)


def line_filter(root, base):
    """Select added or changed adapter lines without auditing legacy code."""
    diff = subprocess.check_output(
        ["git", "diff", base, "--unified=0", "--", *ADAPTERS], cwd=root, text=True)
    files = []
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = dict(name=os.path.join(os.path.abspath(root), line[6:]), lines=[])
            files.append(current)
        match = re.match(r"@@ .* \+(\d+)(?:,(\d+))? @@", line)
        if match and current is not None:
            start, count = int(match[1]), int(match[2] or 1)
            if count:
                current["lines"].append([start, start + count - 1])
    return [item for item in files if item["lines"]]


def diagnostics(logs, xmls, scope, build_logs=()):
    found = {}
    ranges = {relative(item["name"]): item["lines"] for item in scope}

    def included(path, row):
        return ("/batch_pf/" in path or
                any(first <= row <= last for first, last in ranges.get(path, [])))

    pattern = re.compile(r"(.*?):(\d+):(\d+): (warning|error): (.*?) \[([^]]+)\]")
    for log in [*logs, *build_logs]:
        with open(log) as stream:
            content = stream.read()
        if log not in build_logs and "clang-tidy" not in content:
            raise RuntimeError("No clang-tidy invocation found; verify analyzer coverage")
        for line in content.splitlines():
            if "Found compiler error(s)" in line:
                raise RuntimeError("clang-tidy encountered a compiler error")
            match = pattern.search(line)
            if not match:
                continue
            path, row, col, severity, message, check = match.groups()
            if log in build_logs and not check.startswith("-W"):
                continue
            path = relative(path)
            if not included(path, int(row)):
                continue
            tool = ("compiler" if log in build_logs else "clang-static-analyzer"
                    if check.startswith("clang-analyzer-") else "clang-tidy")
            item = dict(tool=tool, file=path, line=int(row), column=int(col),
                        severity=severity, check=check, message=message)
            found[json.dumps(item, sort_keys=True)] = item
    for xml in xmls:
        for error in ET.parse(xml).findall("./errors/error"):
            if error.get("severity") == "error" and error.get("id") == "preprocessorErrorDirective":
                raise RuntimeError("cppcheck could not preprocess an input: " + error.get("msg", ""))
            for location in error.findall("location")[:1]:
                path = relative(location.attrib["file"])
                row = int(location.get("line", "0"))
                if not included(path, row):
                    continue
                item = dict(tool="cppcheck", file=path, line=row,
                            column=int(location.get("column", "0")), severity=error.get("severity"),
                            check=error.get("id"), message=error.get("msg"))
                found[json.dumps(item, sort_keys=True)] = item
    return sorted(found.values(), key=lambda x: (x["file"], x["line"], x["column"], x["check"]))


def signature(item):
    # A line moving during an edit does not create a new diagnostic. Counts
    # still distinguish two occurrences of the same message in one file.
    return (item["tool"], item["file"], item["severity"], item["check"], item["message"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-log", action="append", default=[])
    parser.add_argument("--cppcheck-xml", action="append", default=[])
    parser.add_argument("--build-log", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--adapter-lines")
    parser.add_argument("--write-line-filter", action="store_true")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--base-revision", default="b32969b0")
    args = parser.parse_args()
    if args.write_line_filter:
        with open(args.output, "w") as stream:
            json.dump(line_filter(args.source_root, args.base_revision), stream)
        return 0
    if not args.clang_log or not args.cppcheck_xml:
        parser.error("Provide clang-tidy and cppcheck results")
    scope = []
    if args.adapter_lines:
        with open(args.adapter_lines) as stream:
            scope = json.load(stream)
    items = diagnostics(args.clang_log, args.cppcheck_xml, scope, args.build_log)
    with open(args.output, "w") as stream:
        for item in items:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(dict(counts=collections.Counter(x["tool"] for x in items),
                          by_check=collections.Counter(x["check"] for x in items)), sort_keys=True))
    if any(item["severity"] == "error" for item in items):
        print("Analyzer errors cannot be accepted into a baseline")
        return 1
    if args.baseline:
        with open(args.baseline) as stream:
            baseline = collections.Counter(signature(json.loads(line)) for line in stream)
        current = collections.Counter(signature(item) for item in items)
        added = current - baseline
        for key, count in added.items():
            print("NEW:", count, *key)
        return 1 if added else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
