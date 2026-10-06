#!/usr/bin/env python3
"""A later CPU outage must agree with that outage checked on its own."""

import argparse
import csv
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from run_ca_test import compare_table, rows, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cax", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--contingency-list", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--solver", default="klu")
    args = parser.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.contingency_list)
    cases = tree.getroot().find("./Contingency_analysis/Contingencies")
    if cases is None or len(cases) != 2:
        parser.error("the restoration fixture must contain two ordered outages")
    later_name = cases[1].findtext("contingencyName")
    cases.remove(cases[0])
    single = work / "single.xml"
    tree.write(single, encoding="utf-8", xml_declaration=True)

    errors = []
    for name, outages in (("sequence", args.contingency_list), ("single", str(single))):
        directory = work / name
        code, _ = run(args.cax, str(directory), args.raw, None, args.solver,
                      contingency_list=outages)
        if code:
            errors.append(name + " CPU run failed")
            continue
        convergence = rows(directory / "ca_results_convergence.csv")
        selected = [row for row in convergence[1:] if row[1] == later_name]
        if len(selected) != 1 or selected[0][-1] != "OK":
            errors.append(name + " did not solve the later outage successfully")
            continue
        event = selected[0][0]
        filtered = work / (name + "_later")
        filtered.mkdir(exist_ok=True)
        for table in ("convergence", "delta", "violations", "contingencies"):
            filename = "ca_results_%s.csv" % table
            data = rows(directory / filename)
            with (filtered / filename).open("w") as output:
                writer = csv.writer(output)
                writer.writerow(data[0])
                for row in data[1:]:
                    if row[0] == event:
                        writer.writerow(["1"] + row[1:])
    if not errors:
        for table, nkey in (("convergence", 1), ("delta", 6),
                            ("violations", 4), ("contingencies", 1)):
            compare_table(str(work / "sequence_later"), str(work / "single_later"),
                          "ca_results_%s.csv" % table, nkey, 1e-6, errors)
    for error in errors:
        print("FAILED:", error)
    print("%d failure detected" % len(errors) if errors else "No errors detected")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
