"""Fail if any of the three benchmark milestone gates is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_directory", type=Path)
    args = parser.parse_args()
    files = {
        "public_therm_edge": "therm_spacer_benchmark.json",
        "low_e_thin_triple_pywincalc": "pywincalc_low_e_thin_triple.json",
        "severity_separation": "severity_separation_summary.json",
        "many_query_cpu": "many_query_cpu_summary.json",
        "batched_gpu": "many_query_gpu_summary.json",
    }
    gates = {}
    for name, filename in files.items():
        data = json.loads((args.validation_directory / filename).read_text())
        checks = data["acceptance"]
        gates[name] = {"file": filename, "checks": checks,
                       "passed": all(checks.values())}
    manifest = {"all_gates_passed": all(item["passed"] for item in gates.values()),
                "gates": gates}
    (args.validation_directory / "milestone_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not manifest["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
