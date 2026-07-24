"""
Run the full integration lab: six company archetypes + complex cases.

    python -m integrations.run_all

Writes results to integrations/results/integration_results.json and prints
a pass/fail table. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .companies import ALL_COMPANIES
from .complex_cases import (ALL_COMPLEX, check_trap_avoided, run_drift,
                            run_multitenant)
from .core import run_full_suite

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CHECK_COLUMNS = [
    "deterministic", "goal_reached", "chain_valid", "tamper_evident",
    "provenance_closed", "replay_ok", "injection_refused",
    "recovered_after_injection", "checkpoint_blocks_without_human",
]


def run_case(case: dict) -> dict:
    rt_kwargs = case.get("rt_kwargs", {})
    res = run_full_suite(
        case["name"], case["builder"], case["goal"],
        case["lie_actions"], case["claimed_true"], **rt_kwargs)
    res["injection_story"] = case.get("injection_story", "")
    for extra in case.get("extra_checks", []):
        if extra == "trap_avoided":
            res["trap_avoided"] = check_trap_avoided(res)
            res["all_passed"] = res["all_passed"] and res["trap_avoided"]
    return res


def main() -> int:
    all_results = []
    for case in ALL_COMPANIES + ALL_COMPLEX:
        res = run_case(case)
        all_results.append(res)
        flag = "PASS" if res["all_passed"] else "FAIL"
        print(f"[{flag}] {res['company']:<12} steps={res['steps']:<3} "
              f"plan={len(res['plan'])} status={res['status']}")

    for special in (run_multitenant, run_drift):
        res = special()
        all_results.append(res)
        flag = "PASS" if res["all_passed"] else "FAIL"
        print(f"[{flag}] {res['company']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "integration_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nwrote {out}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r["all_passed"])
    print(f"{passed}/{total} cases fully passed")
    if passed < total:
        for r in all_results:
            if not r["all_passed"]:
                failing = {k: v for k, v in r.items()
                           if isinstance(v, bool) and not v}
                print(f"  FAIL {r['company']}: {sorted(failing)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
