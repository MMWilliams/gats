"""
Run every layer of the integration lab:

  Layer 1: per-company suites (companies + complex cases + multitenant + drift)
  Layer 2: deep validation (generated family, scale, tamper, injection,
           crash-resume)
  Layer 3: graph database validation over the entire corpus of runs

    python -m integrations.run_deep

Writes integrations/results/deep_results.json. Exit 0 only if all pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .companies import ALL_COMPANIES
from .complex_cases import ALL_COMPLEX, run_drift, run_multitenant
from .core import AuditedRuntime
from .deep_validation import (build_scale_system, generate_system,
                              run_crash_resume, run_generated_family,
                              run_injection_battery, run_scale,
                              run_tamper_battery)
from .graph_validation import run_graph_validation
from .run_all import run_case

RESULTS_DIR = Path(__file__).resolve().parent / "results"
APPROVE = lambda tool: True


def main() -> int:
    t_start = time.perf_counter()
    layer1, layer2 = [], []
    corpus = []           # (company, initial_facts, records) for the graph DB
    reference_chains = {}  # for the exhaustive tamper battery

    print("=" * 66)
    print("LAYER 1 - company integrations")
    print("=" * 66)
    for case in ALL_COMPANIES + ALL_COMPLEX:
        res = run_case(case)
        layer1.append(res)
        print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] {res['company']:<14} "
              f"plan={len(res['plan'])} status={res['status']}")
        # reference run for the corpus + tamper battery
        rt = AuditedRuntime(case["builder"](), case["goal"],
                            **case.get("rt_kwargs", {}))
        rt.execute(rt.plan(), human_approver=APPROVE)
        corpus.append((rt.sys.company_id, rt.sys.initial_facts, rt.records))
        reference_chains[res["company"]] = rt.records
    for special in (run_multitenant, run_drift):
        res = special()
        layer1.append(res)
        print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] {res['company']}")

    print()
    print("=" * 66)
    print("LAYER 2 - deep validation batteries")
    print("=" * 66)

    res = run_generated_family(120)
    layer2.append(res)
    print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] generated family: "
          f"{res['solved']}/{res['systems']} solved, "
          f"{res['deterministic']} deterministic, "
          f"{res['injection_refused']}/{res['injection_attempts']} lies refused, "
          f"max plan {res['max_plan_len']}")
    # fold a sample of generated runs into the graph corpus
    for seed in range(0, 120, 6):
        sys_, goal, _ = generate_system(seed)
        rt = AuditedRuntime(sys_, goal, budget=24, max_steps=60, horizon=24)
        p = rt.plan()
        if p:
            rt.execute(p, human_approver=APPROVE)
            corpus.append((rt.sys.company_id, rt.sys.initial_facts, rt.records))

    scale_res, scale_rt = run_scale()
    layer2.append(scale_res)
    print(f"[{'PASS' if scale_res['all_passed'] else 'FAIL'}] scale: "
          f"{scale_res['tools']} tools, plan={scale_res['plan_len']}, "
          f"{scale_res['plan_seconds']}s plan / {scale_res['total_seconds']}s total")
    corpus.append((scale_rt.sys.company_id, scale_rt.sys.initial_facts,
                   scale_rt.records))
    reference_chains["scale"] = scale_rt.records

    res = run_tamper_battery(reference_chains)
    layer2.append(res)
    print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] tamper: "
          f"{res['detected']}/{res['mutations']} mutations detected")

    res = run_injection_battery()
    layer2.append(res)
    print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] injection: "
          f"{res['refused']}/{res['attempts']} refused")

    res = run_crash_resume()
    layer2.append(res)
    print(f"[{'PASS' if res['all_passed'] else 'FAIL'}] crash-resume: "
          f"{res['final_state_matched']}/{res['prefixes']} prefixes converged")

    print()
    print("=" * 66)
    print("LAYER 3 - graph database validation")
    print("=" * 66)
    graph_res = run_graph_validation(corpus)
    print(f"[{'PASS' if graph_res['all_passed'] else 'FAIL'}] graph db: "
          f"{graph_res['nodes']} nodes / {graph_res['edges']} edges, "
          f"integrity_clean={graph_res['integrity_clean']}, "
          f"corruptions {graph_res['corruptions_detected']}"
          f"/{graph_res['corruptions_injected']}, "
          f"why {graph_res['why_correct']}/{graph_res['why_queries']}, "
          f"blast {graph_res['blast_correct']}/{graph_res['blast_queries']}")

    all_results = {"layer1": layer1, "layer2": layer2, "layer3": graph_res,
                   "wall_seconds": round(time.perf_counter() - t_start, 1)}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "deep_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nwrote {out}  ({all_results['wall_seconds']}s total)")

    ok = (all(r["all_passed"] for r in layer1)
          and all(r["all_passed"] for r in layer2)
          and graph_res["all_passed"])
    print("ALL LAYERS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
