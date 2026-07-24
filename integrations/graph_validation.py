"""
Layer 3: prove the provenance graph database itself is robust.

  G1. Integrity      - merged graph over every run in the lab validates clean.
  G2. Round-trip     - serialize -> load -> identical canonical hash.
  G3. Order-invariance - ingesting runs in a different order yields the
                       identical canonical hash (safe for distributed merge).
  G4. Corruption     - a seeded battery of graph corruptions (dangling nodes,
                       cross-tenant edges, deleted producers, time-reversed
                       provenance, flipped chain order); the validator must
                       flag every single one.
  G5. Query truth    - why() support trees and blast_radius() closures are
                       recomputed brute-force from the raw audit records and
                       must agree exactly.
  G6. Scale/perf     - node/edge counts and wall-clock for build, validate,
                       and queries over the full lab corpus.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Set, Tuple

from .graph_db import ProvenanceGraph


def build_corpus_graph(runs: List[Tuple[str, frozenset, List[dict]]]
                       ) -> ProvenanceGraph:
    g = ProvenanceGraph()
    for company, initial, records in runs:
        g.add_run(company, initial, records)
    return g


# -- G5 brute-force oracles (from raw records, independent of the graph) ----

def oracle_last_producer(records: List[dict], fact: str,
                         before_seq: int = 10 ** 9):
    last = None
    for rec in records:
        if rec["type"] != "execution" or rec["seq"] >= before_seq:
            continue
        if fact in rec.get("added", []):
            last = rec
        if fact in rec.get("removed", []):
            last = "DELETED"
    return last


def oracle_why_matches(g: ProvenanceGraph, company: str, initial: frozenset,
                       records: List[dict], fact: str,
                       before_seq: int = 10 ** 9,
                       _memo: dict = None) -> bool:
    if _memo is None:
        _memo = {}
    oracle = oracle_last_producer(records, fact, before_seq)
    key = (fact, oracle["seq"] if isinstance(oracle, dict) else oracle)
    if key in _memo:
        return _memo[key]
    _memo[key] = True  # provisional (DAG, no true cycles)
    tree = g.why(company, fact, at_seq=None if before_seq == 10 ** 9 else before_seq)
    if oracle is None:
        ok = (tree is not None and tree.get("source") == "initial") \
            if fact in initial else tree is None
        _memo[key] = ok
        return ok
    if oracle == "DELETED":
        _memo[key] = tree is None
        return _memo[key]
    if tree is None or tree.get("source") != "event" \
            or tree["seq"] != oracle["seq"] \
            or tree["record_hash"] != oracle["record_hash"]:
        _memo[key] = False
        return False
    # recurse: each support must itself match the oracle at the parent's seq
    pres = sorted(oracle.get("preconditions", []))
    subs = sorted(tree.get("supported_by", []),
                  key=lambda t: (t or {}).get("fact", ""))
    if len(pres) != len(subs):
        _memo[key] = False
        return False
    for pre, sub in zip(pres, subs):
        if sub is None or sub.get("fact") != pre:
            _memo[key] = False
            return False
        if not oracle_why_matches(g, company, initial, records, pre,
                                  before_seq=oracle["seq"], _memo=_memo):
            _memo[key] = False
            return False
    return _memo[key]


def oracle_blast_radius(records: List[dict], company: str,
                        fact: str) -> Set[str]:
    """Forward closure computed straight from the records."""
    tainted_facts = {fact}
    tainted_ids: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for rec in records:
            if rec["type"] != "execution":
                continue
            eid = f"act::{company}::{rec['seq']}"
            if eid in tainted_ids:
                continue
            if tainted_facts & set(rec.get("preconditions", [])):
                tainted_ids.add(eid)
                for f in rec.get("added", []):
                    if f not in tainted_facts:
                        tainted_facts.add(f)
                changed = True
    out = set(tainted_ids)
    out |= {f"fact::{company}::{f}" for f in tainted_facts - {fact}}
    return out


# -- G4 corruption operators -------------------------------------------------

def _corruptions(g: ProvenanceGraph, rng: random.Random, n: int):
    """Yield n corrupted copies; each must be flagged by validate()."""
    made = 0
    guard = 0
    while made < n and guard < n * 60:
        guard += 1
        bad = ProvenanceGraph()
        bad.nodes = copy.deepcopy(g.nodes)
        bad.edges = copy.deepcopy(g.edges)
        op = rng.choice(["dangle", "cross", "kill_producer",
                         "future_provenance", "flip_follows"])
        if op == "dangle":
            e = rng.choice(bad.edges)
            if e["dst"] in bad.nodes:
                del bad.nodes[e["dst"]]
            else:
                continue
        elif op == "cross":
            reqs = [e for e in bad.edges if e["type"] == "REQUIRES"]
            facts = list({e["dst"] for e in bad.edges if e["type"] == "REQUIRES"})
            if not reqs or len(facts) < 2:
                continue
            e = rng.choice(reqs)
            own_company = bad.nodes[e["dst"]]["company"]
            others = [f for f in facts
                      if bad.nodes[f]["company"] != own_company]
            if not others:
                continue
            e["dst"] = rng.choice(others)
        elif op == "kill_producer":
            # remove a PRODUCES edge for a non-initial fact that is
            # REQUIRED later and produced exactly once
            prods: Dict[str, List[dict]] = {}
            for e in bad.edges:
                if e["type"] == "PRODUCES":
                    prods.setdefault(e["dst"], []).append(e)
            required = {e["dst"]: e for e in bad.edges if e["type"] == "REQUIRES"}
            candidates = [fid for fid, es in prods.items()
                          if len(es) == 1 and fid in required
                          and not bad.nodes[fid].get("initial")
                          and es[0]["seq"] < required[fid]["seq"]]
            if not candidates:
                continue
            fid = rng.choice(candidates)
            bad.edges.remove(prods[fid][0])
        elif op == "future_provenance":
            reqs = [e for e in bad.edges if e["type"] == "REQUIRES"]
            if not reqs:
                continue
            e = rng.choice(reqs)
            e["resolved_from"] = e["seq"] + 100
        elif op == "flip_follows":
            fols = [e for e in bad.edges if e["type"] == "FOLLOWS"]
            if not fols:
                continue
            e = rng.choice(fols)
            e["src"], e["dst"] = e["dst"], e["src"]
        made += 1
        yield op, bad


def run_graph_validation(runs: List[Tuple[str, frozenset, List[dict]]],
                         n_corruptions: int = 60) -> dict:
    t0 = time.perf_counter()
    g = build_corpus_graph(runs)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    problems = g.validate()
    validate_s = time.perf_counter() - t0

    # G2 round-trip
    g2 = ProvenanceGraph.from_json(g.to_json())
    roundtrip_ok = g2.canonical_hash() == g.canonical_hash()

    # G3 order invariance
    g3 = build_corpus_graph(list(reversed(runs)))
    order_ok = g3.canonical_hash() == g.canonical_hash()

    # G4 corruption battery
    rng = random.Random(1337)
    corrupted = detected = 0
    by_op: Dict[str, List[int]] = {}
    for op, bad in _corruptions(g, rng, n_corruptions):
        corrupted += 1
        found = len(bad.validate()) > 0
        detected += int(found)
        by_op.setdefault(op, [0, 0])
        by_op[op][0] += int(found)
        by_op[op][1] += 1

    # G5 query truth on every non-generated run (archetypes + complex + scale)
    why_checked = why_ok = blast_checked = blast_ok = 0
    t_q = 0.0
    for company, initial, records in runs:
        if company.startswith("gen"):
            continue
        exec_recs = [r for r in records if r["type"] == "execution"]
        if not exec_recs:
            continue
        final_facts = set()
        for r in exec_recs:
            final_facts |= set(r.get("added", []))
        for fact in sorted(final_facts):
            t1 = time.perf_counter()
            ok = oracle_why_matches(g, company, initial, records, fact)
            t_q += time.perf_counter() - t1
            why_checked += 1
            why_ok += int(ok)
        probe = sorted(initial)[0] if initial else None
        if probe:
            t1 = time.perf_counter()
            got = g.blast_radius(company, probe)
            t_q += time.perf_counter() - t1
            want = oracle_blast_radius(records, company, probe)
            blast_checked += 1
            blast_ok += int(got == want)

    res = {
        "battery": "graph_db",
        "runs_ingested": len(runs),
        "nodes": len(g.nodes), "edges": len(g.edges),
        "build_seconds": round(build_s, 2),
        "validate_seconds": round(validate_s, 2),
        "integrity_clean": len(problems) == 0,
        "integrity_problems": problems[:5],
        "roundtrip_ok": roundtrip_ok,
        "order_invariant": order_ok,
        "corruptions_injected": corrupted,
        "corruptions_detected": detected,
        "corruptions_by_op": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_op.items())},
        "why_queries": why_checked, "why_correct": why_ok,
        "blast_queries": blast_checked, "blast_correct": blast_ok,
        "query_seconds_total": round(t_q, 2),
        "all_passed": (len(problems) == 0 and roundtrip_ok and order_ok
                       and corrupted == n_corruptions == detected
                       and why_checked == why_ok and blast_checked == blast_ok),
    }
    return res
