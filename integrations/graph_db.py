"""
Provenance graph database: the audit log lifted into a typed, queryable
graph, plus an integrity validator an auditor can run independently.

Node types
    company   company::<id>
    fact      fact::<company>::<name>       (company-scoped: partitions
                                             can never share a fact node)
    event     act::<company>::<seq>         (executions, refusals,
                                             checkpoint blocks, security
                                             refusals - ALL events)

Edge types
    INITIAL   company -> fact               fact held before any action
    REQUIRES  event -> fact                 checked precondition, with
                                            resolved_from = seq of the
                                            producing event or "initial"
    PRODUCES  event -> fact
    DELETES   event -> fact
    FOLLOWS   event -> event                the per-company chain order

Everything is stdlib, deterministic, and serializable; canonical_hash()
is stable across insertion order.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple


class ProvenanceGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        # per-company fact timelines: fact name -> [(seq, "produce"|"delete")]
        self._timelines: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
        self._last_event: Dict[str, str] = {}

    # -- construction --------------------------------------------------------
    def _fact_id(self, company: str, fact: str) -> str:
        return f"fact::{company}::{fact}"

    def _ensure_fact(self, company: str, fact: str) -> str:
        fid = self._fact_id(company, fact)
        if fid not in self.nodes:
            self.nodes[fid] = {"type": "fact", "company": company, "name": fact,
                               "initial": False}
        return fid

    def add_company(self, company: str, initial_facts) -> None:
        cid = f"company::{company}"
        if cid not in self.nodes:
            self.nodes[cid] = {"type": "company", "name": company}
            self._timelines.setdefault(company, {})
        for f in sorted(initial_facts):
            fid = self._ensure_fact(company, f)
            self.nodes[fid]["initial"] = True
            self.edges.append({"type": "INITIAL", "src": cid, "dst": fid})

    def add_record(self, company: str, rec: dict) -> None:
        """Ingest one audit record (any type) as an event node + edges."""
        eid = f"act::{company}::{rec['seq']}"
        self.nodes[eid] = {
            "type": "event", "company": company, "seq": rec["seq"],
            "event": rec["type"], "action": rec.get("action"),
            "actor": rec.get("actor"), "record_hash": rec["record_hash"],
        }
        tl = self._timelines.setdefault(company, {})

        if company in self._last_event:
            self.edges.append({"type": "FOLLOWS",
                               "src": self._last_event[company], "dst": eid})
        self._last_event[company] = eid

        if rec["type"] == "execution":
            for pre in rec.get("preconditions", []):
                fid = self._ensure_fact(company, pre)
                resolved = "initial"
                for seq, kind in tl.get(pre, []):
                    if seq < rec["seq"] and kind == "produce":
                        resolved = seq
                self.edges.append({"type": "REQUIRES", "src": eid, "dst": fid,
                                   "seq": rec["seq"], "resolved_from": resolved})
            for fact in rec.get("added", []):
                fid = self._ensure_fact(company, fact)
                self.edges.append({"type": "PRODUCES", "src": eid, "dst": fid,
                                   "seq": rec["seq"]})
                tl.setdefault(fact, []).append((rec["seq"], "produce"))
            for fact in rec.get("removed", []):
                fid = self._ensure_fact(company, fact)
                self.edges.append({"type": "DELETES", "src": eid, "dst": fid,
                                   "seq": rec["seq"]})
                tl.setdefault(fact, []).append((rec["seq"], "delete"))

    def add_run(self, company: str, initial_facts, records: List[dict]) -> None:
        self.add_company(company, initial_facts)
        for rec in records:
            self.add_record(company, rec)

    # -- canonical form ------------------------------------------------------
    def canonical(self) -> str:
        nodes = {k: self.nodes[k] for k in sorted(self.nodes)}
        edges = sorted(self.edges, key=lambda e: json.dumps(e, sort_keys=True))
        return json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True,
                          separators=(",", ":"))

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def to_json(self) -> str:
        return self.canonical()

    @classmethod
    def from_json(cls, blob: str) -> "ProvenanceGraph":
        data = json.loads(blob)
        g = cls()
        g.nodes = dict(data["nodes"])
        g.edges = list(data["edges"])
        # rebuild timelines from edges (needed only for further ingestion)
        for e in g.edges:
            if e["type"] in ("PRODUCES", "DELETES"):
                node = g.nodes[e["dst"]]
                tl = g._timelines.setdefault(node["company"], {})
                kind = "produce" if e["type"] == "PRODUCES" else "delete"
                tl.setdefault(node["name"], []).append((e["seq"], kind))
        for comp in g._timelines.values():
            for events in comp.values():
                events.sort()
        return g

    # -- queries -------------------------------------------------------------
    def why(self, company: str, fact: str, at_seq: Optional[int] = None,
            _memo: Optional[dict] = None) -> Optional[dict]:
        """Support tree: which event established `fact` (as of at_seq), and
        recursively, what supported THAT event's preconditions. Memoized on
        (fact, resolved producer) so deep DAGs stay linear; subtrees are
        shared, not duplicated."""
        if _memo is None:
            _memo = {}
        fid = self._fact_id(company, fact)
        if fid not in self.nodes:
            return None
        tl = self._timelines.get(company, {}).get(fact, [])
        last = None
        for seq, kind in tl:
            if at_seq is not None and seq >= at_seq:
                break
            last = (seq, kind)
        key = (company, fact, last)
        if key in _memo:
            return _memo[key]
        if last is None:
            result = {"fact": fact, "source": "initial"} \
                if self.nodes[fid].get("initial") else None
            _memo[key] = result
            return result
        if last[1] == "delete":
            _memo[key] = None  # fact not held at this point
            return None
        seq = last[0]
        eid = f"act::{company}::{seq}"
        event = self.nodes.get(eid, {})
        supports = []
        for e in self.edges:
            if e["type"] == "REQUIRES" and e["src"] == eid:
                pre_name = self.nodes[e["dst"]]["name"]
                supports.append(self.why(company, pre_name, at_seq=seq,
                                         _memo=_memo))
        result = {"fact": fact, "source": "event", "seq": seq,
                  "action": event.get("action"),
                  "record_hash": event.get("record_hash"),
                  "supported_by": supports}
        _memo[key] = result
        return result

    def blast_radius(self, company: str, fact: str) -> Set[str]:
        """Everything downstream of a fact: events that required it, facts
        those events produced, transitively."""
        start = self._fact_id(company, fact)
        out: Dict[str, List[dict]] = {}
        for e in self.edges:
            out.setdefault(e["src"], []).append(e)
        req_of: Dict[str, List[str]] = {}
        for e in self.edges:
            if e["type"] == "REQUIRES":
                req_of.setdefault(e["dst"], []).append(e["src"])
        seen, frontier = set(), [start]
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.startswith("fact::"):
                frontier.extend(req_of.get(node, []))
            else:
                for e in out.get(node, []):
                    if e["type"] == "PRODUCES":
                        frontier.append(e["dst"])
        seen.discard(start)
        return seen

    # -- the independent validator -------------------------------------------
    def validate(self) -> List[str]:
        problems: List[str] = []
        node_ids = set(self.nodes)

        for e in self.edges:
            if e["src"] not in node_ids or e["dst"] not in node_ids:
                problems.append(f"dangling edge {e}")
                continue
            src, dst = self.nodes[e["src"]], self.nodes[e["dst"]]
            # cross-partition edges are forbidden
            sc, dc = src.get("company") or src.get("name"), \
                dst.get("company") or dst.get("name")
            if sc != dc:
                problems.append(f"cross-company edge {e['type']} {e['src']} -> {e['dst']}")
            if e["type"] == "INITIAL" and (src["type"] != "company"
                                           or dst["type"] != "fact"):
                problems.append(f"INITIAL edge with wrong endpoint types: {e}")
            if e["type"] in ("REQUIRES", "PRODUCES", "DELETES") and (
                    src["type"] != "event" or dst["type"] != "fact"):
                problems.append(f"{e['type']} edge with wrong endpoint types: {e}")
            if e["type"] == "FOLLOWS":
                if src["type"] != "event" or dst["type"] != "event":
                    problems.append(f"FOLLOWS edge with wrong endpoint types: {e}")
                elif src["seq"] >= dst["seq"]:
                    problems.append(f"FOLLOWS edge not forward in time: {e}")

        # temporal soundness: every REQUIRES must be satisfied by the initial
        # state or an earlier PRODUCES with no DELETES in between
        timelines: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        for e in self.edges:
            if e["type"] in ("PRODUCES", "DELETES"):
                n = self.nodes.get(e["dst"])
                if n is None:
                    continue
                key = (n["company"], n["name"])
                kind = "produce" if e["type"] == "PRODUCES" else "delete"
                timelines.setdefault(key, []).append((e["seq"], kind))
        for tl in timelines.values():
            tl.sort()

        for e in self.edges:
            if e["type"] != "REQUIRES":
                continue
            n = self.nodes.get(e["dst"])
            if n is None:
                continue
            key = (n["company"], n["name"])
            held = bool(n.get("initial"))
            for seq, kind in timelines.get(key, []):
                if seq >= e["seq"]:
                    break
                held = kind == "produce"
            if not held:
                problems.append(
                    f"temporal violation: event seq {e['seq']} requires "
                    f"'{n['name']}' which was not held at that point")
            if e.get("resolved_from") != "initial" and \
                    isinstance(e.get("resolved_from"), int) and \
                    e["resolved_from"] >= e["seq"]:
                problems.append(f"provenance points forward in time: {e}")

        # partition isolation: every connected component has exactly 1 company
        adj: Dict[str, Set[str]] = {}
        for e in self.edges:
            adj.setdefault(e["src"], set()).add(e["dst"])
            adj.setdefault(e["dst"], set()).add(e["src"])
        seen: Set[str] = set()
        for node in self.nodes:
            if node in seen:
                continue
            comp, frontier = set(), [node]
            while frontier:
                x = frontier.pop()
                if x in comp:
                    continue
                comp.add(x)
                frontier.extend(adj.get(x, ()))
            seen |= comp
            companies = [x for x in comp
                         if self.nodes.get(x, {}).get("type") == "company"]
            if len(companies) > 1:
                problems.append(
                    f"partition breach: component contains {sorted(companies)}")
        return problems
