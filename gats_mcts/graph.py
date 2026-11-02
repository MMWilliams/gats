"""SQLite transition graph: predictions and sourced execution evidence stay distinct."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .domain import facts, Outcome, Prediction


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def state_id(state):
    return hashlib.sha256(canonical(sorted(state)).encode("utf-8")).hexdigest()


class TransitionGraph:
    def __init__(self, path=":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30.0)
        self._prediction_batch_depth = 0
        self.db.execute("PRAGMA foreign_keys=ON")
        if str(path) != ":memory:":
            # Several agents share one file: readers never block the writer.
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS catalogs(
                namespace TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS states(
                namespace TEXT NOT NULL REFERENCES catalogs(namespace),
                id TEXT NOT NULL, facts TEXT NOT NULL, PRIMARY KEY(namespace,id));
            CREATE TABLE IF NOT EXISTS observations(
                namespace TEXT NOT NULL, event TEXT NOT NULL, before_id TEXT NOT NULL,
                action TEXT NOT NULL, after_id TEXT NOT NULL, source TEXT NOT NULL,
                PRIMARY KEY(namespace,event),
                FOREIGN KEY(namespace,before_id) REFERENCES states(namespace,id),
                FOREIGN KEY(namespace,after_id) REFERENCES states(namespace,id));
            CREATE INDEX IF NOT EXISTS observation_action ON observations(namespace,action);
            CREATE TABLE IF NOT EXISTS predictions(
                namespace TEXT NOT NULL, before_id TEXT NOT NULL, action TEXT NOT NULL,
                revision INTEGER NOT NULL, body TEXT NOT NULL,
                PRIMARY KEY(namespace,before_id,action),
                FOREIGN KEY(namespace,before_id) REFERENCES states(namespace,id));
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def prediction_batch(self):
        """Commit regenerable search predictions once per decision, not per edge."""
        if self._prediction_batch_depth:
            yield
            return
        with self.db:
            self._prediction_batch_depth = 1
            try:
                yield
            finally:
                self._prediction_batch_depth = 0

    def register(self, namespace, catalog):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace required")
        body = canonical(catalog)
        old = self.db.execute("SELECT body FROM catalogs WHERE namespace=?", (namespace,)).fetchone()
        if old and old[0] != body:
            raise ValueError("namespace already belongs to a different catalog/model configuration")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO catalogs VALUES (?,?)", (namespace, body))

    def catalog(self, namespace):
        row = self.db.execute("SELECT body FROM catalogs WHERE namespace=?", (namespace,)).fetchone()
        return json.loads(row[0]) if row else None

    def namespaces(self):
        return [r[0] for r in self.db.execute("SELECT namespace FROM catalogs ORDER BY namespace")]

    def observations_since(self, namespace, rowid):
        """Evidence appended after `rowid`, in commit order, by any writer of this file."""
        rows = self.db.execute("""SELECT o.rowid,o.event,s.facts,o.action,t.facts,o.source
            FROM observations o JOIN states s ON s.namespace=o.namespace AND s.id=o.before_id
            JOIN states t ON t.namespace=o.namespace AND t.id=o.after_id
            WHERE o.namespace=? AND o.rowid>? ORDER BY o.rowid""", (namespace, rowid))
        return [{"rowid": r, "event": e, "before": json.loads(b), "action": a,
                 "after": json.loads(t), "source": src} for r, e, b, a, t, src in rows]

    def _state(self, namespace, state):
        state = facts(state)
        key = state_id(state)
        self.db.execute("INSERT OR IGNORE INTO states VALUES (?,?,?)",
                        (namespace, key, canonical(sorted(state))))
        return key

    def observations(self, namespace):
        rows = self.observations_since(namespace, 0)
        for row in rows:
            del row["rowid"]
        return sorted(rows, key=lambda r: r["event"])

    def observe(self, namespace, event, before, action, after, source):
        if self._prediction_batch_depth:
            raise RuntimeError("execution observations cannot be written inside a simulation batch")
        if not all(isinstance(v, str) and v.strip() for v in (event, source)):
            raise ValueError("observation event id and source are required")
        row = (state_id(before), action, state_id(after), source)
        old = self.db.execute("""SELECT before_id,action,after_id,source FROM observations
            WHERE namespace=? AND event=?""", (namespace, event)).fetchone()
        if old:
            if old != row:
                raise ValueError("conflicting observation event id")
            return False
        with self.db:
            self._state(namespace, before)
            self._state(namespace, after)
            self.db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?)",
                            (namespace, event, *row))
            # Conservative action-wide invalidation includes projected L2 contexts.
            self.db.execute("DELETE FROM predictions WHERE namespace=? AND action=?",
                            (namespace, action))
        return True

    def get_prediction(self, namespace, state, action, revision):
        row = self.db.execute("""SELECT body FROM predictions
            WHERE namespace=? AND before_id=? AND action=? AND revision=?""",
                              (namespace, state_id(state), action, revision)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return Prediction(tuple(Outcome(frozenset(o["state"]), o["probability"])
                                for o in data["outcomes"]), data["layer"],
                          data["support"], data["confidence_lower"])

    def put_prediction(self, namespace, state, action, revision, prediction):
        data = {"layer": prediction.layer, "support": prediction.support,
                "confidence_lower": prediction.confidence_lower,
                "outcomes": [{"state": sorted(o.state), "probability": o.probability}
                             for o in prediction.outcomes]}
        with self.prediction_batch():
            key = self._state(namespace, state)
            for outcome in prediction.outcomes:
                self._state(namespace, outcome.state)
            self.db.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)",
                            (namespace, key, action, revision, canonical(data)))

    def export(self):
        # Full facts, stable IDs, and event provenance make the graph inspectable.
        result = {"format": "gats-mcts-graph-v1"}
        for table, columns in (
            ("catalogs", ("namespace", "body")),
            ("states", ("namespace", "id", "facts")),
            ("observations", ("namespace", "event", "before_id", "action", "after_id", "source")),
            ("predictions", ("namespace", "before_id", "action", "revision", "body")),
        ):
            rows = self.db.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY "
                                   + ",".join(columns[:3]))
            result[table] = [dict(zip(columns, row)) for row in rows]
        for row in result["catalogs"] + result["predictions"]:
            row["body"] = json.loads(row["body"])
        for row in result["states"]:
            row["facts"] = json.loads(row["facts"])
        result["sha256"] = hashlib.sha256(canonical(result).encode("utf-8")).hexdigest()
        return result

    @staticmethod
    def verify_export(data):
        body = {k: v for k, v in data.items() if k != "sha256"}
        return (data.get("format") == "gats-mcts-graph-v1" and
                data.get("sha256") == hashlib.sha256(canonical(body).encode("utf-8")).hexdigest())

    def validate(self):
        errors = []
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            errors.append("SQLite integrity check failed")
        errors.extend("foreign key violation" for _ in self.db.execute("PRAGMA foreign_key_check"))
        data = self.export()
        catalogs = {r["namespace"]: {a["name"] for a in r["body"]["actions"]}
                    for r in data["catalogs"]}
        nodes = {(r["namespace"], r["id"]) for r in data["states"]}
        for r in data["states"]:
            if state_id(facts(r["facts"])) != r["id"]:
                errors.append("state hash mismatch")
        for r in data["observations"] + data["predictions"]:
            if r["action"] not in catalogs.get(r["namespace"], set()):
                errors.append("unknown action")
        for r in data["predictions"]:
            p = r["body"]
            if p["layer"] not in {"L1", "L1r", "L2"} or p["support"] < 0 or r["revision"] < 0:
                errors.append("invalid prediction metadata")
            probabilities = [o["probability"] for o in p["outcomes"]]
            if (not probabilities or any(not math.isfinite(v) or not 0 < v <= 1 for v in probabilities)
                    or not math.isclose(sum(probabilities), 1.0)):
                errors.append("invalid outcome distribution")
            for outcome in p["outcomes"]:
                if (r["namespace"], state_id(outcome["state"])) not in nodes:
                    errors.append("missing predicted successor")
        return errors
