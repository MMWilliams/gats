"""Local, stdlib-only commands for planning, trace ingestion and graph inspection."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .benchmark import run_panel
from .audit import audit_population
from .domain import read_catalog
from .graph import TransitionGraph
from .model import LayeredWorldModel
from .search import MCTSPlanner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark", help="evaluate scenarios and populate a new SQLite graph")
    bench.add_argument("--output", default="results/mcts")
    bench.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    bench.add_argument("--simulations", type=int, default=128)
    inspect = commands.add_parser("inspect", help="validate and summarize an existing graph")
    inspect.add_argument("--database", required=True)
    inspect.add_argument("--export", help="optional full graph JSON destination")
    audit = commands.add_parser("audit", help="audit model predictions on bounded reachable fixture states")
    audit.add_argument("--database", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--max-states", type=int, default=128)
    ingest = commands.add_parser("ingest", help="ingest sourced observation JSONL, idempotently")
    ingest.add_argument("--database", required=True)
    ingest.add_argument("--namespace", required=True)
    ingest.add_argument("--catalog", required=True)
    ingest.add_argument("--traces", required=True)
    plan = commands.add_parser("plan", help="choose one action from a populated model; no execution")
    plan.add_argument("--database", required=True)
    plan.add_argument("--namespace", required=True)
    plan.add_argument("--state", required=True, help="JSON array of current facts")
    plan.add_argument("--goal", required=True, help="JSON array of goal facts")
    plan.add_argument("--simulations", type=int, default=128)
    plan.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_population(args.database, args.output, max_states=args.max_states)
        print(json.dumps({"summary": result["summary"], "passed": result["passed"]}, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "benchmark":
        result = run_panel(args.output, args.seeds, simulations=args.simulations, progress=print)
        print(json.dumps({"summary": result["summary"], "graph": result["graph"],
                          "passed": result["passed"]}, indent=2))
        return 0 if result["passed"] else 1
    if args.command in {"plan", "inspect"} and not Path(args.database).is_file():
        parser.error("database does not exist")
    graph = TransitionGraph(args.database)
    try:
        if args.command == "inspect":
            errors, data = graph.validate(), graph.export()
            if args.export:
                Path(args.export).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(json.dumps({"counts": {k: len(data[k]) for k in ("catalogs", "states", "observations", "predictions")},
                              "integrity_errors": errors, "sha256": data["sha256"]}, indent=2))
            return 1 if errors else 0
        if args.command == "ingest":
            actions, contracts = read_catalog(json.loads(Path(args.catalog).read_text(encoding="utf-8")))
            model = LayeredWorldModel(actions, contracts, graph=graph, namespace=args.namespace)
            rows = [json.loads(line) for line in Path(args.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
            print(json.dumps({"new_observations": model.ingest(rows), "coverage": model.coverage()}, indent=2))
            return 0
        row = graph.db.execute("SELECT body FROM catalogs WHERE namespace=?", (args.namespace,)).fetchone()
        if row is None:
            parser.error("namespace not found")
        catalog = json.loads(row[0])
        actions, contracts = read_catalog(catalog)
        model = LayeredWorldModel(actions, contracts, graph=graph, namespace=args.namespace,
                                  min_support=catalog["model"]["min_support"])
        result = MCTSPlanner(model, simulations=args.simulations, seed=args.seed).search(json.loads(args.state), json.loads(args.goal))
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.action is not None or result.status == "goal" else 2
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
