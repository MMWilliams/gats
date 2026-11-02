"""Fifty corporate tasks, two agents, one shared graph."""
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

from integrations.corporate_tasks import TASKS, instance
from integrations.run_corporate import compare, run_agent

ROOT = Path(__file__).resolve().parents[1]


def truly_reachable(system, goal):
    """Breadth-first search over the company's REAL effects, independent of any planner."""
    queue, seen = deque([system.observe()]), {system.observe()}
    while queue:
        state = queue.popleft()
        if goal <= state:
            return True
        for tool in system.tools.values():
            if tool.preconditions <= state:
                after = (state | (tool.effects_add - tool.fails_to_add) | tool.hidden_add) - tool.effects_del - tool.hidden_del
                if after not in seen:
                    seen.add(after)
                    queue.append(after)
    return False


def test_catalog_has_fifty_distinct_tasks_with_verified_reachability():
    assert len(TASKS) == 50
    assert len({t.id for t in TASKS}) == 50 and len({t.title for t in TASKS}) == 50
    assert len({t.department for t in TASKS}) >= 8
    drifted = 0
    for task in TASKS:
        system = task.builder()
        assert truly_reachable(system, task.goal) == task.reachable, task.id
        tools = system.tools.values()
        assert system.human_gated <= set(system.tools), task.id
        if any(t.hidden_add or t.fails_to_add or t.hidden_del for t in tools):
            drifted += 1
            assert task.drift, task.id
        # Every published contract would reach the goal on paper, so failures come from drift.
        assert truly_reachable(task.builder(), task.goal) or task.drift
    assert drifted >= 20
    assert sum(not t.reachable for t in TASKS) == 2
    assert instance(TASKS[0], "c9").observe() == TASKS[0].builder().observe() | {"case:c9"}


def test_two_agents_in_separate_processes_share_one_graph(tmp_path):
    database = tmp_path / "shared.sqlite"
    tasks = ["t01", "t02", "t08", "t14", "t17", "t45"]  # hidden add, silent failure, hidden reset, unreachable

    def run(agent, cases):
        output = tmp_path / f"{agent}.json"
        command = [sys.executable, "-m", "integrations.run_corporate", "run", "--database", str(database),
                   "--agent", agent, "--cases", *cases, "--tasks", *tasks, "--output", str(output)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(output.read_text(encoding="utf-8"))

    first = run("A", ["c1", "c2"])
    second = run("B", ["c1", "c3"])
    assert first["passed"] and second["passed"]
    # A registers each system on its first case and finds its own catalog on the second.
    assert all(r["retrieved_from_graph"] == (r["case"] == "c2") for r in first["rows"])
    assert first["summary"]["divergences"] > 0 and first["summary"]["drift_events"] > 0
    result = compare(first, second)
    assert result["passed"], result["checks"]
    # The unseen case needs repaired contracts, the seen case exact evidence.
    unseen = [r for r in second["rows"] if r["case"] == "c3" and r["success"]]
    assert unseen and sum(r["served"]["L1r"] for r in unseen) > 0
    seen = [r for r in second["rows"] if r["case"] == "c1" and r["success"]]
    assert seen and all(r["served"]["L2"] > 0 for r in seen)
    # A third agent in this process sees both agents' evidence and repeats B exactly.
    third = run_agent(database, "C", ["c3"], tasks)
    assert third["passed"] and third["summary"]["divergences"] == 0
    assert [r["plan"] for r in third["rows"]] == [r["plan"] for r in second["rows"] if r["case"] == "c3"]
