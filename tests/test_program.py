"""A small program: dependencies ordered by the plan, three months, checkpoint and resume."""
import json

from integrations.corporate_tasks import BY_ID
from integrations.run_program import DEPENDENCIES, run_program

SMALL = [BY_ID[t] for t in ("t29", "t41", "t01", "t02", "t17", "t45")]  # t29 -> t41 -> t01 -> t02


def ordered_by_dependencies(sequence):
    for position, name in enumerate(sequence):
        if name.startswith("run:"):
            _, task, case = name.split(":")
            for dependency in DEPENDENCIES.get(task, ()):
                needed = f"run:{dependency}:{case}"
                assert needed not in sequence or sequence.index(needed) < position, (needed, name)
    return True


def test_program_plans_dependencies_and_resumes_after_interruption(tmp_path):
    database = tmp_path / "shared.sqlite"
    full = run_program(3, 2, tmp_path / "full", database, workstreams=SMALL)
    assert full["status"] == "complete" and len(full["month_reports"]) == 3
    for month in (1, 2, 3):
        plan = json.loads((tmp_path / "full" / "plans" / f"m{month:03d}.json").read_text())
        assert len(plan["sequence"]) == 13 and plan["sequence"][-1] == f"close_month_{month:03d}"
        assert ordered_by_dependencies(plan["sequence"])
    assert all(c["success"] for c in full["cells"]) and len(full["cells"]) == 36
    assert full["integrity_errors"] == [] and all(m["chain_valid"] for m in full["month_reports"])
    # Month 1 meets the drift (two cases: the second already carries the first's
    # evidence); from month 2 on repaired contracts leave nothing to explore.
    first, second, third = full["month_reports"]
    assert first["divergences"] + first["explorations"] > 0
    assert second["divergences"] == 0 and second["explorations"] == 0 and second["served"]["L1r"] > 0
    assert third["divergences"] == 0 and third["explorations"] == 0
    # Step counts are not monotone: a tool nobody has run yet is still trusted on paper.

    # Interrupt after four abstract steps, then resume on a fresh process state.
    database2 = tmp_path / "shared2.sqlite"
    partial = run_program(2, 1, tmp_path / "part", database2, workstreams=SMALL, limit=4)
    assert partial["status"] == "running" and len(partial["cells"]) == 4
    status = json.loads((tmp_path / "part" / "status.json").read_text())
    assert status["month"] == 1 and status["index"] == 4
    resumed = run_program(2, 1, tmp_path / "part", database2, workstreams=SMALL, resume=True)
    assert resumed["status"] == "complete"
    records = [json.loads(l) for l in (tmp_path / "part" / "records.jsonl").read_text().splitlines()]
    plans = [json.loads((tmp_path / "part" / "plans" / f"m{m:03d}.json").read_text())["sequence"] for m in (1, 2)]
    assert [e["action"] for e in records] == plans[0] + plans[1]
    assert len(resumed["cells"]) == 8  # only the unfinished workstreams ran again
