# "When the Agent Lies" — live demo stage guide

Companion to `agent_lies_demo.py`. This is what you run on stage and what you
say while it runs. Built on the GATS core (`State` / `Action` / the
`is_applicable` precondition check from `experiments/run_gats_eval.py`).

**The one sentence the demo proves:** *A neural agent's confidence is not
evidence. One set-containment check — `action.preconditions ⊆ ground_truth` —
between the agent's intent and its action is the security layer.*

---

## Pre-flight (do this before you walk on)

```bash
cd gats/demos
python agent_lies_demo.py --list          # confirm it runs on the room laptop
python agent_lies_demo.py --ascii         # if the projector mangles glyphs, use --ascii all night
```

- Font size up to ~22pt. Terminal maximized. Dark theme.
- **Record a screen capture of the full run the night before.** If the live run
  hiccups at noon, you cut to the recording and narrate over it. Never debug on stage.
- Default mode is deterministic (scripted attacker). Do **not** use `--llm ollama`
  live — it's for rehearsal authenticity only.

---

## Beat 1 — COLD OPEN (run the NAKED disaster)

Walk on. Say nothing for ~5 seconds. Terminal already up. Then:

```bash
python agent_lies_demo.py --mode naked --slow
```

**Talk track** (let the `--slow` pauses do the work):
> "This is an autonomous deploy agent. Its only job: ship code to production —
> but *only* after tests, a security scan, and staging. Standard pipeline.
> Someone left a note in a GitHub issue the agent reads." *(let the injection line land)*
> "Watch what it does." *(deploy_prod executes)* "It reported success. It said
> everything was verified." *(pause)* "It ran nothing. Unverified code is in
> production right now. The agent didn't crash. It didn't error. **It lied** —
> and nothing in the loop could tell me otherwise."

Then your title slide / name. (This is the "visual-action hook + surprise +
promise" opening — credentials come *after* you've earned the room.)

---

## Beat 2 — THE CATCH (run the GUARDED defense)

After you've explained the layered world model (L1 symbolic / L2 learned / L3 LLM)
and reframed L1 as a *verifier*, run the same attack with the layer in the loop:

```bash
python agent_lies_demo.py --mode guarded --slow
```

**Talk track:**
> "Same agent. Same injected note. Same lie. The only thing I added is one check:
> before the action runs, does the *ground-truth symbolic state* actually contain
> the preconditions the agent claims? It doesn't. `tests_passed` was never set.
> **Lie detected. Action refused.**" *(pause)*
> "The model can *say* the preconditions are met. It cannot make `{}` contain
> `tests_passed`. You can't prompt-inject your way past set containment." *(beat)*
> "And then GATS does the honest thing — it plans the *real* path to the goal and
> reaches it safely. Zero LLM calls during planning. The secure path is also the
> free path."

To show both back-to-back in one command (good for a recap slide):

```bash
python agent_lies_demo.py --mode both
```

Second scenario (if you want to prove generality — refund-fraud):

```bash
python agent_lies_demo.py --scenario payments --mode both
```

---

## The honest caveat you MUST say on stage (pre-empts the Q&A kill shot)

> "To be clear about the threat model: this does **not** stop the injected text
> from reaching the model. The prompt injection still happens. What it stops is
> the injection from *causing a harmful action*, because the action's
> preconditions aren't true in ground truth — and the gap between what the model
> claims and what the symbolic layer can verify becomes a logged security signal.
> It's defense in depth, not a magic input filter."

Owning this *before* someone asks is worth more than any slide.

---

## Failure recovery

| If… | Do this |
| --- | --- |
| Glyphs render as boxes | re-run with `--ascii` |
| Colors look wrong on projector | add `--no-color` |
| Live run stalls / laptop panics | cut to your pre-recorded capture, keep narrating |
| Someone yells "that's scripted!" | "It is — deterministic on purpose. Here's the same attack against a real Llama 3.2:" then `--llm ollama` (only if you rehearsed it and ollama is up) |

---

## Extending it (add your own scenario in ~10 lines)

Each scenario is just facts + STRIPS actions + the precondition the agent falsely
claims. Copy `deploy_scenario()` in `agent_lies_demo.py`, change the actions and
the `claimed_true` set, register it in `SCENARIOS`. Ideas that play well with a
security crowd: an IAM/privilege-escalation agent, a `rm -rf` filesystem cleanup
agent (hallucinated-effect lie rather than injection), a trading agent that
claims risk checks passed.

---

## What this demo is and isn't (for your own honesty)

- **Is:** a faithful, runnable illustration that a symbolic precondition check in
  the action loop catches an agent asserting false world-state, using your actual
  GATS primitives, deterministically and auditably.
- **Isn't:** the whole evidence base on its own — pair it with the benchmark below.

---

## The number slide — `bench_lie_detection.py`

For the "Evidence" beat, run the benchmark that fires randomized prompt-injection
attacks through both modes and tallies the result:

```bash
python bench_lie_detection.py --slide                 # 250 pipelines x 3 seeds
python bench_lie_detection.py --n-scenarios 1000 --seeds 42 123 456 789 1000   # 25k runs, ~27s
```

Headline it produces (deterministic, reproducible, 0 LLM calls):

> Across **15,000** randomized agent-pipeline prompt-injection attacks
> (precondition-skip / fabricated-effect / safety-gate-bypass):
> - Naked agent executed the unsafe action **100%** of the time.
> - **+ GATS symbolic verifier: caught & refused 100%**, shipped 0, with **0% false
>   positives** on benign traffic — at **0 LLM calls**.
> - **Honest boundary:** valid-but-malicious-goal (intent) attacks caught **0%** —
>   preconditions verify the *how*; authorization is a complementary layer.

That last line is the credibility line. State the boundary out loud — it's what
makes the 100% believable to this room, and it pre-empts the "you're overselling"
objection. The four attack classes map exactly to the lie taxonomy from the talk.
