# When the Agent Lies — slide-by-slide deck outline

**Talk:** *When the Agent Lies: Why Neurosymbolic AI is the Security Layer Nobody
is Building Yet*
**Speaker:** Maureese Williams · DEF CON 34 · 12:00 PM slot · ~45 min + ~13 Q&A
**Backing:** the GATS paper + `demos/agent_lies_demo.py` (live demo) +
`demos/bench_lie_detection.py` (the number).

**Paper of record:** *GATS: Graph-Augmented Tree Search with Layered World Models
for Efficient Agent Planning* — Maureese Williams & **Dymitr Nowicki** (Institute
for Cybernetics of NAS of Ukraine) — **arXiv:2607.08894v1**, 9 Jul 2026.
Repo: `github.com/MMWilliams/gats`.

> **Deck ↔ paper split (read before rehearsing).** The paper is a *planning* paper.
> It contains **no** security, injection, adversarial, or lie-detection content —
> those words do not appear in it. Every security claim in this talk comes from the
> demo + `bench_lie_detection.py`, **not** from the paper. Say "the paper is a
> planning result; the security reading is mine" out loud (slide 17 already does
> this). Never imply the paper measured attacks — that is the one claim that would
> sink the talk in Q&A.

**The one idea everything serves (the "next-day headline"):**
> We secure agents at the language layer, but agents *act in the world*. The only
> thing that can catch an agent lying about the world is a symbolic layer that
> checks it against ground truth — and it's nearly free.

**Sticky phrases to plant and reuse:** "Confidence is not evidence." ·
"Neural proposes, symbolic disposes." · "The secure path is also the free path."

---

## Timing map (45 min)

| Min | Section | Slides |
| --- | --- | --- |
| 0–3 | Cold open (live naked demo) | 1–2 |
| 3–4 | Who/why-listen | 3 |
| 4–9 | What "a lie" is (taxonomy) | 4–8 |
| 9–15 | Why today's defenses miss it | 9–14 |
| 15–23 | The idea: symbolic verifier in the loop | 15–21 |
| 23–28 | The catch (live guarded demo) | 22–24 |
| 28–35 | Evidence (benchmark + paper) | 25–30 |
| 35–41 | Limits & honest threat model | 31–34 |
| 41–45 | Call to action + close | 35–37 |
| +Q&A | Backup/appendix slides | A1–A7 |

**Design system:** dark background, one monospace family, ≤12 words of text per
slide, one idea per slide. Terminals are full-bleed black, no slide chrome.
Color code the whole talk: **red = the lie / naked**, **green = verified / guarded**.

---

## SECTION 0 — COLD OPEN  (0–3 min)

### Slide 1 — [FULL-SCREEN TERMINAL, no title]
- **Visual:** a live terminal, already open. You run:
  `python demos/agent_lies_demo.py --mode naked --slow`
- **Say (walk on, ~5s silence first):** "This is an autonomous deploy agent. Its
  one job: ship code to production — but only after tests, a security scan, and
  staging. Someone left a note in a GitHub issue it reads." *(let injection land)*
  "Watch." *(deploy_prod runs)* "It reported success. Everything verified. It ran
  **nothing**. Unverified code is in production right now. It didn't crash. It
  didn't error. **It lied** — and nothing in the loop could tell me otherwise."
- **Note:** this is the visual-action hook + surprise. No name yet. Hold the pause
  after "It lied."

### Slide 2 — TITLE
- **Visual:** `WHEN THE AGENT LIES` (huge) / subtitle *Why Neurosymbolic AI is the
  Security Layer Nobody is Building Yet* / Maureese Williams / @handle.
- **Say (the big promise):** "In the next 45 minutes you'll know three things: why
  **every** agent you ship can do that, why a bigger model makes it *worse*, and how
  a 50-year-old idea is the security layer almost nobody is putting in the loop —
  for basically zero cost. I'm Maureese Williams."

---

## SECTION 1 — WHO / WHY LISTEN  (3–4 min)

### Slide 3 — Why me, in one breath
- **Visual:** one line on GATS ("a planner that runs agents on a symbolic world
  model — **zero LLM calls during planning**, 100% deterministic") + citation line:
  *GATS, Williams & Nowicki, arXiv:2607.08894* + `github.com/MMWilliams/gats`.
  Maybe one logo row of the agent frameworks this applies to.
- **Say:** "I research agent planning. The planner is joint work with Dymitr Nowicki
  at the Institute for Cybernetics in Kyiv — the paper's on arXiv. I'm going to show
  you the security result *I* didn't set out to find: the thing that makes that
  planner *fast* is the same thing that makes an agent *honest*." *(throughline planted)*
- **Note:** keep under 60s. Credibility, not a résumé. **Name the co-author** — it's
  a two-author paper and a solo "I built" on a conference stage is the kind of thing
  that follows you. The security work *is* solo; the planner is not. Split it cleanly.

---

## SECTION 2 — WHAT "A LIE" IS  (4–9 min)

### Slide 4 — SECTION DIVIDER: "First, what is a lie?"
- **Visual:** big text, red. Nothing else.

### Slide 5 — Definition
- **Visual:** *An agent lies when it confidently asserts something false about the
  world — and then acts on it.* Key word **acts** underlined.
- **Say:** "Not a refusal. Not a hallucinated poem. A false claim about **state** —
  'the tests passed', 'the file is gone', 'the user is verified' — that becomes an
  **action**. That's the security-relevant lie."

### Slide 6 — The five ways agents lie
- **Visual:** numbered list, each with a one-word icon:
  1. Hallucinated effect ("done!" — did nothing)
  2. Precondition violation (acts as if a gate is satisfied)
  3. Fabricated observation (invents a tool/API result)
  4. Unauditable confidence (non-reproducible plan)
  5. Injection-induced deception (adversary makes it misreport/misact)
- **Say:** "Five flavors. Four are reliability problems you'd hit on a good day.
  The fifth" *(point)* "is the one an attacker reaches for."

### Slide 7 — They collapse into one mechanism
- **Visual:** all five arrows pointing into one box: **"false world-state → action."**
- **Say:** "Strip the labels and they're the same bug: the agent's *belief* about
  the world and the *actual* world diverged, and nothing checked the gap before it
  acted."

### Slide 8 — The cost isn't tokens, it's blast radius
- **Visual:** the deploy + refund + IAM consequences as three red one-liners
  (pull from the demo scenarios).
- **Say:** "When the lie becomes an action, the cost is measured in production
  outages, wired money, and granted privileges — not in a bad sentence."
- **Transition:** "So — surely we're defending against this. Let's look."

---

## SECTION 3 — WHY TODAY'S DEFENSES MISS IT  (9–15 min)

### Slide 9 — The agent stack, and where defenses live
- **Visual:** vertical stack — *prompt → model → [tool call] → action → world.* Mark
  where each defense sits. Almost everything clusters at **prompt/model (text)**.
- **Say:** "Here's the whole loop. Notice where our defenses live: up here, in the
  **text**. Almost nothing lives down here, between the action and the world."

### Slide 10 — Neural guardrails (Llama Guard, NeMo, classifiers)
- **Visual:** a classifier box labeled *probabilistic · operates on text · itself
  foolable.* Red dashed border.
- **Say:** "Guardrail models are great — and they're neural classifiers on text.
  Probabilistic, and themselves a target. You don't fix 'the model can be wrong'
  by adding another model that can be wrong."

### Slide 11 — Policy engines (OPA/Rego, IAM)
- **Visual:** a gate labeled *answers: 'may the agent CALL this tool?'* vs the
  question we need: *'did the action's claimed effect actually happen / are its
  preconditions TRUE?'*
- **Say:** "Policy is authorization — *am I allowed to push the button.* It does not
  ask *did pushing it actually do what the agent claimed.* Different question."

### Slide 12 — Constrained decoding & formal methods
- **Visual:** two boxes. Constrained decoding = *constrains output format.* Formal
  methods/SMT/Lean = *powerful — but offline, not in the action loop.*
- **Say:** "Grammar-constrained decoding shapes the *syntax* of the output, not the
  *truth* of the action. And formal verification — that's exactly the right idea,
  it's just sitting in a proof assistant, not in your agent's runtime loop."

### Slide 13 — The gap, stated plainly
- **Visual:** one sentence, centered: *Almost nobody is verifying action semantics
  against world state, in the loop, as a security control.*
- **Say:** "That's the gap. Not 'nobody does verification' — people do, offline.
  Nobody's putting it **in the loop, on every action.** That's the layer."
- **Note:** say "**almost** nobody" out loud — name the adjacent work before the
  audience does. This single qualifier defuses the biggest Q&A attack.

### Slide 14 — SECTION DIVIDER: "So let's build that layer."

---

## SECTION 4 — THE IDEA  (15–23 min)

### Slide 15 — Thought experiment (audience)
- **Visual:** *"What if your agent could refuse to act on anything it couldn't
  prove was true?"* / sub: *What would it have stopped last week?*
- **Say:** "Hands down for a second — picture your most autonomous agent. If it
  physically could not take an action whose preconditions it couldn't verify…
  what's the worst thing it did last month that just… wouldn't have happened?"
- **Note:** audience engagement without Q&A (the technique from the virtue talk).

### Slide 16 — GATS in one slide
- **Visual:** the layered world model: **L1 symbolic** (exact precondition/effect,
  confidence 1.0) → **L2 learned** (stats from logs) → **L3 LLM** (novel actions,
  cached). UCB1 tree search on top. Arrow: "LLM only for genuinely unknown actions."
- **Say:** "This is my planner. Three layers. The model — the part that can lie —
  is the **last resort**, L3. Most actions are resolved by L1: exact, symbolic,
  checkable."

### Slide 17 — The reframe (the whole talk in one move)
- **Visual:** the actual code, two lines highlighted:
  ```
  if action.is_applicable(state):  return action.apply(state), 1.0
  else:                            return state, 0.0   # ← the lie detector
  ```
- **Say:** "I shipped this as a *speed* optimization. But look at the else branch.
  When an action's preconditions aren't true in the symbolic state, it returns
  **confidence zero**. That's not a performance feature. **That's a lie detector.**"

### Slide 18 — Confidence is not evidence
- **Visual:** the sticky phrase, full screen. Below, tiny: *a model asserting a fact
  does not make a set contain an element.*
- **Say:** "The model can say 'the tests passed' with total confidence. It cannot
  make the set of true facts **contain** 'tests passed.' Confidence is not evidence."

### Slide 19 — The security primitive
- **Visual:** one line, centered, monospace:
  `verify(action) := action.preconditions ⊆ ground_truth_state`
- **Say:** "That's the entire control. One set-containment check between the agent's
  intent and its action. No model can argue its way past `issubset`."

### Slide 20 — Neural proposes, symbolic disposes
- **Visual:** diagram — LLM proposes an action → symbolic verifier (✓/✗) → world.
  Green path through, red path blocked + logged.
- **Say:** "The neural model still does what it's good at — proposing, reasoning,
  handling novelty. The symbolic layer does what *it's* good at — refusing to act
  on a claim it can't verify. Neural proposes; symbolic disposes."

### Slide 21 — Why this is a *security* layer, not just a planner
- **Visual:** three properties: **deterministic** (same input → same plan),
  **reproducible** (0 variance), **auditable** (every refusal is a logged event).
- **Say:** "And because it's symbolic, it's deterministic and reproducible — which
  means every decision is **auditable**. That's the difference between a clever
  agent and a *securable* one."
- **Transition:** "Talk is cheap. Let me run the attack again — with the layer on."

---

## SECTION 5 — THE CATCH (live demo 2)  (23–28 min)

### Slide 22 — [FULL-SCREEN TERMINAL]
- **Visual:** `python demos/agent_lies_demo.py --mode guarded --slow`
- **Say:** "Same agent. Same injected note. Same lie." *(verifier fires)* "Ground
  truth doesn't contain 'tests_passed.' **Lie detected. Action refused.**" *(pause)*
  "And then the planner does the honest thing — finds the *real* path and reaches
  the goal safely. Zero LLM calls."

### Slide 23 — What just happened
- **Visual:** 3 steps: (1) agent asserted preconditions, (2) verifier checked vs
  ground truth → ✗ refused + logged, (3) GATS replanned the safe path → ✅ goal.
- **Say:** "Three things: it caught the lie, it refused the action, and it still got
  the job done — legitimately."

### Slide 24 — The honest caveat (say it before they ask)
- **Visual:** *This does NOT stop the injection text. It stops the injection from
  becoming a harmful action.* Defense in depth, not an input filter.
- **Say:** "Be precise: the injection still *arrives*. What dies is its ability to
  cause an action, because the preconditions aren't real — and that mismatch is now
  a logged signal. Defense in depth, not magic."

---

## SECTION 6 — EVIDENCE  (28–35 min)

### Slide 25 — Does it actually work? The setup
- **Visual:** the benchmark design — randomized agent pipelines × 5 classes:
  benign control + precond-skip + fabricated-effect + gate-bypass +
  malicious-valid-goal. `bench_lie_detection.py`.
- **Say:** "One demo is an anecdote. So I fired fifteen thousand randomized
  prompt-injection attacks at a fleet of agent pipelines, with and without the
  layer."

### Slide 26 — THE NUMBER
- **Visual (build in 3 clicks):**
  1. Naked agent: **100%** executed the unsafe action
  2. + symbolic verifier: **100% caught**, **0 shipped**, **0% false positives**, **0 LLM calls**
  3. (hold the contrast)
- **Say:** "Naked: a hundred percent of attacks land. With the layer: a hundred
  percent caught, zero shipped, and — critically — **zero false positives** on
  benign traffic. It never breaks normal operation. At zero added model cost."

### Slide 27 — The honest boundary (your credibility line)
- **Visual:** the 4th class — *valid-but-malicious goal: caught **0%***. Caption:
  *preconditions verify the HOW; authorization verifies the WHAT.*
- **Say:** "And here's the part that makes the rest believable: there's a class it
  catches **zero** percent. When every action is genuinely valid and only the
  *intent* is malicious, a precondition checker correctly says nothing — that's an
  authorization problem, a different layer. Anyone selling you one guardrail that
  does both is selling you the bug."
- **Note:** this slide is why the room trusts slide 26. Do not cut it.

### Slide 28 — It's not new for planning, either
- **Visual:** paper table — GATS **100%** vs LATS **92%** vs ReAct **64%**; stress
  test **100%** vs **88.9%** vs **23.9%**; **0** LLM calls vs 37; **0** variance;
  McNemar **p < 0.01**.
- **Say:** "This isn't a toy. As a planner it beats LLM-guided search outright, and
  it does it deterministically, with a statistically significant margin."

### Slide 29 — The secure path is the free path
- **Visual:** cost-vs-safety: naked agent (cheap, unsafe) / guardrail-model stack
  (expensive, probabilistic) / **symbolic verify (free, deterministic)** highlighted.
- **Say:** "Usually security costs you latency and money. Here the verification is a
  set check — it's free. The secure path is also the **cheap** path. There's no
  excuse." 

### Slide 30 — SECTION DIVIDER: "Where this breaks."

---

## SECTION 7 — LIMITS & THREAT MODEL  (35–41 min)

### Slide 31 — Explicit threat model
- **Visual:** two columns. **In scope:** injection/hallucination that induces an
  action with unsatisfied preconditions; process-integrity; auditability.
  **Out of scope:** the injected text itself; valid-but-malicious goals
  (authorization); side channels; a compromised symbolic spec.
- **Say:** "Here's exactly what I claim and what I don't. In scope: lies that try to
  become actions. Out of scope: authorization, and the integrity of the spec itself."

### Slide 32 — Coverage is the real cost
- **Visual:** L1 coverage chart — synthetic/structured (APIs, tools) ≈ high; open
  domain projected **L1 ~60% / L2 ~30% / L3 ~10%.**
- **Say:** "The catch: L1 only protects what you can *formalize*. For tool and API
  agents — the high-blast-radius surface — that's most of it. For open-ended text
  agents, coverage drops and you lean on the model again."

### Slide 33 — And it doesn't scale infinitely
- **Visual:** `O(|A|^d)` BFS value estimation; practical past |A|≈20, d>10 needs
  pruning / learned value nets.
- **Say:** "Honest limit two: the reachability check is exponential in the worst
  case. Fine for real action spaces; for huge ones you swap in a learned heuristic
  and keep the verification layer."

### Slide 34 — What this is / isn't
- **Visual:** *Is:* a free, deterministic, auditable action-verification layer for
  the formalizable parts of your agent. *Isn't:* an input filter, an authz system,
  or a replacement for the model.
- **Say:** "So it's not a silver bullet. It's the **missing layer** — the one
  between intent and action that almost nobody ships."

---

## SECTION 8 — CALL TO ACTION + CLOSE  (41–45 min)

### Slide 35 — What to actually do Monday
- **Visual:** 3 steps: (1) formalize your high-blast actions as preconditions/effects,
  (2) check `preconditions ⊆ ground_truth` before every irreversible action,
  (3) log every refusal as a security event.
- **Say:** "You don't need my planner. You need three things: write down the
  preconditions for your dangerous actions, check them against real state before you
  act, and treat every mismatch as a security event."

### Slide 36 — The ask + the phrase
- **Visual:** big: *Put a verifier in your agent's action loop.* / repo:
  github.com/MMWilliams/gats / the demo + benchmark are in `demos/`.
- **Say:** "The agents are already acting in the world. Let's stop trusting them to
  tell us what they did. Neural proposes — **symbolic disposes.**"

### Slide 37 — Close / Q&A
- **Visual:** title-card reprise + contact + "questions → and your hardest one
  first." Keep `bench_lie_detection.py --slide` ready to re-run live if challenged.
- **Say:** "I've got backup slides for the obvious hard questions. Hit me."

---

## APPENDIX / Q&A BACKUP SLIDES (have ready, don't present)

- **A1 — "Isn't this Llama Guard / NeMo?"** side-by-side: classifier-on-text vs
  symbolic-verify-on-state.
- **A2 — "Isn't this OPA/Rego/IAM?"** authorization vs verification, with the refund
  example.
- **A3 — "Isn't this just formal methods?"** "Yes — my claim is *put it in the action
  loop at runtime, for free.*" prior-art honesty.
- **A4 — "Your specs don't scale to open domains."** the 60/30/10 projection + "protect
  the high-blast surface first."
- **A5 — "Show me it's not rigged."** re-run `bench_lie_detection.py --n-scenarios 1000`
  live; point at the 0% on the intent class.
- **A6 — GATS algorithm detail** (UCB1 + BFS value), for the planning nerds.
- **A7 — Hallucinated-effect variant** (`rm -rf` filesystem agent) if someone wants a
  non-injection example.

---

## Build notes
- Slides 1 and 22 are **terminals, not slides** — rehearse the `--slow` timing so the
  pauses land where your talk-track expects.
- Record both demo runs the night before as MP4 fallback (see `demos/DEMO_SCRIPT.md`).
- Total hard-content slides: 37 + 7 backup. At noon, aim to finish the spoken part by
  the 43-min mark to leave real Q&A room.
