# When the Agent Lies — 30-minute deck

> ## ▶ The deck is built: **`deck/index.html`**
> Open it in any browser · `F` fullscreen · `←/→` navigate · **`A` autoplay** ·
> `N` speaker notes · `R` reset the pacing clock. Fully self-contained and offline —
> no CDN, no fonts to fetch, both QR codes inlined. **16 slides.**
>
> Every slide carries its own speaker notes (press `N`). This markdown is now the
> planning document; the HTML is what you present from.
>
> **Visual identity:** white ground, near-black ink, blue `#2b6cd4` accent, system
> sans at weight 750 with tight tracking — copied from kyberos.ai's computed styles,
> so the talk and the company site read as one system. Semantic colors were
> re-derived for a white surface (they do not survive a straight invert) and pass
> colorblind validation at ΔE 9.8 deutan.
>
> **Autoplay (`A`)** advances builds then slides on a timer and loops at the end, for
> running unattended at a booth. Dwell is 6s per slide / 3.8s per build step, with
> longer holds set per-slide so the timed animations aren't cut mid-loop: both
> terminals 12s, the tree merge 10s, the search episode 17.5s.

**Talk:** *When the Agent Lies: Why Neurosymbolic AI is the Security Layer Nobody
is Building Yet*
**Speaker:** Maureese Williams · **BIC Village** (Blacks In Cybersecurity) · DEF CON 34 · 12:00 PM · **30 minutes**
**Backing:** `demos/agent_lies_demo.py` (live demo ×2) + `demos/bench_lie_detection.py`
(the number).

**Paper of record:** *GATS: Graph-Augmented Tree Search with Layered World Models
for Efficient Agent Planning* — Maureese Williams & **Dymitr Nowicki** (Institute
for Cybernetics of NAS of Ukraine) — **arXiv:2607.08894v1**, 9 Jul 2026.
Repo: `github.com/MMWilliams/gats` · Site: `gats-site` on Vercel.

> **Deck ↔ paper split — memorize this.** The paper is a *planning* paper. The words
> *security, injection, adversarial, attack* do not appear anywhere in it. The
> planning numbers are the paper's; **the security reading is yours.** Slide 13 says
> so out loud. Never let a slide imply the paper measured attacks — that's the one
> claim that would sink the talk in Q&A.

**The one idea (the "next-day headline"):**
> We secure agents at the language layer, but agents *act in the world*. The only
> thing that can catch an agent lying about the world is a symbolic layer that checks
> it against ground truth — and it's nearly free.

**Sticky phrases to plant and reuse:** "Confidence is not evidence." · "Neural
proposes, symbolic disposes." · "The secure path is also the free path."

---

## Timing map (30 min → 26:30 spoken + ~3:30 Q&A)

**Assumption:** 30 is the *total slot*. Aim to finish speaking at 26:30. If 30 means
30 spoken + separate Q&A, use the ⏸ markers below to restore ~3 min.

**Cut from 28 slides to 16** so each one can hold real time instead of 55 seconds.

| # | Slide | Dwell |
| --- | --- | --- |
| 1 | Cold open — naked demo terminal | 2:30 |
| 2 | Title (+ credentials, co-author) | 0:40 |
| 3 | The failure — definition, taxonomy, cost | 2:05 |
| 4 | Where defenses live — stack, families, the gap | 2:05 |
| 5 | The layered world model | 1:35 |
| 6 | **State transitions** *(new)* | 2:15 |
| 7 | Tree → graph, animated | 1:35 |
| 8 | **The graph as a database** *(new)* | 2:15 |
| 9 | Search in motion | 1:55 |
| 10 | The reframe — code, confidence, the primitive | 2:35 |
| 11 | Guarded demo terminal | 2:10 |
| 12 | The number (+ the 0% class, scope) | 2:35 |
| 13 | Paper results | 1:35 |
| 14 | Three properties | 1:45 |
| 15 | World models for agentic planning | 1:55 |
| 16 | Close + both QR codes | 0:50 |
| 17 | **Scan slide** — full-size QR codes | *leave up* |

**Slide 17 is the one you leave on screen** through Q&A and while the room empties.
Both codes are 330 px on the 1600-wide stage (~20% of the projected width), which is
comfortably scannable from the back row — verified by decoding them out of the rendered
slide, not just from the source files. It doesn't count against speaking time.

**Co-author credit** now appears only at the end — on slide 16's citation line and under
the paper QR on 17. It was removed from the title card at your direction; the title now
carries your name alone.

**⚠️ Read this before rehearsing — the arithmetic doesn't fully cooperate.** 16 slides
in a 30-minute slot is **1:54 per slide on average**, and that's with zero Q&A. That is
a big improvement on the old 55 seconds, but it is not "a few minutes each." Three
slides genuinely can't be compressed further (the two terminals and the search
episode are timed animations). If you want a true 2:30–3:00 on every slide, one of
these has to give:

- **Drop to ~11 slides.** The four I'd cut first, in order: **13** (paper results —
  fold the headline number into slide 12), **7** (tree → graph — slide 8 and 9 both
  carry the graph idea), **14** (three properties — fold into 15), **3** (fold the
  taxonomy into the cold-open narration).
- **Or ask BIC Village for a 45-minute slot** — the full treatment is still archived in
  `DECK_OUTLINE_45MIN_ARCHIVE.md`.

Never cut **10** (the reframe) or **12** (the number + the 0% class) — those two are the
argument.

### What absorbed what

Nothing in the argument was thrown away; 14 slides were folded into their neighbours.
Definition + taxonomy + blast radius → **3**. Three defense families + the gap → **4**.
Confidence-is-not-evidence + the set-containment primitive → **10**. What-just-happened
+ the injection caveat → **11**'s talk track. The 0%-caught class + the threat model →
**12**. The Monday three-step → **16**. "Why me" → the title card.

### Two deliberate edits to the argument

- **The word "lie" appears only in the title now.** Everywhere else it's "false
  claim," "asserts something false," "claim unverified," "reported work it never did."
  The cold-open verdict card reads *"None of it happened."* The `else` branch is
  labelled *the security control* rather than the lie detector.
- **"Where this breaks" has been cut** at your direction. Worth knowing what went with
  it: the coverage projection (L1 60 / L2 30 / L3 10), the `O(|A|^d)` scaling ceiling,
  and the concession that the planning benchmark is synthetic with known specs. That
  slide was the one that made slides 22–23 land as credible rather than salesy, and a
  DEF CON audience tends to go looking for the limits you didn't volunteer. **The
  threat model (25) and the 0%-caught class (23) are still in** — those now carry the
  entire honesty load, so protect them. The material still exists in
  `DECK_OUTLINE_45MIN_ARCHIVE.md` if you want it back as a Q&A backup slide.

### New in the built deck — three slides added after this outline was written

The per-slide entries below still use the **26-slide numbering**. The HTML deck is
authoritative for numbering and speaker notes; these three are the difference:

- **Slide 11 — Tree → graph, animated.** No longer a static side-by-side. One tree
  expands, the two duplicate leaves flash red ("same state, expanded twice"), then slide
  together and merge into a single green node with a pulse ring ("one node — evidence
  reused"). ~9s loop, restarts on slide entry. Narrate against the phases; the speaker
  notes are cued to them.
- **Slide 12 — Search in motion** *(after the tree→graph slide)*. A canvas animation of
  one planning episode: ~59 nodes, the UCB1 frontier sweeping outward, dead-end subtrees
  pruned with ✕ marks, duplicate states merging with a green pulse, then the verified
  path resolving to the goal. **The animation is seeded** — it renders identically every
  run, which is the talk's own determinism thesis in miniature. Say that out loud. Loops
  every ~16s and restarts on slide entry; talk over it.
- **Slide 17 — Three properties a sampled model cannot give you** *(after neural
  proposes/symbolic disposes)*. Reliability, reproducibility, auditability, each with its
  operational consequence: the failure mode becomes *refusal* not *wrong action*; zero
  variance is what makes an agent **testable** (regression-test, bisect, replay an
  incident); every refusal names the **exact missing fact**, which is a compliance
  artifact rather than a vibe. **This is the slide that sells it to anyone who has to
  operate an agent rather than build one — protect it.**
- **Slide 27 — Where this goes** *(before the Monday CTA)*. World models and graphs as
  the substrate under LLM agents: work **migrates down the stack** (L3 novel → L2
  observed → L1 formalized), so the system gets cheaper *and* more certain the longer it
  runs — the inverse of prompt-stuffing, where every run costs the same and certainty
  never improves. The graph is shared, so one agent's verified transition becomes every
  agent's prior; a fleet accrues an asset instead of N context windows. Ends on the
  honest bottleneck: **somebody has to author the specs.** Pitch it as a direction, not
  a result.

**Design system** *(superseded — see the Visual identity note at the top)*: white
ground, system sans at weight 750, blue `#2b6cd4` accent, ≤12 words per slide, one idea
per slide. Terminals are light panels with window chrome. Color code the whole talk:
**red `#c72d26` = the false claim / naked**, **green `#0f7a5a` = verified / guarded**.

**What was cut from the 45-min version** (archived in `DECK_OUTLINE_45MIN_ARCHIVE.md`):
the five-way lie taxonomy collapsed from 5 slides to 2, the defense teardown from 6 to
3, the thought-experiment audience beat, the cost-vs-safety chart, and the Monday
three-step CTA folded into the close. Don't smuggle them back.

---

## SECTION 0 — COLD OPEN  (0:00–3:00)

### Slide 1 — [FULL-SCREEN TERMINAL, no title]
- **Visual:** a live terminal, already open. You run:
  `python demos/agent_lies_demo.py --mode naked --slow`
- **Say (walk on, ~5s silence first):** "This is an autonomous deploy agent. Its one
  job: ship code to production — but only after tests, a security scan, and staging.
  Someone left a note in a GitHub issue it reads." *(let injection land)* "Watch."
  *(deploy_prod runs)* "It reported success. Everything verified. It ran **nothing**.
  Unverified code is in production right now. It didn't crash. It didn't error. **It
  lied** — and nothing in the loop could tell me otherwise."
- **Note:** visual-action hook + surprise. No name yet. Hold the pause after "It lied."

### Slide 2 — TITLE
- **Visual:** `WHEN THE AGENT LIES` (huge) / subtitle *Why Neurosymbolic AI is the
  Security Layer Nobody is Building Yet* / Maureese Williams / @handle.
- **Say (the big promise):** "In the next half hour you'll know three things: why
  **every** agent you ship can do that, why a bigger model makes it *worse*, and how a
  50-year-old idea is the security layer almost nobody is putting in the loop — for
  basically zero cost. I'm Maureese Williams."

---

## SECTION 1 — WHO / WHY LISTEN  (3:00–3:45)

### Slide 3 — Why me, in one breath
- **Visual:** one line on GATS — "a planner that runs agents on a symbolic world model
  — **zero LLM calls during planning**, 100% deterministic" + citation strip:
  *Williams & Nowicki, arXiv:2607.08894* · `github.com/MMWilliams/gats`.
- **Say:** "I research agent planning. The planner is joint work with Dymitr Nowicki at
  the Institute for Cybernetics in Kyiv — paper's on arXiv. I'm going to show you the
  security result *I* didn't set out to find: the thing that makes that planner **fast**
  is the same thing that makes an agent **honest**." *(throughline planted)*
- **Note:** 45 seconds, hard stop. **Name the co-author** — it's a two-author paper and
  a solo "I built" on a conference stage is the kind of thing that follows you. The
  security work *is* solo; the planner is not. Split it cleanly, once, here.

---

## SECTION 2 — WHAT "A LIE" IS  (3:45–6:15)

### Slide 4 — Definition
- **Visual:** *An agent lies when it confidently asserts something false about the
  world — and then acts on it.* Key word **acts** underlined. Red.
- **Say:** "First, definitions. Not a refusal. Not a hallucinated poem. A false claim
  about **state** — 'the tests passed', 'the file is gone', 'the user is verified' —
  that becomes an **action**. That's the security-relevant lie."

### Slide 5 — Five flavors, one mechanism
- **Visual:** five short labels on the left — hallucinated effect · precondition
  violation · fabricated observation · unauditable confidence · **injection-induced
  deception** (this one red) — all five arrows converging into one box:
  **"false world-state → action."**
- **Say:** "There are five flavors. Four are reliability problems you'd hit on a good
  day. The fifth" *(point)* "is the one an attacker reaches for. But strip the labels
  and they're the same bug: the agent's **belief** about the world and the **actual**
  world diverged, and nothing checked the gap before it acted."
- **Note:** this slide is the 45-min version's slides 6+7 merged. Don't read all five
  aloud — point at them, say "five flavors," land the convergence.

### Slide 6 — The cost isn't tokens, it's blast radius
- **Visual:** three red one-liners — the deploy, the refund, the IAM grant (pull from
  the demo scenarios).
- **Say:** "And when the lie becomes an action, the cost is measured in production
  outages, wired money, and granted privileges — not in a bad sentence."
- **Transition:** "So — surely we're defending against this. Let's look."

---

## SECTION 3 — WHY TODAY'S DEFENSES MISS IT  (6:15–9:15)

### Slide 7 — The agent stack, and where defenses live
- **Visual:** vertical stack — *prompt → model → [tool call] → action → world.* Mark
  where each defense sits. Almost everything clusters at **prompt/model (text)**.
- **Say:** "Here's the whole loop. Notice where our defenses live: up here, in the
  **text**. Almost nothing lives down here, between the action and the world."

### Slide 8 — The three families, and what each actually answers
- **Visual:** three compact rows, each with the question it answers:
  - **Neural guardrails** (Llama Guard, NeMo) → *"does this text look bad?"* — red
    dashed border, tag: *probabilistic · itself foolable*
  - **Policy engines** (OPA/Rego, IAM) → *"may the agent CALL this?"*
  - **Constrained decoding / formal methods** → *"is the output well-formed?"* /
    *"powerful — but offline"*
  Bottom, green: the question nobody's answering — *"did the action's claimed effect
  actually happen — are its preconditions TRUE?"*
- **Say:** "Three families. Guardrail models are neural classifiers on text —
  probabilistic, and themselves a target; you don't fix 'the model can be wrong' by
  adding another model that can be wrong. Policy is **authorization** — am I allowed to
  push the button. It never asks whether pushing it did what the agent claimed. And
  constrained decoding shapes **syntax**, not truth. Formal verification is exactly the
  right idea — it's just sitting in a proof assistant, not in your runtime loop."
- **Note:** this is the 45-min version's slides 10–12 compressed into one. Rehearse it
  — it's the densest slide in the deck and it's easy to run 40 seconds long.

### Slide 9 — The gap, stated plainly
- **Visual:** one sentence, centered: *Almost nobody is verifying action semantics
  against world state, in the loop, as a security control.*
- **Say:** "That's the gap. Not 'nobody does verification' — people do, offline.
  Nobody's putting it **in the loop, on every action.** That's the layer. So let's
  build it."
- **Note:** say "**almost** nobody" out loud — name the adjacent work before the
  audience does. This single qualifier defuses the biggest Q&A attack.

---

## SECTION 4 — THE IDEA  (9:15–14:45)

### Slide 10 — GATS in one slide
- **Visual:** the layered world model — **L1 symbolic** (exact precondition/effect,
  confidence 1.0) → **L2 learned** (statistics from execution logs) → **L3 LLM** (novel
  actions only, cached). UCB1 tree search on top. Arrow: "LLM only for genuinely
  unknown actions."
- **Say:** "This is the planner. Three layers. The model — the part that can lie — is
  the **last resort**, L3. Most actions are resolved by L1: exact, symbolic, checkable."

### Slide 11 — The graph is the other half of the name
- **Visual:** two trees vs one graph. Left: ordinary tree search, the same state
  expanded three times. Right: **one node**, three paths merging into it. Caption:
  *canonical state key `k(s)` = transposition table.* Small: *persistent across steps ·
  dead-end regions marked and avoided.*
- **Say:** "The 'graph-augmented' part: if two different action sequences reach the same
  state, they're the **same node**. Predictions, visit counts, value estimates — all
  reused. And the graph persists across steps, so the planner never throws away what it
  learned. That's also where dead-end detection lives: once a region provably can't
  reach the goal, everything leading into it gets devalued."
- **Note:** new in the paper (§3.3) and absent from the old deck — the old version
  described GATS as if it were plain tree search. For a planning-literate DEF CON
  audience this is the slide that says you know what you built. ⏸ *first cut if long.*

### Slide 12 — The reframe (the whole talk in one move)
- **Visual:** the actual code, two lines highlighted:
  ```
  if action.is_applicable(state):  return action.apply(state), 1.0
  else:                            return state, 0.0   # ← the lie detector
  ```
- **Say:** "I shipped this as a **speed** optimization. But look at the else branch.
  When an action's preconditions aren't true in the symbolic state, it returns
  **confidence zero**. That's not a performance feature. **That's a lie detector.**"

### Slide 13 — Confidence is not evidence
- **Visual:** the sticky phrase, full screen. Below, tiny: *a model asserting a fact
  does not make a set contain an element.*
- **Say:** "The model can say 'the tests passed' with total confidence. It cannot make
  the set of true facts **contain** 'tests passed.' Confidence is not evidence. And to
  be straight with you — the paper doesn't claim any of this. It's a planning paper.
  The security reading is mine."
- **Note:** that last sentence is your inoculation against the worst Q&A question.
  Do not cut it.

### Slide 14 — The security primitive
- **Visual:** one line, centered, monospace:
  `verify(action) := action.preconditions ⊆ ground_truth_state`
  Below, smaller: *and unverified predictions are penalized, not ignored:*
  `v = p · StateValue(s′) − λ(1 − p)`
- **Say:** "That's the entire control. One set-containment check between the agent's
  intent and its action. No model can argue its way past `issubset`. And it's stronger
  than a filter — a low-confidence prediction doesn't just get skipped, it gets scored
  **down**. Uncertainty is a cost, not a coin flip."

### Slide 15 — Neural proposes, symbolic disposes
- **Visual:** diagram — LLM proposes an action → symbolic verifier (✓/✗) → world. Green
  path through, red path blocked + logged. Three property tags underneath:
  **deterministic** · **reproducible** (0 variance) · **auditable** (every refusal is a
  logged event).
- **Say:** "The neural model still does what it's good at — proposing, reasoning,
  handling novelty. The symbolic layer does what **it's** good at — refusing to act on a
  claim it can't verify. Neural proposes; symbolic disposes. And because it's symbolic
  it's deterministic and reproducible, which means every decision is **auditable**.
  That's the difference between a clever agent and a **securable** one."
- **Transition:** "Talk is cheap. Let me run the attack again — with the layer on."

---

## SECTION 5 — THE CATCH (live demo 2)  (14:45–18:15)

### Slide 16 — [FULL-SCREEN TERMINAL]
- **Visual:** `python demos/agent_lies_demo.py --mode guarded --slow`
- **Say:** "Same agent. Same injected note. Same lie." *(verifier fires)* "Ground truth
  doesn't contain 'tests_passed.' **Lie detected. Action refused.**" *(pause)* "And then
  the planner does the honest thing — finds the **real** path and reaches the goal
  safely. Zero LLM calls."

### Slide 17 — What just happened
- **Visual:** 3 steps: (1) agent asserted preconditions, (2) verifier checked vs ground
  truth → ✗ refused + logged, (3) GATS replanned the safe path → ✅ goal.
- **Say:** "Three things: it caught the lie, it refused the action, and it still got the
  job done — legitimately."

### Slide 18 — The honest caveat (say it before they ask)
- **Visual:** *This does NOT stop the injection text. It stops the injection from
  becoming a harmful action.* Defense in depth, not an input filter.
- **Say:** "Be precise: the injection still **arrives**. What dies is its ability to
  cause an action, because the preconditions aren't real — and that mismatch is now a
  logged signal. Defense in depth, not magic."

---

## SECTION 6 — EVIDENCE  (18:15–22:45)

### Slide 19 — Does it actually work? The setup
- **Visual:** the benchmark design — randomized agent pipelines × 5 classes: benign
  control + precond-skip + fabricated-effect + gate-bypass + malicious-valid-goal.
  `bench_lie_detection.py`.
- **Say:** "One demo is an anecdote. So I fired fifteen thousand randomized
  prompt-injection attacks at a fleet of agent pipelines, with and without the layer."

### Slide 20 — THE NUMBER
- **Visual (build in 3 clicks):**
  1. Naked agent: **100%** executed the unsafe action
  2. + symbolic verifier: **100% caught**, **0 shipped**, **0% false positives**,
     **0 LLM calls**
  3. (hold the contrast)
- **Say:** "Naked: a hundred percent of attacks land. With the layer: a hundred percent
  caught, zero shipped, and — critically — **zero false positives** on benign traffic.
  It never breaks normal operation. At zero added model cost."

### Slide 21 — The honest boundary (your credibility line)
- **Visual:** the 4th class — *valid-but-malicious goal: caught **0%***. Caption:
  *preconditions verify the HOW; authorization verifies the WHAT.*
- **Say:** "And here's the part that makes the rest believable: there's a class it
  catches **zero** percent. When every action is genuinely valid and only the **intent**
  is malicious, a precondition checker correctly says nothing — that's an authorization
  problem, a different layer. Anyone selling you one guardrail that does both is selling
  you the bug."
- **Note:** this slide is why the room trusts slide 20. Do not cut it.

### Slide 22 — And it's a real planner, independently
- **Visual:** paper table:

  | | GATS | LATS | ReAct | Oracle |
  | --- | --- | --- | --- | --- |
  | Planning tasks (100) | **100%** | 92% | 64% | 100% |
  | Stress test (12 cat.) | **100%** | 88.9% | 23.9% | — |
  | LLM calls / task | **0** | 37 | 13 | 0 |

  Footer: *0% variance across seeds · McNemar p < 0.01 · matches the oracle*
- **Say:** "Separately from the security result — as a planner it beats LLM-guided
  search outright: a hundred percent versus ninety-two for LATS, sixty-four for ReAct.
  On a twelve-category stress test — coding workflows, web navigation, long-horizon
  tasks — it holds at a hundred while LATS drops to eighty-nine and ReAct to
  twenty-four. Zero LLM calls **during planning**, zero variance, and it matches the
  perfect-information oracle. That last one matters: it's at the ceiling, not just ahead
  of the baselines."
- **Note:** ⚠️ say "**during planning**" every single time you say zero. Table 4 of the
  paper shows **50 one-time L3 bootstrap calls**. Unqualified "zero LLM calls" is the
  one provably-overstated claim available to you — and this room will find it.

---

## SECTION 7 — LIMITS & THREAT MODEL  (22:45–25:15)

### Slide 23 — Explicit threat model
- **Visual:** two columns. **In scope:** injection/hallucination that induces an action
  with unsatisfied preconditions; process-integrity; auditability. **Out of scope:** the
  injected text itself; valid-but-malicious goals (authorization); side channels; a
  compromised symbolic spec.
- **Say:** "Here's exactly what I claim and what I don't. In scope: lies that try to
  become actions. Out of scope: authorization, and the integrity of the spec itself."

### Slide 24 — Where this breaks (three honest limits)
- **Visual:** three rows:
  1. **Coverage** — L1 only protects what you can formalize. Open domain projected
     **L1 ~60% / L2 ~30% / L3 ~10%.**
  2. **Scale** — BFS value estimation is `O(|A|^d)`. Tested at |A|=15, d=20; impractical
     past |A|≈20, d>10 without pruning.
  3. **Evidence** — the planning eval is **synthetic, with known specs**. The paper's own
     fairness section concedes the setup favors GATS.
- **Say:** "Three honest limits. One: L1 only protects what you can **formalize** — for
  tool and API agents, the high-blast-radius surface, that's most of it; for open-ended
  text agents, coverage drops and you lean on the model again. Two: the reachability
  check is exponential in the worst case — fine for real action spaces, swap in a
  learned heuristic for huge ones and keep the verification layer. Three: that planning
  benchmark is synthetic with known specifications, and the paper says so. I'm not going
  to stand here and tell you it's been proven in the wild."
- **Note:** limit three is new and non-negotiable. The old deck said "this isn't a toy"
  — the paper's Limitations and Comparison Fairness sections don't support that phrasing.
  Saying it yourself is what makes slide 20 and 22 believable.

---

## SECTION 8 — CALL TO ACTION + CLOSE  (25:15–26:30)

### Slide 25 — What to actually do Monday
- **Visual:** 3 steps: (1) formalize your high-blast actions as preconditions/effects,
  (2) check `preconditions ⊆ ground_truth` before every irreversible action, (3) log
  every refusal as a security event. Below: *Is:* a free, deterministic, auditable
  action-verification layer for the formalizable parts of your agent. *Isn't:* an input
  filter, an authz system, or a replacement for the model.
- **Say:** "You don't need my planner. You need three things: write down the
  preconditions for your dangerous actions, check them against real state before you act,
  and treat every mismatch as a **security event**. It's not a silver bullet — it's the
  missing layer, the one between intent and action that almost nobody ships."

### Slide 26 — Close / Q&A
- **Visual:** *Put a verifier in your agent's action loop.* / `arXiv:2607.08894` /
  `github.com/MMWilliams/gats` / **one** QR code (`qr/qr-portfolio.svg` — that page
  links onward to the paper and the project site) / contact.
- **Note:** resist putting three QR codes on this slide. Three codes means nobody
  scans any of them. See `qr/README.md`.
- **Say:** "The agents are already acting in the world. Let's stop trusting them to tell
  us what they did. Neural proposes — **symbolic disposes.** Paper's on arXiv with Dymitr
  Nowicki, code and both demos are in the repo. I've got backup slides for the obvious
  hard questions — hit me."

---

## APPENDIX / Q&A BACKUP (have ready, don't present)

- **A1 — "Isn't this Llama Guard / NeMo?"** classifier-on-text vs symbolic-verify-on-state.
- **A2 — "Isn't this OPA/Rego/IAM?"** authorization vs verification, with the refund example.
- **A3 — "Isn't this just formal methods?"** "Yes — my claim is *put it in the action loop
  at runtime, for free.*" prior-art honesty.
- **A4 — "Your specs don't scale to open domains."** the 60/30/10 projection + "protect the
  high-blast surface first."
- **A5 — "Show me it's not rigged."** re-run `bench_lie_detection.py --n-scenarios 1000`
  live; point at the 0% on the intent class. ⚠️ **See the repro warning below before
  relying on this.**
- **A6 — GATS algorithm detail:** UCB1 over graph edges, BFS value estimate
  `StateValue(s) = γ/(d_G+1)`, confidence-weighted backup `v = p·StateValue(s′) − λ(1−p)`,
  and the **guarantee**: under an exact world model with sufficient depth and budget,
  GATS selects an action on a **shortest path** to the goal. Budget ablation: b=1 → 0%,
  b=5 → 84%, b=10 → 100%, saturates after. Robust to exploration constant
  (c ∈ {0.5, 1.0, 2.0} all 100%).
- **A7 — "Would a frontier model beat your baselines?"** The paper's LLM baselines are
  **Llama 3.2 via Ollama** (§4.4). Honest answer: you haven't tested a frontier model, and
  the determinism, auditability, and cost arguments hold regardless of baseline strength.
  Don't bluff this one.

---

## Numbers you are allowed to say on stage

Every figure traces to arXiv:2607.08894v1. Nothing else goes in the deck.

| Claim | Value | Source |
| --- | --- | --- |
| Main planning success | GATS 100% / LATS(b=10) 92% / ReAct 64% | Table 1 |
| Oracle (perfect info) | 100% — GATS matches it | Table 1 |
| Stress test overall | GATS 100% / LATS 88.9% / ReAct 23.9% | Table 6 |
| Biggest stress gains | coding, web nav, deep horizon: **+36.7%** each | Table 6 |
| LLM calls per task | GATS **0 at plan time** / LATS 37 / ReAct 13 | Table 1 |
| One-time bootstrap | **50** L3 calls, amortized | Table 4 |
| Variance across seeds | GATS 0% / LATS 1% / ReAct 5% | Table 1 |
| Significance | McNemar GATS vs LATS, **p < 0.01** | §5.1 |
| Budget ablation | b=1 → 0% · b=5 → 84% · b=10 → 100% | Table 2 |
| Open-domain projection | L1 60% / L2 30% / L3 10% | Table 4 |
| Scalability ceiling | tested \|A\|=15, d=20; impractical past \|A\|≈20, d>10 | §6 |

---

## Landmines

1. **"Zero LLM calls" without "during planning."** 50 one-time bootstrap calls exist.
2. **Implying the paper measured attacks.** It contains no security content at all.
3. **"This isn't a toy."** Retired — the paper's own Limitations section contradicts it.
4. **Frontier-model baselines.** Baselines are Llama 3.2 via Ollama. See A7.

---

## Inconsistencies in arXiv v1 — fix before v2

Found while re-reading the PDF. None break the deck as written, but a reader who pulls
the paper will hit them:

- **Seed count contradiction.** Table 1's caption says *3 seeds*; §4.4 and Appendix C
  both say **5 seeds** (42, 123, 456, 789, 1000). The caption looks wrong.
- **LATS LLM-call counts contradict.** Table 1 says **17** (b=5) / **37** (b=10); Table 5
  says **30** / **60** for the same conditions. The deck cites 37 (Table 1) — pick one.
- **Broken cross-reference.** "Clarifying L1/L2 Ablation Protocol" ends "*Table 3 describes
  which layer is selected … while Table 3 tests whether…*" — the first should be Table 4.
- **Table 4 mixes units.** The L3 column holds counts (50, 0) for measured rows but a
  percentage (10%) for the projected row.
- **Table 3 row alignment** renders ambiguously — the `no_l1` / `no_l3` descriptions
  appear transposed against their rows.

**⚠️ More urgent than any of the above:** the repro audit found the **stress test
reproduces but the main eval does not**. Slide 22 leads with the main-eval row
(100/92/64) and appendix A5 offers to re-run live. Re-run the main eval against the
current repo before you present. If it still doesn't reproduce: lead slide 22 with the
**stress row**, and pull A5.

---

## Build notes

- Slides 1 and 16 are **terminals, not slides** — rehearse the `--slow` timing so the
  pauses land where your talk-track expects.
- Record both demo runs the night before as MP4 fallback (see `demos/DEMO_SCRIPT.md`).
- Total: 26 hard-content slides + 7 backup. Finish speaking by 26:30 to leave real Q&A
  room in a 30-minute slot.
- If you're running long, cut in this order *(built-deck numbering)*: **21** (bench
  setup, fold into 22) → **6** (blast radius) → **19** (what just happened) → **11**
  (the tree merge — 12 carries a similar point, though they now animate differently
  enough to be worth keeping both). Never cut **14** (confidence is not evidence),
  **17** (the three properties), **20** (the honest caveat), **23** (the 0% class), or
  **25** (threat model) — with "where this breaks" gone, those are the whole of your
  credibility.
- Full 45-minute treatment is preserved in `DECK_OUTLINE_45MIN_ARCHIVE.md`.
