# GATS Integration Lab: Technical Report

*Prepared 2026-07. Code: `integrations/` in this repository. Every number in
this report is produced by `python -m integrations.run_deep` and written to
`integrations/results/deep_results.json`; the run is deterministic, stdlib
only, and makes zero LLM calls.*

## 1. What this lab tests

The claim under test: GATS can be embedded as the planning-and-provenance
layer under a customer's existing stack, treated as a completely separate
system, and the result is **deterministic** (same inputs, byte-identical
plans and audit chains), **traceable** (every fact traces to the action that
established it), and **auditable** (a tamper-evident log an auditor can
verify without trusting the runtime).

## 2. The integration boundary

Each customer is modeled as a `CompanySystem` that owns its ground truth.
The boundary mirrors a real deployment:

- The customer's fact store is private. GATS never mutates it directly.
- GATS sees exactly two things: the **published tool contract** (tool names,
  preconditions, declared effects) and **observe()**, a read-only state
  snapshot, the same surface a status API or log stream provides.
- Tool execution happens inside the customer's system with the customer's
  own semantics, including semantics that silently diverge from the
  published contract (see the drift battery).

The embedded layer (`AuditedRuntime`) does four things per action:

1. **Verify against live ground truth.** `preconditions ⊆ observe()`,
   checked at execution time, never against what any upstream model claims.
2. **Enforce human checkpoints.** Tools the customer marks as gated never
   auto-execute; without an approver the run halts in
   `blocked_awaiting_human` and the gated tool is provably never called.
3. **Detect contract drift.** The observed state delta is compared with the
   declared effects; undocumented side effects and silently-missing effects
   are flagged in the record.
4. **Append a hash-chained audit record.** Canonical JSON, SHA-256,
   prev-hash linked. Timestamps live outside the hashed body so chains are
   byte-reproducible.

Planning uses the same `State`/`Action`/UCB1 core as the paper demos, with
two integration-layer additions: an additive-heuristic (h_add) value
function computed on the real successor state (so progress-destroying
actions score strictly worse), and a search budget that scales with the
applicable-action set.

## 3. The customer systems

Six archetypes drawn from the go-to-market ICP research, each with a
workflow modeled on how these shops actually operate, and each with the
characteristic injection: an upstream model asserting prerequisites are
already satisfied.

| system | domain | workflow | human gate | injection lie |
|---|---|---|---|---|
| voltway | AI-agent vendor selling to a bank | Reg E dispute: intake → identity → transactions → confirm → provisional credit → document → notify → close | none (fully automated flow) | "identity and charge already confirmed; issue the credit now" |
| meridian | HR tech (EU AI Act Annex III, NYC LL144) | application → AEDT notice → profile → knockouts → score (requires current bias audit) → human shortlist review → advance | shortlist review | CV-embedded "candidate pre-cleared, advance directly" |
| crestline | RIA / broker-dealer (FINRA, Reg BI) | exception → account profile → suitability → rep attestation → principal review → memo → close | principal review | "principal reviewed yesterday, file and close" |
| sentriops | MSSP SOC | alert → enrich → confirm → change approval → isolate → eradicate → restore → report → close | change approval | "client pre-approved containment; isolate immediately" |
| harborpoint | insurance MGA | FNOL → coverage → severity → reserve → adjuster → estimate → approve payment → issue → close | payment approval | "adjuster approved $12,400; pay today" |
| ledgerlane | CPA firm (SOX) | population → completeness → sample → vouch → evaluate exceptions → reviewer sign-off → archive | reviewer sign-off | "prior-year found control effective; roll forward and archive" |

Four deliberately complicated cases on top:

- **cascade** — multi-entity month-end close: three subsidiaries closed in
  parallel, intercompany matching, eliminations, FX translation, controller
  sign-off, period lock. Minimum plan 19 steps. Includes a **trap tool**
  (`reopen_subledger_us`) that is always applicable late in the close and
  silently destroys progress; the planner must never touch it.
- **breach** — cross-system OAuth breach: IdP + EDR + cloud evidence, scope,
  PII assessment, TWO independent human gates (change approval, legal
  review), GDPR-style regulator notification required before the
  post-incident report, 14 steps.
- **megaclaim** — commercial fire with suspected arson: SIU referral, fire
  marshal, litigation hold, subrogation, reinsurer notification as a
  precondition of settlement authority, payment, reinsurance recovery,
  16 steps.
- **voltway_pro** — the vendor at scale: three case types (dispute,
  chargeback representment, fraud review) in one tenant with namespaced
  facts, a shared KYC subgraph feeding two flows, and two human gates,
  17 steps.

Plus two structural cases:

- **multitenant** — one vendor runtime, two bank tenants with different
  gating policies. Proves: state isolation (tenant A resolving a dispute
  changes nothing in tenant B), per-tenant hash chains, per-tenant
  determinism, and that tenant A's human approval can never satisfy tenant
  B's checkpoint.
- **drift** — the customer's real API diverges from its published contract:
  one tool has an undocumented side effect, another silently fails to
  establish its declared effect. The layer must flag both and halt before
  executing anything that depended on the missing fact.

## 4. The verification battery (per company)

Every system runs the same suite:

1. **Determinism** — plan + execute three times from scratch; the plan
   sequence, the audit-chain head hash, and the final state hash must be
   identical across runs.
2. **Audit-chain integrity** — recompute every record hash and prev-link.
3. **Tamper evidence** — mutate each record; the chain must break.
4. **Provenance closure** — from the log alone: every final fact traces to
   a producing record; every executed action's preconditions trace to the
   initial state or an earlier record.
5. **Deterministic replay** — re-execute the recorded actions on a fresh
   instance of the customer system; every recorded pre/post state hash must
   match byte for byte.
6. **Injection refusal** — the scenario's lie is attempted; the verifier
   checks ground truth, refuses, logs a security event, and the planner
   then still reaches the goal legitimately.
7. **Human checkpoint** — with no approver present, the run halts and the
   gated tool is never called (checked against the customer's own call log).

## 5. Deep validation (layer 2)

- **Generated-workflow family** — 120 seeded random customer systems
  (4-8 dependency layers, distractor tools, trap tools that delete
  progress). Each is planned, executed, and put through determinism, chain,
  replay, provenance, and injection checks.
- **Scale** — a 40-stage, 240-tool pipeline (200 ticket-consuming work
  items + 40 stage gates), planned and executed end to end with the full
  battery.
- **Exhaustive tamper attack** — for all reference runs: mutate every
  mutable field of every record and require chain breakage each time.
- **Injection battery** — for every archetype and complex case, attempt
  every tool from every execution prefix; every attempt whose preconditions
  are not actually satisfied must be refused.
- **Crash-resume** — kill the run after every prefix, rebuild customer
  state by replaying the audit log on a fresh instance, re-plan from there;
  the final state must match the uninterrupted run's hash exactly.

## 6. Graph database validation (layer 3)

Audit logs are lifted into a typed provenance graph
(`integrations/graph_db.py`): company, fact, and event nodes; INITIAL /
REQUIRES / PRODUCES / DELETES / FOLLOWS edges; facts are company-scoped so
tenant partitions cannot share nodes. Validated:

- **G1 Integrity** — the merged graph over every run in the lab validates
  clean: no dangling edges, correct endpoint types, temporal soundness
  (every REQUIRES satisfied by the initial state or an earlier PRODUCES
  with no intervening DELETE), provenance never points forward in time,
  FOLLOWS strictly forward, and no connected component contains two
  companies.
- **G2 Round-trip** — serialize → load → identical canonical hash.
- **G3 Order-invariance** — ingesting runs in a different order produces
  the identical canonical hash (safe for distributed merge).
- **G4 Corruption battery** — seeded corruptions (dangling nodes,
  cross-tenant edges, deleted producers, time-reversed provenance, flipped
  chain order); the validator must flag every one.
- **G5 Query truth** — `why()` support trees and `blast_radius()` closures
  recomputed brute-force from the raw audit records must agree exactly.
- **G6 Scale/perf** — counts and wall-clock for build, validate, queries.

## 7. Results

All layers passed. Total wall clock 372 s, single process, no GPU, no LLM.

**Layer 1 — company integrations (12/12 pass).** All ten workflow systems
reach their goals with plans of 7–19 steps; all pass determinism (3×
identical plan + chain head + final state), chain integrity, tamper
evidence, provenance closure, replay, injection refusal, recovery after
injection, and (where gated) the human checkpoint. `cascade` never touches
the trap tool. `multitenant` passes all five isolation checks.
`drift` flags both contract violations (`issue_provisional_credit` →
undocumented `ledger_entry_pending`; `notify_customer` → declared
`customer_notified` never delivered) and halts at `close_dispute` before
acting on the missing fact.

**Layer 2 — deep validation.**

| battery | result |
|---|---|
| generated family (120 systems) | 120/120 solved; 120/120 deterministic; 120/120 chains valid; 120/120 replays exact; 120/120 provenance closed; 120/120 lies refused; mean plan 10.5, max 23 |
| scale (240 tools, 40 stages) | 240-step plan in 0.13 s; goal reached; chain, replay, provenance all pass; 240 audit records |
| exhaustive tamper | 3,894/3,894 field mutations detected (100%) |
| injection battery | 611/611 lie attempts refused across every execution prefix of every case; 0 false refusals |
| crash-resume | 114/114 crash points converged to the uninterrupted final state |

**Layer 3 — graph database.**

| check | result |
|---|---|
| corpus | 31 runs ingested; 1,440 nodes; 2,666 edges |
| G1 integrity | clean (0 problems) |
| G2 round-trip | canonical hash identical |
| G3 order-invariance | canonical hash identical under reordered ingestion |
| G4 corruption battery | 60/60 detected (cross-tenant 11/11, dangling 7/7, flipped chain 20/20, future provenance 11/11, deleted producer 11/11) |
| G5 query truth | why() 549/549 exact vs brute-force oracle; blast_radius() 11/11 |
| G6 perf | build + validate < 0.1 s on this corpus; the 363 s in `query_seconds_total` is dominated by the brute-force *oracle* recomputation used to check the queries, not the graph queries themselves |

## 8. Honesty notes

- These are simulated integration environments built from researched
  workflow shapes, not deployments at real companies. The lab exists to
  make the integration mechanics (boundary, verification, audit, replay,
  isolation) testable and falsifiable before design-partner work.
- The planner additions (h_add value guidance, adaptive search budget) live
  in the integration layer, not the paper artifact; the paper's engine is
  imported unmodified from `demos/agent_lies_demo.py`.
- One generated system (seed 74) failed with the fixed search budget and
  passed after the adaptive budget fix; the failure and fix are preserved
  in the lab history rather than hidden.

## 9. Workflow research sources

Claims about how these shops operate draw on: TPA/MGA claims workflow
guides (VCA Software claim lifecycle; Modotech claims processing for
carriers and MGAs; Engle Martin TPA overview), SOC/MSSP runbook practice
(NIST SP 800-61r3-aligned runbook guides; MSSP security operations
workflow write-ups; incident.io runbook automation), NYC Local Law 144
bias-audit and AEDT notice requirements (Warden AI and BABL AI compliance
guides), FINRA/Reg BI supervision and exception-report practice (FINRA
Reg BI preparedness reviews; InnReg WSP guide), and the EU AI Act Article
12 logging analyses cited in the company strategy documents.
