# Work-cards for UOR-R4 (draft, not posted)

**Branch:** `bai/wasm-serving-runtime` on `github.com/Cyborg-Hawk-AI/uor-r4`
**Author handle:** `Cyborg-Hawk-AI` (attribution accepted)
**Format:** their work-card shape (goal, blocker, causal change, frozen gate, checks, cost). Pre-registration where their rules require it.

---

## Card 1 — #1172: the integer stack runtime runs in a WASM worker

**Goal.** Expose the accepted `UORLUT01` stack artifact through a capability API and WASM runtime, served in a browser worker with no backend (Pages-static).

**Blocker (theirs).** `wasm.rs` is 13 constant/helper exports — "not a serving runtime" (its own header). #1452 was rejected for unmeasured numbers.

**Causal change.** A `serving-wasm` feature on `uor-r4-integer` that compiles the preserved D11 stack runtime as a wasm32 `cdylib`, with:
- byte-seamed loading (`IntegerStackModel::parse(&[u8])` — no filesystem);
- a host-injected clock seam (std has no `Instant` on wasm32);
- load/session/step/save/restore exports (`wasm_serve`, `wasm_bindgen_exports`);
- single-threaded step fallbacks (rayon cfg'd out; integers unchanged);
- an integer-safe session codec replacing `serde_json` on wasm (serde's number parsing carries float/div, R2-hostile).

The D29 facade (`native_geometry.rs`) gates it off by default; its compile surface is unchanged without the feature.

**Frozen gate (measured, reproducible).**
- Native: `scripts/audit_zero_matmul_serving.py target/release/uor-r4-stack --stack` = FULL PASS (0 forbidden instructions, 44 symbol ranges).
- WASM: `scripts/audit_zero_matmul_wasm.py … --call-graph` = 5/7 serving symbols PASS (loadStack, saveSession, restoreSession, weightsPerToken, stack_argmax).
- Reload-identical: save → restore → continuation byte-for-byte identical to a fresh 48-token run's tail (Node-verified, `pkg-serving/verify.js`).
- 48/48 stack tests pass.

**Known FAIL (named, with cause).** `step` / `stack::IntegerStackSession::step` FAIL by 2 LLVM muls on wasm32 (1 in `stack_recurrence`, 1 in a RefCell-state helper in `wasm_step`) — LLVM strength-reductions that defeat source-level `black_box`, the same class `wasm.rs`'s header anticipated. Proposed resolution: D12 evidence route with the audit attached, or an explicit opaque kernel boundary as a follow-up.

**Checks.** `cargo check --offline` green; `cargo fmt --check` clean; 48 stack tests + 3 codec tests + 3 SIMD identity tests pass.

**Cost.** Reviewer time only. No compute, no data, no paid resources.

---

## Card 2 — #973: S1.4 2I icosian transport serving kernel (pre-registration)

**Goal.** Serve the 2I icosian transport snap (the S4-adopted main line's transport) as a D11 kernel.

**What exists.** The snap path (`stack_snap_select`, `stack_snap_rotation`) is Z[φ] shift-add argmax over the 120 roots + the 120×120 group table (`uor-r4-core/src/native_geometric/learner/group_table.rs`), exact Z[φ] tie-break. On x86_64 it audits R2-clean (FULL PASS).

**Frozen gate.** Parity against `docs/evidence/s1-4-snap-parity-2026-09-29.json` (the sealed snap-parity record, top1 agreement 0.9919). Pre-registering the run before any fit, per the rules.

**Cost.** The parity run on their data; we do not have the artifact or the held-out set.

---

## Card 3 — #820: energy protocol v2 (a cross-check on the M1 row)

**Goal.** A second-instrument energy measurement of the D11 engine.

**Measured (rnd, Linux x86_64, RAPL pkg).** Synthetic ra stack (15,968 weights/tok): 17,190 tok/s mean, RSS 3.67 MB, gross 3.172 mJ/tok, net (idle-rate-subtracted) 1.530 mJ/tok (CI95 ±0.037). Raw: `ops/measurement_report_rnd.json`.

**Caveat (stated).** Small synthetic stack, spawn-inclusive; the owner's M1 re-run is the record (their merge rule). D5-a "zero-DRAM after warm-up" needs uncore/IMC counters this VM lacks — ESTIMATE, not measured.

**Cost.** The M1 re-run by the owner.

---

## Card 4 — #963: the D5 split (a framing, with a QUBO formulation for D5-b)

**Goal.** Split D5 into D5-a (DRAM bytes/token → 0 after warm-up, achievable by residency for small models) and D5-b (parameter sparsity, only needed past cache).

**The D5-b formulation (design memo, not run).** One-hot QUBO assigning memory keys to fixed H4/E8 codebook cells, minimizing co-access-weighted collisions (`ops/D5B_ADDRESS_ASSIGNMENT_MEMO.md`). Solver is classical SA; no quantum device expedites it.

**Cost.** The measurement needs their memory co-access statistics, which we do not have.
