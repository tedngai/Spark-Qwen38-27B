# Results

Recorded results, newest first. Every entry references its evidence files.
Status in `README.md` follows this file, not the other way around.

Entry template:

```markdown
## <profile> — <YYYY-MM-DD>

- Config: <profile file or key diffs vs baseline>
- Started: cold start, ready in <n> s; no errors during run
- A/B bench (evidence/<profile>-bench-<ts>.json):
  - short-c1: decode p50 <n> tok/s, ttft p50 <n> s, accept <n>
  - short-c8: aggregate <n> tok/s, decode p50 <n> tok/s, ttft p50 <n> s
  - long-c8:  aggregate <n> tok/s, ttft p50 <n> s
  - failures/retractions: <n>/<n>
- vllm-bench (evidence/<profile>-vllmbench-<ts>/): peak <n> tok/s, grade <x>
- Verdict: <better/worse/comparable vs baseline — and why>
```

## Findings so far

### Measurement caveat — vllm-benchmark-suite on a thinking model

The suite counts **`content` tokens only**. This model is a reasoning model:
small output budgets are consumed by `reasoning_content`, so suite throughput
reads near-zero (grade F) even when the engine is healthy. This also explains
the suspect 2026-08-16 legacy run. Ranking decisions use the A/B harness
(`tests/benchmark.py`, raw `/generate` forced decodes — counts every token);
the suite is kept only as a consistent, content-only cross-check.

### dspark-d4 — 2026-10-16 — invalid (engine constraint)

- DSPARK requires `speculative_num_draft_tokens == gamma + 1 == 8`; the d4
  variant cannot start (`ValueError` at startup). See
  `evidence/dspark-d4-invalid-20261016.md`.
- Consequence: DSPARK's draft-token count is fixed at 8 on this build; the
  slot was repurposed to `dflash-d8` (DFLASH d8), so the draft-token A/B is
  done inside the DFLASH family.

### dspark-d8 — 2026-10-16 — repeatability check, not a variant

- Discovery: DSpark's default `--speculative-num-draft-tokens` is **8**, so
  this profile ran the same effective config as baseline (verified: identical
  container command apart from the explicit `... 8`). It therefore measures
  **run-to-run variance**, not an experiment. Retained for exactly that.
- Cold start ready in 231 s; 0 failures/retractions.

| scenario | agg | dec p50 | accept | baseline agg | baseline dec |
|---|---|---|---|---|---|
| short-c1 | 36.57 | 38.06 | 4.34 | 30.1 | 31.1 |
| short-c4 | 101.61 | 35.22 | 4.86 | 122.7 | 37.9 |
| short-c8 | 137.88 | 33.25 | 5.83 | 144.5 | 29.5 |
| short-c12 | 144.53 | 26.82 | 5.04 | 151.1 | 28.3 |
| short-c16 | 154.62 | 26.54 | 5.11 | 141.8 | 25.0 |
| long-c1 | 22.86 | 40.05 | 4.66 | 21.1 | 35.0 |
| long-c8 | 38.28 | 11.64 | 5.03 | 37.4 | 11.0 |
| long-c12 | 39.46 | 10.79 | 5.20 | 40.4 | 10.63 |

- Variance read (identical config!): short-prompt aggregates swing ±5–22%
  between runs (worst at c1/c4), decode p50 ±10–22%, long-prompt aggregates
  are stable (±3–8%). **Implication for the sweep: only treat a difference as
  real if it clears this noise band; c1 numbers are the noisiest.**
- The profile was repurposed to `dspark-d4` (a real variant) after this run;
  this entry stays as the variance reference.

### no-spec — 2026-10-16

- Config: baseline minus speculative decoding (`SPEC_ALGORITHM=`).
- Cold start ready in 202 s; no errors; `accept=None` throughout (expected).
- A/B bench (`evidence/no-spec-bench-20261016T044723Z.json`):

| scenario | agg tok/s | decode p50 | ttft p50 | accept | fails | retr |
|---|---|---|---|---|---|---|
| short-c1 | 12.45 | 12.61 | 0.25 s | – | 0 | 0 |
| short-c4 | 45.05 | 11.56 | 0.58 s | – | 0 | 0 |
| short-c8 | 45.05 | 10.92 | 1.36 s | – | 0 | 0 |
| short-c12 | 64.20 | 10.98 | 1.12 s | – | 0 | 0 |
| short-c16 | 58.03 | 10.90 | 25.44 s | – | 0 | 0 |
| long-c1 | 10.02 | 12.30 | 4.74 s | – | 0 | 0 |
| long-c8 | 24.26 | 6.95 | 21.75 s | – | 0 | 0 |
| long-c12 | 28.70 | 7.06 | 30.81 s | – | 0 | 0 |

- Verdict: DSPARK is worth **~2.5–3.3×** decode (p50 ~11–12.6 tok/s here vs
  ~25–38 baseline) and ~3× aggregate (c8: 45 vs 144). No-spec is rejected as
  a speed config; keep as a control for spec-decode questions.

### baseline — 2026-10-16

- Config: `configs/docker/candidates/baseline.env` (DSPARK 4 draft tokens,
  flashinfer, mem 0.85, cap 8, chunk 2048, fp8 KV, 262K ctx).
- Cold start ready in **216 s**; no errors; KV pool **962,380 slots**.
- A/B bench (`evidence/baseline-bench-20261016T043511Z.json`):

| scenario | agg tok/s | decode p50 | ttft p50 | ttft p90 | accept | fails | retr |
|---|---|---|---|---|---|---|---|
| short-c1 | 30.1 | 31.1 | 0.27 s | 0.27 s | 3.51 | 0 | 0 |
| short-c4 | 122.7 | 37.9 | 0.97 s | 0.97 s | 5.40 | 0 | 0 |
| short-c8 | 144.5 | 29.5 | 1.46 s | 4.03 s | 5.31 | 0 | 0 |
| short-c12 | 151.1 | 28.3 | 1.16 s | 10.90 s | 5.33 | 0 | 0 |
| short-c16 | 141.8 | 25.0 | 9.66 s | 15.14 s | 4.60 | 0 | 0 |
| long-c1 | 21.1 | 35.0 | 4.81 s | 4.81 s | 4.06 | 0 | 0 |
| long-c8 | 37.4 | 11.0 | 22.31 s | 37.19 s | 5.05 | 0 | 0 |
| long-c12 | 40.4 | 10.6 | 31.55 s | 62.84 s | 5.33 | 0 | 0 |

- All outputs exactly 256 forced tokens; zero failures/retractions.
- vllm-bench (`evidence/baseline-vllmbench-20261016T043511Z/`): 2 configs
  (32K × c1/c4, 150 output tokens) — 15.2 s / 8.4 s per request; score reads
  0.4 "tok/s" (content-only caveat above).
- Extra: `evidence/baseline-server-info-20261016T042635Z.json` is a retained
  failed capture from the pre-fix readiness race (`/health` 503 right after
  startup); the packaged tooling now retries and the launcher waits for
  `/health` too.
- Verdict (reference point): aggregate rises to ~151 tok/s at c12 even with
  the cap of 8 (queued requests keep the pipeline full); TTFT degrades past
  the cap (1.2 s → 9.7 s at c16). Long prompts expose the prefill cost:
  16K×8 pushes TTFT to 22 s and per-user decode p50 falls to 11 tok/s.


## Legacy (pre-packaging)

### vllm-bench run — 2026-08-16 (suspect, retained)

`evidence/legacy-20260816/` keeps the first vLLM Benchmark Suite pass over
this deployment (32K/64K/128K × 1/4/8/12, 500 output tokens). It reports
~0.03 req/s and a peak of 1.3 tok/s with grade F — inconsistent with the
server's measured behaviour at the time (~30–47 s wall per single request,
i.e. ~13 tok/s decode). The run is retained for lineage but should not be
trusted or compared against the packaged sweep results; re-baseline with the
current tooling instead (`tests/run_profile.sh baseline`).
