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

_No sweep results yet — the first pass through
`configs/docker/candidates/` was prepared 2026-10-16 (see
`docs/NEXT_EXPERIMENT.md`). Add entries here as runs complete._

## Legacy (pre-packaging)

### vllm-bench run — 2026-08-16 (suspect, retained)

`evidence/legacy-20260816/` keeps the first vLLM Benchmark Suite pass over
this deployment (32K/64K/128K × 1/4/8/12, 500 output tokens). It reports
~0.03 req/s and a peak of 1.3 tok/s with grade F — inconsistent with the
server's measured behaviour at the time (~30–47 s wall per single request,
i.e. ~13 tok/s decode). The run is retained for lineage but should not be
trusted or compared against the packaged sweep results; re-baseline with the
current tooling instead (`tests/run_profile.sh baseline`).
