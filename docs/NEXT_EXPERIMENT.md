# Experiments

The prepared first-pass sweep ran 2026-10-16 (results: `docs/results.md`).
This file tracks outcomes and what remains.

## Completed — first pass (2026-10-16)

| profile | outcome |
|---|---|
| `baseline` | reference point (DSPARK d8, cap 8, fp32 GDN pool) — kept as fallback |
| `no-spec` | rejected: ~2.5–3.3× slower decode |
| `dspark-d8` | identical-config repeat — defines the noise band (short agg ±5–22%) |
| `dspark-d4` | invalid: DSPARK pins draft tokens to 8 (engine constraint) |
| `mtp-flash`, `dflash-d8` | blocked: image lacks `DFlash2DraftModel` |
| `mamba-bf16` | runner-up (≤8-user shape): +55% KV, c8 decode edge; queues at c12+ |
| `mem-90` | rejected: mixed speed, +10.5% KV only |
| `maxreq-12` | kept as a parent of the winner (queue-free c12+) |
| `chunk-8192` | rejected: long-prompt TTFT +8–10 s |
| `combo-bf16-c12` | **promoted** (recommended config in `README.md`) |

Runtime constraints discovered and recorded: DSPARK draft tokens are fixed at
8; DFLASH needs `extra_buffer` mamba strategy *and* a DFlash2-capable image.
Sweep hygiene that mattered: identical-config repeatability runs set the
noise band — single-run differences inside it were not treated as wins.

## Backlog — second pass (rough priority order)

1. **`combo16`** — `combo-bf16-c12` variant with `MAX_RUNNING_REQUESTS=16`:
   does capacity hold (KV shrinks) and does c16–c24 stay queue-free?
2. **`512k-yarn-safe`** — 512K context with the winning pool +
   `MEM_FRACTION=0.90` + cap 12; attempt only with the historic OOM failure
   mode known (reproduce carefully, watch unified memory).
3. **`combo-mem90`** — combo + `MEM_FRACTION=0.90`: capacity vs stability.
4. **DFLASH on a newer image** — an image that registers `DFlash2DraftModel`
   (e.g. the `lmsysorg/sglang:latest` class the RTX node used). A new image
   is a new runtime baseline: re-check SM121 compatibility first, then re-run
   the full cycle.
5. **Output-quality checks** of `combo-bf16-c12` vs `baseline`
   (spot-check arithmetic/reasoning; the benchmarks cover speed and
   stability only).
6. `triton-attn` — control only if FlashInfer misbehaves.

## Rules

- One profile at a time, cold start, hand-invoked:
  `tests/run_profile.sh <profile>` (see `docs/TESTING.md`).
- Evidence before conclusions; retain failed probes.
- Only call a difference real if it clears the noise band (see the
  repeatability entry in `docs/results.md`).
- Update `docs/results.md` per run; promote in `README.md` only on a clear,
  repeated win.

## Ranking criteria (in order)

1. Zero failures/retractions and exact output lengths in `benchmark.py`.
2. Aggregate tok/s at `short-c8-512x256`.
3. Per-request decode tok/s p50 at c1 and c8.
4. TTFT p50/p90 on long prompts (`long-c8-16384x256`).
5. Speculative acceptance length.
6. KV pool size (`max_total_num_tokens` in the capture) as the capacity
   tiebreaker.
7. Stability across repeated runs.
