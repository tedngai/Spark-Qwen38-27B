# Next experiments — the config sweep

Goal: find the best-performing launch configuration on the GB10 and record
the evidence. One profile at a time, cold start, hand-invoked via
`tests/run_profile.sh <profile>` (see `docs/TESTING.md`).

## Prepared profiles (configs/docker/candidates/)

`mtp-flash` replaced `mtp-eagle` on 2026-10-16 (user request): DFLASH
draft-model speculation instead of the in-checkpoint EAGLE MTP head.

| # | Profile | Changes vs baseline | Hypothesis to test |
|---|---|---|---|
| 0 | `baseline` | — | Reference point (validated config; first sweep entry) |
| 1 | `no-spec` | DSPARK off | Quantify the speedup DSPARK buys; how much memory/PPL it costs |
| 2 | `dflash-d8` | DFLASH, 8 draft tokens — identical surroundings to `mtp-flash` (d4) | Completes the DFLASH draft-token A/B (the old `dspark-d4` attempt was invalid — DSPARK pins 8; see `docs/results.md`) |
| 3 | `mamba-bf16` | bf16 GDN pool, ratio 4.21, cache 64 | ~20 GB back to KV cache; decode-speed cost? |
| 4 | `mem-90` | mem fraction 0.85→0.90 | Bigger KV pool/graph room; watch unified-memory headroom |
| 5 | `maxreq-12` | admission 8→12 | Does 12 fit and improve aggregate throughput? |
| 6 | `chunk-8192` | chunked prefill 2048→8192 | Long-prompt TTFT vs decode interference |
| 7 | `mtp-flash` | DFLASH, draft `incoai/Qwen3.8-27B-DFlash2`, 4 draft tokens, on the old MTP-era settings (chunk 8192, no prefill graph, bf16, mem 0.92, c12) | DFLASH vs DSPARK acceptance/speed (the RTX PRO 6000 node's chosen algorithm) |
| 8 | `triton-attn` | triton attention everywhere | Control run for FlashInfer failures (not a speed candidate) |
| 9 | `512k-yarn` | 512K context, YaRN, mem 0.95, c12, EAGLE 3/1/4 (historic) | **Risky.** Reproduces the historic OOM; only with a safer variant in mind |

## Suggested order

0 → 1 → 7 → 2 (DFLASH d4 then d8 back-to-back) → 3 → 4 → 5 → 6 (stop and
reassess if a run fails; 8 is on-demand; 9 last, deliberately).

Ideas after the first pass (combine winners only once each is understood):

- `mem-90` + `maxreq-12` combined if both hold up individually.
- `--disable-prefill-cuda-graph` A/B on the DSPARK config (recipe sets it;
  baseline does not).
- KV cache dtype comparison (fp8_e4m3 vs bf16) — memory vs quality/latency.
- Higher `max_running_requests` with the bf16 GDN pool (capacity rebalance).

## Rules

- Cold start per profile; never measure two configs against one running
  process.
- Every run goes through `tests/run_profile.sh` (or the individual tests with
  `--save-dir evidence`) — saved evidence before conclusions.
- Abort → inspect → retry by hand. No loops, no auto-retries.
- Record findings in `docs/results.md`; update `README.md`'s status table
  only when a new profile beats the baseline on the criteria below.

## Ranking criteria (in order)

1. Zero failures/retractions and exact output lengths in `benchmark.py`.
2. Aggregate tok/s at `short-c8-512x256` (the admission cap).
3. Per-request decode tok/s p50 at c1 and c8.
4. TTFT p50/p90 on long prompts (`long-c8-16384x256`).
5. Speculative acceptance length (DSPARK/EAGLE efficiency).
6. KV pool size (`max_total_num_tokens` in server-info evidence) as the
   context/concurrency capacity tiebreaker.
7. Stability: no OOM, no swap storm, stable for repeated runs.
