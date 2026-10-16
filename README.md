# Qwen3.8-27B-Deployment — SGLang serving + measurement on DGX Spark (GB10)

An evidence-driven SGLang deployment and measurement project for serving
Qwen3.8-27B (NVFP4 weights, FP8 KV) with DSPARK speculative decoding on an
NVIDIA DGX Spark (GB10, SM121, 128 GB unified memory). It records how the
container, model, KV cache, attention, and speculative-decoding settings were
chosen and what has actually been measured — not just headline tokens per
second.

This repository captures one tested environment. It is a reference for
experimentation, not a turnkey distribution. The status below reflects the
latest evidence in the repository; it is not a live check of a running
server.

## Latest recorded configuration

The recommended profile after the 2026-10-16 first sweep pass is
`configs/docker/candidates/combo-bf16-c12.env` (launcher:
`./configs/docker/run_qwen38_sglang.sh combo-bf16-c12`). It keeps the
baseline's DSPARK/flashinfer/262K setup and changes only the GDN state pool
and the admission cap; `baseline` remains the validated fallback and
`mamba-bf16` the runner-up for ≤8-user workloads. Full comparison:
`docs/results.md`.

| Setting | Recorded value |
| --- | --- |
| Hardware | DGX Spark GB10 (SM121), single GPU, 128 GB unified LPDDR5X |
| Host / image | `lmsysorg/sglang:qwen38-27b` (SGLang `0.0.0.dev0+qwen38.27b.g561c8f3`) per the SGLang DGX Spark NVFP4 recipe |
| Runtime | Docker; container `qwen3.8-27b-nvfp4-sglang`, host network, port **8000**, `--restart no`, manual starts only |
| Model | `RadixArk/Qwen3.8-27B-NVFP4` (~16.5 GB, NVFP4 W4A4 + FP8 projections, VL, hybrid GDN/attention, 262K context) |
| Speculative decoding | **DSPARK**, draft `RadixArk/Qwen3.8-27B-DSpark` (draft attention: flashinfer) |
| KV cache | `fp8_e4m3`; **1,381,531 token slots** (combo profile; baseline 962,380; bf16 pool alone reaches 1,490,777) |
| Attention backend | `flashinfer` (SM121-supported; `trtllm_mla` is SM100-only) |
| GDN state pool | `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 --max-mamba-cache-size 96 --mamba-radix-cache-strategy extra_buffer_lazy` (combo profile) |
| Static memory fraction | `0.85` |
| Context | 262,144 tokens (native, no YaRN; `CONTEXT_LENGTH`/`YARN` knobs in `.env`) |
| Admission | `max_running_requests` **12** (combo profile; baseline 8; queue-free to c16) |
| API | OpenAI-compatible; `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` |

## What passed on the latest profile

- **Startup:** ready in ~236–261 s on warm caches (weight load ~141 s;
  range across the sweep 202–322 s); no post-ready OOM or runtime error in
  any run.
- **Interfaces:** `/health` 200, `/v1/models` lists the served model, chat
  completions return with `finish_reason` and reasoning separated into
  `reasoning_content`.
- **Speculative decoding active:** `/generate` responses carry
  `spec_accept_length` / acceptance stats; DSPARK acceptance ~3.5 (c1) to
  ~5.5 (c8+).
- **First sweep pass (2026-10-16):** 9 profiles measured (cold start →
  capture → smoke → A/B bench → vllm-bench suite each), **0 failures and 0
  retractions in every scenario**. Winner `combo-bf16-c12`: ≥baseline at
  c8–c16 (191 tok/s c8, 183 c12, 164–205 c16), no queueing to c16 (TTFT
  ~1.8 s vs 7–10 s at cap 8), +44% KV capacity — consistent across two runs.
  Raw evidence: `evidence/`; findings: `docs/results.md`.
- **Measurement:** the benchmark path is `tests/run_profile.sh`; compare
  profiles with `python3 tests/compare.py`.

## Configuration workflow

```bash
./configs/docker/prefetch.sh                    # one-time: image + weights (~16 GB)
tests/run_profile.sh baseline                   # full benchmark cycle for a profile
python3 tests/compare.py                        # compare profiles measured so far
```

Profiles live in `configs/docker/candidates/*.env`; the sweep plan and
ranking criteria are in `docs/NEXT_EXPERIMENT.md`; findings go to
`docs/results.md`. A/B benchmark prompts are fixed-seed, so profiles are
directly comparable.

## Failures, regressions, and caveats

- **512K context + 12 concurrent + mem 0.95 OOM'd the 128 GB unified memory**
  (NVRM `NV_ERR_NO_MEMORY`, heavy swap, 2026-10-13). Current baseline is
  262K / 8 / 0.85; the failing configuration is retained as
  `configs/docker/candidates/512k-yarn.env` for reproduction only.
- **SM121 kernel gaps:** TokenSpeed and vLLM paths fail on GB10 (TRTLLM
  attention is SM100-only; Triton MHA exceeds shared memory at head_dim=256).
  SGLang + FlashInfer is the supported path — see `docs/INFERENCE_CONTEXT.md`.
- **MTP with FlashInfer failures:** fall back to `--attention-backend triton`
  or drop the speculative flags (`triton-attn.env` / `no-spec.env`).
- **Reasoning-model output shape:** a small `max_tokens` budget can produce
  reasoning only, with empty `content`; give short Q&A ≥ ~100 tokens.
- **No authentication:** the server binds `0.0.0.0:8000`; do not expose it to
  an untrusted network.
- **Legacy measurement artifact:** the 2026-08-16 vllm-bench result set
  (`evidence/legacy-20260816/`) is internally inconsistent (grade F,
  ~0.03 req/s) and is retained for lineage only; see `docs/results.md`.
- **vllm-benchmark-suite caveat:** it scores `content` tokens only; on this
  thinking model small output budgets are consumed by `reasoning_content`, so
  suite scores read near-zero even when the engine is healthy. Use the A/B
  harness for ranking (see `docs/results.md`).
- **DFLASH is not runnable on this image:** DFlash 2 draft checkpoints
  (`incoai/Qwen3.8-27B-DFlash2`) declare `DFlash2DraftModel`, which
  `lmsysorg/sglang:qwen38-27b` does not register; DSPARK also pins draft
  tokens to 8 (no d4 variant). See
  `evidence/mtp-flash-unsupported-20261016.md`.

## Not yet validated

- 512K-context risk profile (`512k-yarn` reproduces the historic OOM; no
  safer variant run yet) and `triton-attn` (on-demand fallback control).
- Above-cap admission at 16 (`combo-bf16-c12` at 12 already shows the shape).
- DFLASH on a DFlash2-capable image (blocked here — see `docs/results.md`).
- Near-limit context behaviour (largest benchmarked prompt ≈ 16K of 262,144
  tokens).
- Output-quality regressions across profiles (benchmarks measure speed and
  stability, not quality).

## Repository contents

- `configs/docker/` — the profile-driven SGLang launcher, `prefetch.sh`,
  `stop.sh`, and `candidates/` (one `.env` per launch profile).
- `tests/` — host-side validation and measurement scripts (`smoke_test.py`,
  `benchmark.py`, `capture_server.py`, `run_profile.sh`, `compare.py`,
  `gen_manifest.py`).
- `evidence/` — captured, committed reports (server snapshots, smoke, A/B
  bench, suite outputs); `legacy-20260816/` keeps the pre-packaging run.
- `docs/TESTING.md` — the manual-runs-only operating policy (hard rules).
- `docs/INFERENCE_CONTEXT.md` — hardware/model/flag background and
  migration/memory history.
- `docs/NEXT_EXPERIMENT.md` — the prepared config sweep and ranking criteria.
- `docs/results.md` — recorded findings.
- `AGENTS.md` — operating manual for AI agents working in this repo.

Historic note: `./start.sh` from the pre-packaging layout is now
`configs/docker/run_qwen38_sglang.sh [profile]`; `./stop.sh` is now
`configs/docker/stop.sh`.

## Validation

```bash
bash -n configs/docker/*.sh tests/*.sh
python3 -m py_compile tests/*.py
python3 tests/smoke_test.py --base-url http://127.0.0.1:8000   # server must be running
python3 tests/gen_manifest.py --check
```

## Adapting the configurations safely

The launch scripts are specific to this host and model. Before using them
elsewhere, review the container image tag, model paths, HF cache mount, port,
and memory settings. Model weights are not included; obtain them under the
model provider's terms. Captured reports may contain host-specific metadata.

## Operating policy

Manual runs only: no systemd, no cron, no restart policies, no retry loops.
One server at a time, started by hand. See `docs/TESTING.md`.
