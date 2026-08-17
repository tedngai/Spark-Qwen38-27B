# Qwen3.8-27B Deployment — Agent Context

## Project
- Self-hosted inference via Docker for `RadixArk/Qwen3.8-27B-NVFP4` on NVIDIA GB10 (DGX Spark)
- Dense 27B, NVFP4 W4A4 + FP8 projections, vision-language (image + video), hybrid GDN/full-attention layers, 262K context, MTP-trained
- Engine: SGLang (OpenAI-compatible API)
- Image: `lmsysorg/sglang:qwen38-27b` (official SGLang image per https://docs.sglang.io/cookbook/qwen3.8-27b/)
- Container name: `qwen3.8-27b-nvfp4-sglang`; host network, port 8000
- Repo: http://itgit.pratt.edu/tngai/Qwen3.8-27B-Deployment.git

## Files
| File | Purpose |
|---|---|
| `prefetch.sh` | Pull image + download model into `.cache/huggingface` |
| `start.sh` | Docker launcher — `--network host`, `--gpus all`, polls `/v1/models` until ready |
| `stop.sh` | Remove serve/prefetch containers + PID file (logs preserved) |
| `.env.example` | Template for `.env` (`HF_TOKEN`, `CONTEXT_LENGTH`, `YARN`) |
| `.env` | NOT in repo — create from `.env.example` (gitignored) |
| `.gitignore` | Excludes `.env`, `.cache/`, logs, pid, benchmark_results |
| `README.md` | Project docs and troubleshooting |
| `AGENTS.md` | This file — the agent's operating manual |

## Fresh-Spark Setup (run in order)
1. `git clone http://itgit.pratt.edu/tngai/Qwen3.8-27B-Deployment.git && cd Qwen3.8-27B-Deployment`
2. `cp .env.example .env` and fill in `HF_TOKEN` (Hugging Face token for downloads)
3. `./prefetch.sh` — pulls image, downloads ~16 GB model into `.cache/huggingface`
4. `./start.sh` — blocks until ready (polls `GET /v1/models`); then verify:
   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"RadixArk/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":256}'
   ```
5. `./stop.sh` to stop; `rm -rf .cache/ .sglang.* .prefetch.*` for full cleanup

## Key Config (start.sh) — do not change without reason
- Serve command follows the SGLang DGX Spark NVFP4 recipe: https://docs.sglang.io/cookbook/qwen3.8-27b/
- `--attention-backend flashinfer`, `--mem-fraction-static 0.92`, `--disable-prefill-cuda-graph`
- `--chunked-prefill-size 8192`, `--trust-remote-code`, `--allow-auto-truncate`
- `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`
- GDN state pool (tuned; ~7 GB vs ~26.6 GB fp32 default): `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size $((MAX_RUNNING_REQUESTS*8))`
- Context: `CONTEXT_LENGTH=262144` + `YARN=0` default (native 262K, no rope override). Range 262144..1000000; >262K requires `YARN=1` (or exactly 1000000, auto). start.sh computes the `--json-model-override-args` text_config YaRN block and sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
- MTP enabled via in-checkpoint head: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (~3-4x decode, accept len ~3.5/4)
- Tuning knobs: `MAX_RUNNING_REQUESTS=8` (concurrency), `CONTEXT_LENGTH`, `YARN`, `--mem-fraction-static`

## Agent Role
- Manage inference lifecycle (start / stop / restart / logs / tune)
- Modify `start.sh` config flags only when the user requests changes
- Monitor logs: `tail -f .sglang.log` (server), `.prefetch.log` (downloads)
- Verify readiness with `curl -fsS http://127.0.0.1:8000/v1/models`
- Push config changes back to the repo (see Git Workflow); update this file and README.md when flags or setup change significantly

## Server Info
- OpenAI-compatible endpoint: `http://0.0.0.0:8000/v1`
- Model name: `RadixArk/Qwen3.8-27B-NVFP4`
- Readiness: `GET /v1/models`; chat: `POST /v1/chat/completions`
- Reasoning: `reasoning_content` field in responses; `enable_thinking` toggles thinking; thinking models need 500+ output tokens to produce content beyond reasoning

## Benchmark Protocol
- Harness: `vllm-bench` (`vllm-benchmark-suite`) against `http://127.0.0.1:8000/v1/chat/completions`
- Result JSON goes to `benchmark_results/result_*.json` (gitignored)

## Troubleshooting (verified fallbacks)
- Container exits before ready → `tail -200 .sglang.log`
- Host OOM (GB10 unified memory) → lower `--mem-fraction-static` to `0.90`
- MTP with FlashInfer fails → `--attention-backend triton` or remove the `--speculative-*` flags
- Re-download on every start → verify `.cache/huggingface` mount and `HF_HOME=/root/.cache/huggingface` in container
- Image-only text inference → vision tower is live by default; no extra flags needed

## Memory History (why current values)
- Host OOM: 512K context + 12 concurrent + 0.95 mem fraction exhausted 128 GB unified memory (NVRM NV_ERR_NO_MEMORY, heavy swap). Current: 262K + 8 concurrent + 0.92

## Git Workflow
- Repo holds only deployment-critical files; secrets (`.env`) and caches are never committed
- To push: `git add -A && git commit -m "<msg>" && git push` (repo stores credentials in `.git/config`)

## Migration History (this deployment's lineage)
- v1: TokenSpeed + `Qwen/Qwen3.8-27B-FP8` — failed on GB10 (KV cache budget, SM121 kernel gaps)
- v2: vLLM + `unsloth/Qwen3.8-27B-NVFP4` — reference from tonyd2wild/Qwen3.8-27B-NVFP4-DGX-Spark
- v3 (current): SGLang + `RadixArk/Qwen3.8-27B-NVFP4` — official checkpoint per SGLang docs
