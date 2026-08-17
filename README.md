# SGLang Qwen3.8-27B-NVFP4

Self-hosted inference via Docker for `RadixArk/Qwen3.8-27B-NVFP4` using [SGLang](https://github.com/sgl-project/sglang) on an NVIDIA DGX Spark (GB10, SM121).

- Model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) — dense 27B, NVFP4 W4A4 + FP8 projections, vision-language (image + video), 262K context, hybrid GDN/full-attention, MTP-trained
- Engine: SGLang (OpenAI-compatible API)
- Image: `lmsysorg/sglang:qwen38-27b` (official SGLang image with Qwen3.8 support)
- Recipe source: [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/qwen3.8-27b/) — DGX Spark / NVFP4 cell

## Files

| File | Purpose |
|---|---|
| `prefetch.sh` | Pull the Docker image and download the model into the shared HF cache |
| `start.sh` | Launch the SGLang serve container, follow logs, poll `/v1/models` until ready |
| `stop.sh` | Remove the container and PID file (logs are preserved) |
| `.env.example` | Template for `.env` — copy to `.env` and set `HF_TOKEN` (gitignored) |
| `.cache/huggingface` | Model weights cache (bind-mounted into the container, gitignored) |
| `.prefetch.log` / `.sglang.log` | Prefetch and server logs (gitignored) |
| `AGENTS.md` | Operating manual for AI agents managing this deployment |

## Quickstart

```bash
git clone http://itgit.pratt.edu/tngai/Qwen3.8-27B-Deployment.git
cd Qwen3.8-27B-Deployment
cp .env.example .env        # set HF_TOKEN
./prefetch.sh               # one-time: download model (~16GB) into the shared cache
./start.sh                  # start server, wait for readiness
curl http://127.0.0.1:8000/v1/models
./stop.sh                   # stop and remove container
```

## Server

- OpenAI-compatible endpoint: `http://0.0.0.0:8000/v1`
- Model name: `RadixArk/Qwen3.8-27B-NVFP4`
- Thinking mode is on by default (responses contain `reasoning_content`); control with `enable_thinking` / `reasoning_effort`

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"RadixArk/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":256}'
```

Monitor logs: `tail -f .sglang.log`

## Key config (start.sh)

| Flag | Value |
|---|---|
| `--mem-fraction-static` | 0.92 (DGX Spark unified memory; lower to 0.90 on host OOM) |
| `--attention-backend` | `flashinfer` (SM121-compatible; `trtllm_mla` is SM100-only) |
| `--chunked-prefill-size` | 8192 |
| `--disable-prefill-cuda-graph` | Required on DGX Spark per SGLang recipe |
| `--reasoning-parser` | `qwen3` |
| `--tool-call-parser` | `qwen3_coder` |
| `--trust-remote-code` | Required for qwen3_5 architecture |
| `--max-running-requests` | 8 (concurrency; see Memory History) |
| GDN state pool | `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 64` (~7 GB vs ~26.6 GB fp32 default) |
| MTP (EAGLE) | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (~3-4x decode) |
| Context | `CONTEXT_LENGTH=262144`, `YARN=0` in `.env`; >262K requires `YARN=1` (start.sh builds the YaRN rope override) |

Container flags: `--network host --ipc host --gpus all`

## Why NVFP4 / SGLang on GB10

The FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`) failed on GB10 for two reasons:
1. **TokenSpeed**: hybrid KV cache budget too small for the 56 GB GPU partition; TRTLLM attention kernels don't support SM121; Triton MHA kernel exceeds shared memory for head_dim=256
2. **vLLM**: TRTLLM FMHA kernel doesn't support SM121

SGLang with `--attention-backend flashinfer` works natively on SM121, and the NVFP4 checkpoint (~16.5 GB weights) fits comfortably within the GB10's unified memory, leaving ample room for KV cache and GDN state.

## Memory History

512K context + 12 concurrent + 0.95 mem fraction exhausted the 128 GB unified memory (NVRM `NV_ERR_NO_MEMORY`, heavy swap). Current default is 262K + 8 concurrent + 0.92.

## Troubleshooting

- **Container exits before ready**: `tail -200 .sglang.log`
- **Host OOM (GB10 shared memory)**: lower `--mem-fraction-static` to `0.90` in `start.sh`
- **MTP fails with FlashInfer**: use `--attention-backend triton` or remove the `--speculative-*` flags
- **Re-downloading on every start**: confirm `.cache/huggingface` is mounted at `/root/.cache/huggingface` and `HF_HOME=/root/.cache/huggingface` is set in the container
