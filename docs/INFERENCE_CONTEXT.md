# Inference context — hardware, model, runtime

Background for why the launch configuration looks the way it does. Status in
`README.md` is a recorded snapshot, not a live check.

## Hardware

- NVIDIA DGX Spark, **GB10** GPU (SM121, Blackwell), single GPU, 128 GB
  unified LPDDR5X (GPU and host share the memory — "VRAM" and "RAM" are the
  same pool).
- Because memory is unified, an over-committed server can swap and take the
  whole box down (NVRM `NV_ERR_NO_MEMORY`), not just OOM a process.
- Public DGX Spark kernel gaps found during bring-up: TRTLLM attention
  kernels are SM100-only, and the Triton MHA kernel needs more shared memory
  than SM121 offers for head_dim=256 — FlashInfer is the supported backend.

## Model

- `RadixArk/Qwen3.8-27B-NVFP4`: dense 27B, NVFP4 W4A4 weights + FP8
  projections, vision-language (image + video), hybrid GDN/full-attention
  layers, MTP-trained, native 262,144-token context (~16.5 GB weights).
- `RadixArk/Qwen3.8-27B-DSpark`: speculative draft model (DSpark) used by the
  baseline profile.
- Thinking mode is on by default; responses carry `reasoning_content`.
  Requests with a small `max_tokens` budget can return reasoning only — give
  short Q&A ≥ ~100 tokens.

## Runtime

- Image: `lmsysorg/sglang:qwen38-27b` (official SGLang image for Qwen3.8;
  reports SGLang `0.0.0.dev0+qwen38.27b.g561c8f3`). Per the SGLang cookbook
  recipe for DGX Spark / NVFP4: https://docs.sglang.io/cookbook/qwen3.8-27b/
- Container: host network, `--ipc host`, `--gpus all`, port 8000, HF cache
  bind-mounted from `.cache/huggingface`. No authentication.
- Measured startup (warm, 2026-10-14): weight load ~141 s, ready in ~208–213 s.
- Observed KV pool at 262K context / mem 0.85 / maxreq 8 (fp8_e4m3 KV):
  `max_total_num_tokens=956,756`.
- GDN state pool: `--mamba-ssm-dtype float32 --mamba-full-memory-ratio 11.01
  --mamba-radix-cache-strategy extra_buffer_lazy` (tuned; the fp32 default
  pool is much larger).
- Measurement tools: `tests/benchmark.py` (stdlib A/B harness, uses
  `/generate` with forced decodes) and `vllm-benchmark-suite` 3.1.1
  (`~/miniconda3/bin/vllm-bench`, OpenAI endpoint, saved under `evidence/`).

## Migration history (why SGLang)

1. **v1 — TokenSpeed + `Qwen/Qwen3.8-27B-FP8`**: failed on GB10 (hybrid KV
   cache budget too small for the memory partition; TRTLLM attention
   unsupported on SM121; Triton MHA exceeds shared memory at head_dim=256).
2. **v2 — vLLM + `unsloth/Qwen3.8-27B-NVFP4`**: TRTLLM FMHA kernel does not
   support SM121.
3. **v3 (current) — SGLang + `RadixArk/Qwen3.8-27B-NVFP4`**: works on SM121
   with `--attention-backend flashinfer`; NVFP4 weights fit comfortably in
   unified memory.

## Memory history (why current values)

- 2026-10-13: 512K context (YaRN) + 12 concurrent + `--mem-fraction-static
  0.95` exhausted the 128 GB unified memory (heavy swap, NVRM
  `NV_ERR_NO_MEMORY`). Current baseline: 262K native context, 8 concurrent,
  0.85.
- Historic fallback levers: `--mem-fraction-static 0.90`, fewer concurrent
  users, bf16 GDN state pool (`MAMBA_SSM_DTYPE=bfloat16`,
  `MAMBA_FULL_MEMORY_RATIO=4.21`).
