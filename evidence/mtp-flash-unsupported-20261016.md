# mtp-flash (DFLASH) — unsupported by the current image, 2026-10-16

Two failures, in order, while bringing up `mtp-flash` on
`lmsysorg/sglang:qwen38-27b` (SGLang `0.0.0.dev0+qwen38.27b.g561c8f3`,
`nightly-dev-20260814-c4271c3f`):

1. `extra_buffer_lazy unsupported with DFLASH; use
   --mamba-radix-cache-strategy extra_buffer.`
   — fixed in both flash profiles (`MAMBA_RADIX_CACHE_STRATEGY=extra_buffer`).
2. `ValueError: Cannot find model module. 'DFlash2DraftModel' is not a
   registered model in the Transformers library ... and 'AutoModel' is not
   present in the model config's 'auto_map'.`

## Root cause (verified in the image)

- The image registers only `DFlashDraftModel` / `DFlashLagunaForCausalLM`
  (`srt/models/dflash.py`) plus the DSpark classes (`Qwen3DSparkModel`,
  `DSparkDraftModel`, `srt/models/dspark.py`).
- The DFlash 2 checkpoints (`incoai/Qwen3.8-27B-DFlash2`, mirror
  `z-lab/Qwen3.8-27B-DFlash2`) declare
  `architectures: ["DFlash2DraftModel"]`, `model_type: qwen3`, `auto_map: null`
  — a class this image does not have. There is no draft-architecture override
  flag (`--speculative-draft-model-path` and friends only).
- The checkpoint's own README recommends installing SGLang from source
  (`pip install sglang @ git+...`), i.e. DFlash 2 landed after this image's
  nightly.

## Consequence

DFLASH cannot be evaluated on this deployment image; the DSPARK/DSpark path
remains the supported spec-decode runtime here. The `mtp-flash` and
`dflash-d8` profiles are retained (marked unsupported) for a future image.

Follow-up idea (separate experiment — a new image is a new runtime baseline,
so it re-opens the whole sweep): the RTX PRO 6000 node ran DFLASH on
`lmsysorg/sglang:latest` (v0.5.21); testing an equivalent DFlash2-capable
image on GB10 would need the full capture → smoke → bench cycle again.
