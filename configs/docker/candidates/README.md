# Candidate launch profiles

Each `<name>.env` here is one launch configuration, sourced by
`configs/docker/run_qwen38_sglang.sh <name>` after the repo `.env`. Files are
plain shell assignments; anything not set falls back to the launcher defaults
(which equal `baseline.env`).

Tunable keys (all optional):

| Key | Default | Notes |
|---|---|---|
| `MEM_FRACTION` | 0.85 | `--mem-fraction-static` |
| `MAX_RUNNING_REQUESTS` | 8 | admission cap |
| `KV_CACHE_DTYPE` | fp8_e4m3 | |
| `ATTENTION_BACKEND` | flashinfer | `triton` is the fallback |
| `CHUNKED_PREFILL` | 2048 | `--chunked-prefill-size` |
| `MAMBA_SSM_DTYPE` | float32 | GDN state pool dtype |
| `MAMBA_FULL_MEMORY_RATIO` | 11.01 | GDN state memory sizing |
| `MAMBA_RADIX_CACHE_STRATEGY` | extra_buffer_lazy | |
| `MAX_MAMBA_CACHE_SIZE` | engine default | e.g. `MAX_RUNNING_REQUESTS*8` |
| `DISABLE_PREFILL_CUDA_GRAPH` | 0 | recipe sets 1; baseline runs 0 |
| `SPEC_ALGORITHM` | DSPARK | `DSPARK` \| `EAGLE` \| empty (off) |
| `SPEC_DRAFT_MODEL` | DSpark checkpoint | DSPARK only |
| `SPEC_DRAFT_ATTENTION_BACKEND` | flashinfer | DSPARK only |
| `SPEC_NUM_STEPS`, `SPEC_EAGLE_TOPK`, `SPEC_NUM_DRAFT_TOKENS` | engine defaults | EAGLE needs the first two |
| `CONTEXT_LENGTH`, `YARN` | 262144 / 0 | >262144 requires `YARN=1` |

Rules:

- One profile at a time, by hand, cold start per profile
  (`tests/run_profile.sh <name>` does stop → start → measure → stop).
- A candidate is only "validated" once its run is recorded under `evidence/`;
  no claims without evidence.
- Keep the hypothesis for each candidate in `docs/NEXT_EXPERIMENT.md` and the
  findings in `docs/results.md`.
- No systemd, no restart policies, no retry loops (`docs/TESTING.md`).
