# Agent rules for the Qwen3.8-27B-Deployment repository

Packaged Qwen3.8-27B SGLang deployment project for the DGX Spark (GB10):
profile-driven launch scripts, measurement harnesses, captured evidence, and
documentation. This file is the operating manual for AI agents managing it.

## Repo map

- `configs/docker/run_qwen38_sglang.sh` — the only launcher. `[profile]`
  argument loads `configs/docker/candidates/<profile>.env` (default
  `baseline`); waits for readiness; `--restart no`.
- `configs/docker/stop.sh` / `prefetch.sh` — stop container / one-time pull +
  download.
- `tests/` — measurement scripts. `run_profile.sh` is the one-shot
  stop→start→capture→smoke→bench→stop cycle; the others measure a running
  server and never start/stop it.
- `evidence/` — committed, timestamped reports. `docs/` — policy, context,
  sweep plan, results. `logs/`, `.cache/`, `.env` — local only, never commit.

## Operational rules (hard)

- **Manual runs only.** No systemd/cron/supervisors, no restart policies, no
  retry loops. One server at a time; the launcher refuses to double-start.
  `tests/run_profile.sh` is hand-invoked and aborts on first failure —
  never wrap it in a loop.
- **Cold start per measured profile.** Never A/B two configs against one
  running process.
- The live artifact is the container `qwen3.8-27b-nvfp4-sglang` on port 8000,
  started by hand. Verify `/health` and `/get_server_info` before assessing.
- **Never claim a configuration is validated without runtime evidence saved
  under `evidence/`.**
- Do not run the full sweep yourself; the user drives benchmarks. Assist:
  start/stop/restart, tail logs, prepare profiles, analyze evidence, update
  docs.
- Model/config changes: edit a candidate profile (or add a new one) rather
  than touching launcher defaults; document the hypothesis in
  `docs/NEXT_EXPERIMENT.md` and the finding in `docs/results.md`.

## Workflow

- **Always commit changes.** After any change, commit with a descriptive
  message. Small, incremental commits. Do not leave the tree dirty.
- After adding/removing/renaming/materially changing files, regenerate
  `docs/package-manifest.json` (`python3 tests/gen_manifest.py`) before
  committing.
- Evidence naming: `<profile>-<kind>-<UTC YYYYMMDDTHHMMSSZ>.json`; never
  overwrite an existing report; retain failed/invalid probes and mark them.
- Never commit: model weights or caches (`.cache/`), logs (`logs/`), `.env`,
  shell history, tokens. `.gitignore` is a guardrail, not permission.

## Server info

- Endpoint: `http://0.0.0.0:8000/v1` (host network, no auth — trusted
  networks only); container `qwen3.8-27b-nvfp4-sglang`.
- Recommended profile (first sweep winner, 2026-10-16): `combo-bf16-c12`;
  fallback: `baseline` (`docs/results.md`).
- Model: `RadixArk/Qwen3.8-27B-NVFP4`; reasoning parsed into
  `reasoning_content`; thinking models need ≥ ~100 output tokens for content.
- Useful probes: `curl -s localhost:8000/health`, `/get_server_info`,
  `/v1/models`; server log `logs/qwen38_sglang.log`.

## Publishing

No remote is configured yet (local-only commits by agreement, 2026-10-16).
When publishing is re-enabled, prefer the Pratt Gitea remote
`http://itgit.pratt.edu/tngai/Qwen3.8-27B-Deployment.git` using `GITEA_TOKEN`
from `.env` (the host only resolves `itgit.pratt.edu` on the Pratt network),
or a GitHub remote under `tedngai` with a fine-grained PAT in `.env`
(`GITHUB_PAT`). For either, use a repo-local credential helper that reads the
token from `.env`:

```bash
git config --local credential.helper \
  '!f() { . /home/tngai/data/vllm-qwen38/.env; echo username=x-access-token; echo "password=${GITEA_TOKEN:-$GITHUB_PAT}"; }; f'
```

Never print a token, put one in a remote URL, copy one into tracked files, or
stage `.env` (verify with `git check-ignore .env`).

## Memory history (why the baseline values)

- 2026-10-13: 512K context + 12 concurrent + `--mem-fraction-static 0.95` →
  host OOM (NVRM `NV_ERR_NO_MEMORY`, heavy swap).
- Baseline since: 262K + 8 concurrent + 0.85 (see `docs/INFERENCE_CONTEXT.md`).
