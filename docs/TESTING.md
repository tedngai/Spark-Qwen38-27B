# Testing policy — manual runs only

Every test on this host is started **by hand**. Nothing auto-starts and
nothing auto-restarts.

## Hard rules

1. **No systemd, no cron, no supervisors, no retry loops.** Do not create
   `.service`/`.timer` units, `systemd-run` units, or `Restart=always`
   policies. The launcher runs containers with `--restart no`; a crashed
   server stays down until a human retries it.
2. **One server at a time.** The server runs on the GB10's 128 GB unified
   memory; two servers at once exhaust it (the 2026-10-13 OOM). The launcher
   refuses to start if the container is already running.
3. **Cold start per profile.** Every profile is measured from a fresh
   start — `tests/run_profile.sh <profile>` stops, starts, measures, stops.
   It is a one-shot, hand-invoked cycle: it never loops, never retries, and
   aborts on the first failure, leaving the server up for inspection.
4. **Downloads run once, by hand.** `configs/docker/prefetch.sh` performs a
   single image pull + model download pass and exits; it must never launch a
   server.
5. **No claims without evidence.** A profile is only "validated" once its
   run is recorded under `evidence/` (see `evidence/README.md`).

## Scripts

| Script | What it does | Log |
|---|---|---|
| `configs/docker/run_qwen38_sglang.sh [profile]` | start SGLang for a profile, wait for readiness, exit | `logs/qwen38_sglang.log` |
| `configs/docker/stop.sh` | stop/remove the container + log follower | — |
| `configs/docker/prefetch.sh` | one-time image pull + model download | `logs/prefetch.log` |
| `tests/run_profile.sh <profile>` | one-shot benchmark cycle (stop → start → capture → smoke → A/B bench → suite → stop) | `logs/suite-*.log` |
| `tests/smoke_test.py`, `tests/benchmark.py`, `tests/capture_server.py` | measure a running server (never start/stop it) | stdout + `evidence/` |

## Before starting a server

```bash
docker ps                 # no leftover qwen3.8-27b-nvfp4-sglang container
nvidia-smi                # GB10 idle
```

## Start / verify / stop

```bash
configs/docker/run_qwen38_sglang.sh baseline   # blocks until ready (~3.5 min warm)
tail -f logs/qwen38_sglang.log                 # from another terminal
curl -s localhost:8000/health                  # 200 before sending requests
python3 tests/smoke_test.py                    # optional end-to-end check
configs/docker/stop.sh                         # stop
```

## After a crash

Nothing restarts by itself. Check `docker ps -a` and `nvidia-smi`, read
`logs/qwen38_sglang.log` (and the previous-container tail in
`logs/qwen38_sglang_prev.log`), then restart by hand only if you want to
retry.

## Legacy notes

- Pre-packaging, the container ran with `--restart unless-stopped`; the
  packaged launcher deliberately uses `--restart no`.
- The TokenSpeed/vLLM containers from the 2026-08 migration are removed
  automatically by the launcher and `stop.sh` if they ever reappear.
