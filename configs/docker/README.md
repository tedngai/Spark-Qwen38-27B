# Docker configurations

`run_qwen38_sglang.sh` is the single profile-driven launcher for the
Qwen3.8-27B NVFP4 SGLang container on the DGX Spark (GB10). It takes a profile
name (default `baseline`), loads `candidates/<profile>.env` on top of its
defaults, runs the container detached with `--restart no` (manual runs only —
see `docs/TESTING.md`), and blocks until `/v1/models` answers.

```bash
./configs/docker/run_qwen38_sglang.sh baseline    # start (waits for readiness)
./configs/docker/stop.sh                          # stop + remove
tail -f logs/qwen38_sglang.log                    # follow server log
```

- `candidates/` holds the launch profiles: one small `.env` per experiment
  (plus `baseline.env`, the validated reference configuration). Profiles
  override the launcher defaults; see `candidates/README.md` for the knobs
  and the rules.
- `prefetch.sh` is the one-time image pull + model download (~16 GB + draft
  model into `.cache/huggingface`).
- `stop.sh` stops/removes the container and the log follower. Logs are
  preserved under `logs/`.

Ports/transport: host network, port 8000, no authentication — do not expose
to an untrusted network.

All files are host-specific: review the image tag, model paths, cache mount,
port, and memory settings before reuse on another host.
