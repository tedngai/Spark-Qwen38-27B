# dspark-d4 — invalid profile (engine constraint), 2026-10-16 ~04:58Z

The `dspark-d4` candidate (DSPARK with 4 draft tokens) **cannot start**: DSpark
requires `speculative_num_draft_tokens == gamma + 1 == 8`.

```
ValueError: DSpark speculative_num_draft_tokens must equal gamma + 1 (= 8 for gamma=7),
but got speculative_num_draft_tokens=4.
```

Retained per evidence policy (failed probes are kept and marked). Consequences:

- DSPARK's draft-token count is fixed at 8 on this build; the "fewer draft
  tokens under load" idea only applies to **DFLASH** (block size).
- The profile slot was repurposed to `dflash-d8` (DFLASH, 8 draft tokens,
  identical surroundings to `mtp-flash`) so the d4-vs-d8 comparison gets done
  inside the DFLASH family — mirroring the RTX PRO 6000 node's A/B (which
  chose 4 over 8).
