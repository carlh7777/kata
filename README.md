# PromptForge

PromptForge is an objective prompt optimization repo for SN74/Gittensor.

It is designed to generate repo-specific prompts, evaluate them against repo-specific tasks,
compare them with baseline/manual prompts, and report whether they improve verified task success.

## Registration MVP Interfaces

```bash
promptforge generate --repo <repo-path> --mode contributor
promptforge baseline --repo <repo-path>
promptforge eval --repo <repo-path> --eval-pack evals/<repo-name>
promptforge report --run <run-id>
```

These interfaces are created first so each MVP function can be implemented behind a stable CLI.
