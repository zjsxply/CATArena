# OIBench: Multi-Agent Coding Arena

OIBench runs 4-round (configurable) competitive coding on the Hugging Face `AGI-Eval/OIBench` dataset with multiple agents (default: Claude Code, CodeX, Gemini-CLI, Qwen-Coder). It prepares tasks, gives each agent an isolated workspace, judges Python/C++ submissions, and publishes ACM-style leaderboards.

## Project layout
- `prepare_tasks.py` — sample 11 problems (0 is example) from OIBench and generate `runtime/base_tasks` (markdowns, canonical 0.cpp, OJ tools).
- `start_arena.py` — unified entrypoint for single or multi-round tournaments.
- `OJ.sh` / `oj_runner.py` — local judge; `bash OJ.sh <problem_id> <program_path>` (supports `.py`/`.cpp`, 2s per case).
- `runtime/` — generated tasks, per-round workspaces, shared judge/meta.
- `leaderboard/` — leaderboard JSON/TXT per round, history.json.
- `AGENTS.md` — agent-facing rules and prompt expectations.

## Dependencies
```bash
pip install -r oibench/requirements.txt
```
You need access to the `AGI-Eval/OIBench` dataset (cached or online) and a C++ compiler (g++/clang++) for judging cpp.

## Quick start
1) Generate tasks (English statements):
```bash
python3 oibench/prepare_tasks.py --seed 2025
```

2) Run a single round (agents comma-separated):
```bash
python3 oibench/start_arena.py --round 1 --agents "claude,codex,gemini,qwen"
```

3) Run full competition (default 4 rounds):
```bash
python3 oibench/start_arena.py --rounds 4 --agents "claude,codex,gemini,qwen"
```
- Base task seed can be set with `--base-seed` (default 2025).

### Minimal-CodeAgent support
- Invoke via arena using `minimal[model_key]` in `--agents`, e.g. `--agents "minimal[longcat_flash_chat],minimal[deepseek_v3]"`.
- Supported model keys: `deepseek_v3`, `doubao_seed`, `gemini`, `claude_4_sonnet`, `claude_3_7_sonnet`, `gpt_5`, `gpt_4o`, `o1_mini`, `qwen3_coder`, `glm4_5`, `longcat_flash_chat`.
- The arena will run: `conda activate minimalcodeagent && ADK_MODEL=<model> python /Users/panly/Documents/Minimal-CodeAgent/run_agent.py --prompt "<prompt>" --workdir <dir> --port <port>`.
- Ports start from `--simple-base-port` (default 18080) and increment per minimal agent in order.

## Judging
- Command: `bash OJ.sh <problem_id> <program_path> [--format json]`
- Languages: Python and C++ (auto-detected by extension)
- Per test timeout: 2 seconds; C++ compiled in a temp dir with hash-based binary name to avoid collisions.

## Rounds & learning
- Round 1: fresh start.
- Later rounds: prompts include your previous code path, last-round leaderboard content, and paths to other agents so you can learn/borrow.

## Outputs
- `leaderboard/round_<r>/leaderboard.json|txt` with rank, solved, avg time, statuses, answer path, report file.
- `leaderboard/history.json` aggregated results.

See `AGENTS.md` for agent-side instructions and `README-zh.md` for the Chinese version.
