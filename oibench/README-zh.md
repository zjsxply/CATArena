# OIBench：多 Agent 代码竞技场

OIBench 基于 Hugging Face `AGI-Eval/OIBench` 数据集，支持多轮（默认 4 轮）多 Agent 并行解题、评测与榜单发布，默认 Agent 含 Claude Code、CodeX、Gemini-CLI、Qwen-Coder。

## 目录
- `prepare_tasks.py`：抽取 11 道题（0 为示例）生成题面、0.cpp、OJ 工具。
- `start_arena.py`：统一入口，单轮或多轮竞技。
- `OJ.sh` / `oj_runner.py`：本地评测，`bash OJ.sh <problem_id> <program_path>`（支持 Python/C++，单测 2s）。
- `runtime/`：生成的题面与每轮工作区。
- `leaderboard/`：每轮榜单与历史结果。
- `AGENTS.md`：面向 Agent 的操作规则。

## 依赖
```bash
pip install -r oibench/requirements.txt
```
需可访问（或已缓存）`AGI-Eval/OIBench`，并安装 g++/clang++ 用于评测 C++。

## 快速开始
1) 生成题面：
```bash
python3 oibench/prepare_tasks.py --seed 2025
```
2) 跑单轮：
```bash
python3 oibench/start_arena.py --round 1 --agents "claude,codex,gemini,qwen"
```
3) 跑全流程（默认 4 轮）：
```bash
python3 oibench/start_arena.py --rounds 4 --agents "claude,codex,gemini,qwen"
```

## 评测说明
- 用法：`bash OJ.sh <problem_id> <program_path> [--format json]`
- 语言：Python / C++（按扩展名自动选择）
- 时限：单测 2 秒；C++ 编译在临时目录，二进制名含源码哈希避免冲突。

## 轮次与学习
- 第 1 轮：独立作答。
- 第 2 轮及之后：prompt 会给出上一轮自己的代码路径、上一轮榜单全文，以及其他 Agent 的目录，鼓励学习/复用。

## 输出
- `leaderboard/round_<r>/leaderboard.json|txt`：排名、通过数、平均用时、各题状态、答题路径、报告文件。
- `leaderboard/history.json`：所有轮次汇总。

更多 Agent 侧细节见 `AGENTS.md`。
