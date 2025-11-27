from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

BASE_TASK_DIR = Path(__file__).resolve().parent / "runtime" / "base_tasks"
RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"
LEADERBOARD_ROOT = Path(__file__).resolve().parent / "leaderboard"
OJ_TEMPLATE = Path(__file__).resolve().parent / "OJ-template.sh"

AGENT_FLAGS = {"claude", "codex", "gemini", "qwen", "minimal"}
SIMPLE_BASE_PORT = 18080
MINIMAL_BASE_PORT = 8080
MINIMAL_MODELS = {
    "deepseek_v3",
    "doubao_seed",
    "gemini",
    "claude_4_sonnet",
    "claude_3_7_sonnet",
    "gpt_5",
    "gpt_4o",
    "o1_mini",
    "qwen3_coder",
    "glm4_5",
    "longcat_flash_chat",
}
MINIMAL_MODEL_LIST = sorted(MINIMAL_MODELS)
MINIMAL_BASE_PORT = 8080
MINIMAL_PORT_MAP = {model: MINIMAL_BASE_PORT + idx for idx, model in enumerate(MINIMAL_MODEL_LIST)}
MINIMAL_DEFAULT_MODEL = MINIMAL_MODEL_LIST[0]
CASE_FIELD_MAX_CHARS = 2000  # cap per-case text fields to keep reports small


def ensure_base_tasks(seed: int):
    if BASE_TASK_DIR.exists():
        return
    print("[INFO] Base tasks not found. Generating via prepare_tasks.py")
    subprocess.run([sys.executable, str(Path(__file__).parent / "prepare_tasks.py"), "--seed", str(seed)], check=True)


def prepare_agent_workspace(agent_label: str, round_dir: Path, shared_meta: Path, shared_runner: Path) -> Path:
    agent_dir = round_dir / agent_label
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(
        BASE_TASK_DIR,
        agent_dir,
        ignore=shutil.ignore_patterns("problems.json", "oj_runner.py", "OJ-template.sh", "OJ.sh"),
    )
    render_oj_script(agent_dir / "OJ.sh", shared_runner, shared_meta)
    return agent_dir


def render_oj_script(dest: Path, runner: Path, meta: Path):
    template_path = OJ_TEMPLATE if OJ_TEMPLATE.exists() else (BASE_TASK_DIR / "OJ-template.sh")
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("__OJ_RUNNER_PATH__", str(runner))
    text = text.replace("__OJ_META_PATH__", str(meta))
    dest.write_text(text, encoding="utf-8")
    os.chmod(dest, 0o755)


def _read_leaderboard_content(path: Path, max_chars: int = 5000) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def build_prompt(
    agent_label: str,
    round_id: int,
    agent_dir: Path,
    prev_agent_dir: Optional[Path],
    prev_round_root: Optional[Path],
    leaderboard_path: Optional[Path],
    agent_list: List[str],
) -> str:
    lines = [
        f"You are {agent_label}, participating in OIBench Round {round_id}.",
        f"Your workspace: {agent_dir}",
        "You have problems 0.md-10.md. 0 is an example (solution 0.cpp from the dataset). Problems 1-10 are the tasks to solve.",
        "Read all problems first, then solve in an order that maximizes solved count (do easy ones first).",
        "Write Python solutions to <id>.py for ids 1..10.",
        f"After writing code, self-test with `cd {agent_dir} && bash OJ.sh <id> <id>.py`. Time limit per test case: 2s.",
        "Total agent time budget: 30 minutes. Generate code without waiting for confirmation.",
    ]
    if round_id > 1:
        lines.append(f"Participants this round: {', '.join(agent_list)}.")
        if prev_agent_dir and prev_agent_dir.exists():
            lines.append(f"Your previous round code: {prev_agent_dir}")
        if leaderboard_path and leaderboard_path.exists():
            lines.append("Last round leaderboard (text):\n" + _read_leaderboard_content(leaderboard_path))
        if prev_round_root and prev_round_root.exists():
            lines.append(f"Please review your own and other agents' code/performance from last round here: {prev_round_root}")
            lines.append("Analyze how to optimize performance and borrow useful ideas where appropriate.")
    return "\n".join(lines)


def run_agent(
    agent_label: str,
    agent_dir: Path,
    prompt: str,
    timeout_minutes: int,
    simple_model: str,
    log_path: Path,
):
    is_minimal = agent_label == "minimal" or agent_label.startswith("minimal[")
    start = time.time()
    try:
        if is_minimal:
            quoted_prompt = shlex.quote(prompt)
            minimal_port = MINIMAL_PORT_MAP.get(simple_model, MINIMAL_PORT_MAP[MINIMAL_DEFAULT_MODEL])
            activate = "source /Users/panly/Documents/Minimal-CodeAgent/.venv/bin/activate"
            cmd = [
                "bash",
                "-lc",
                (
                    f"{activate} && "
                    f"python /Users/panly/Documents/Minimal-CodeAgent/run_agent.py "
                    f"--prompt {quoted_prompt} --workdir {shlex.quote(str(agent_dir))} "
                    f"--port {minimal_port}"
                ),
            ]
        else:
            cmd = ["ai-auto", "--agent", agent_label, "--prompt", prompt]
        proc = subprocess.Popen(
            cmd,
            cwd=agent_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False if not is_minimal else False,
            env=None,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_minutes * 60)
            status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            status = "timeout"
    except FileNotFoundError:
        stdout = ""
        if is_minimal:
            stderr = "Minimal-CodeAgent environment missing; ensure .venv is available and adk is installed"
        else:
            stderr = "ai-auto not found; please install or adjust command"
        status = "not_found"
    elapsed = time.time() - start
    log_path.write_text((stdout or "") + "\n--- stderr ---\n" + (stderr or ""), encoding="utf-8")
    return {"status": status, "elapsed": elapsed, "log": str(log_path)}


def evaluate_agent(agent_dir: Path, problem_ids: List[int]):
    oj_cmd = ["bash", "OJ.sh", "", "", "--format", "json"]
    results = [None] * len(problem_ids)
    with ThreadPoolExecutor(max_workers=min(len(problem_ids), 8)) as ex:
        future_map = {}
        for idx, pid in enumerate(problem_ids):
            cmd = oj_cmd.copy()
            cmd[2] = str(pid)
            cmd[3] = f"{pid}.py"
            future = ex.submit(_run_eval_cmd, cmd, agent_dir)
            future_map[future] = (idx, pid)
        for fut in as_completed(future_map):
            idx, pid = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "problem_id": pid,
                    "status": "EVAL_ERROR",
                    "avg_time": 0.0,
                    "cases": [],
                    "message": str(e),
                }
            results[idx] = res
    return results


def _truncate_cases(result: dict, field_limit: int = CASE_FIELD_MAX_CHARS):
    """Limit extremely long per-case text fields to avoid oversized reports."""
    cases = result.get("cases")
    if not isinstance(cases, list):
        return result

    trimmed_cases = []
    for case in cases:
        if not isinstance(case, dict):
            trimmed_cases.append(case)
            continue
        trimmed = {}
        for k, v in case.items():
            if isinstance(v, str) and len(v) > field_limit:
                trimmed[k] = v[:field_limit] + f"...[truncated {len(v) - field_limit} chars]"
            else:
                trimmed[k] = v
        trimmed_cases.append(trimmed)

    result["cases"] = trimmed_cases
    return result


def _run_eval_cmd(cmd: List[str], cwd: Path):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    res = json.loads(proc.stdout)
    return _truncate_cases(res)


def summarize(results: List[dict]):
    pass_count = sum(1 for r in results if r.get("status") == "AC")
    ac_times = [r.get("avg_time", 0.0) for r in results if r.get("status") == "AC" and r.get("avg_time") is not None]
    avg_time = (sum(ac_times) / len(ac_times)) if ac_times else float("nan")
    def fmt(r):
        status = r.get("status")
        t = r.get("avg_time", 0.0)
        return f"{r['problem_id']}:{status}({t:.3f}s)" if status == "AC" else f"{r['problem_id']}:{status}"
    detail = " ".join(fmt(r) for r in results)
    return pass_count, avg_time, detail


def plot_scores(history: dict, agents: List[dict]):
    if not history:
        return
    rounds = sorted(history.keys())
    labels = [a["label"] for a in agents]
    for label in labels:
        ys = []
        for r in rounds:
            ys.append(history[r]["agents"].get(label, {}).get("pass_count", 0))
        plt.plot(rounds, ys, marker="o", label=label)
    plt.xlabel("Round")
    plt.ylabel("Solved Count")
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = LEADERBOARD_ROOT / "scores_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def write_history_txt(history: dict):
    if not history:
        return
    lines = ["Round\tRank\tAgent\tSolved\tAvgTime(s)\tRunTime(s)"]
    for r in sorted(history.keys(), key=lambda x: int(x)):
        lb_path = history[r]["leaderboard_txt"]
        lb = Path(lb_path)
        if not lb.exists():
            continue
        # recompute ranks from stored data to include run time
        agents = history[r]["agents"]
        rows = []
        for agent, data in agents.items():
            rows.append((agent, data["pass_count"], data["avg_time"], data.get("run", {}).get("elapsed", float("nan"))))
        rows.sort(key=lambda x: (-x[1], math.inf if math.isnan(x[2]) else x[2]))
        for idx, (agent, solved, avg_time, run_time) in enumerate(rows, 1):
            lines.append(f"{r}\t{idx}\t{agent}\t{solved}\t{avg_time:.3f}\t{run_time:.3f}")
    out_path = LEADERBOARD_ROOT / "history.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_agent_report(agent_label: str, round_id: int, round_dir: Path, evals: List[dict], run_info: dict):
    report_dir = LEADERBOARD_ROOT / f"round_{round_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{agent_label}.json"
    payload = {
        "agent": agent_label,
        "round": round_id,
        "round_dir": str(round_dir),
        "agent_dir": str(round_dir / agent_label),
        "evals": evals,
        "run": run_info,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def build_leaderboard(round_id: int, agents_data: dict):
    rows = []
    for agent, data in agents_data.items():
        rows.append(
            {
                "agent": agent,
                "pass_count": data["pass_count"],
                "avg_time": data["avg_time"],
                "detail": data["detail"],
                "report_path": str(data["report"]),
                "answer_path": str(data["agent_dir"]),
            }
        )
    rows.sort(key=lambda x: (-x["pass_count"], math.inf if math.isnan(x["avg_time"]) else x["avg_time"]))

    lb_dir = LEADERBOARD_ROOT / f"round_{round_id}"
    lb_dir.mkdir(parents=True, exist_ok=True)
    lb_json = lb_dir / "leaderboard.json"
    lb_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    header = "Rank\tAgent\tRunTime(s)\tSolved\tAvgTime(s)\tStatuses\tAnswerPath\tReportFile"
    lines = [header]
    for idx, row in enumerate(rows, 1):
        run_time = agents_data[row['agent']].get("run", {}).get("elapsed", float("nan"))
        lines.append(
            f"{idx}\t{row['agent']}\t{run_time:.3f}\t{row['pass_count']}\t{row['avg_time']:.3f}\t{row['detail']}\t{row['answer_path']}\t{row['report_path']}"
        )
    lb_txt = lb_dir / "leaderboard.txt"
    lb_txt.write_text("\n".join(lines), encoding="utf-8")
    return rows, lb_json, lb_txt


def run_round(
    round_id: int,
    agent_entries: List[dict],
    timeout_minutes: int = 30,
    previous_round: Optional[Path] = None,
    base_seed: int = 2025,
):
    ensure_base_tasks(base_seed)
    round_dir = RUNTIME_ROOT / f"round_{round_id}"
    round_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = round_dir / "shared"
    if shared_dir.exists():
        shutil.rmtree(shared_dir)
    shutil.copytree(BASE_TASK_DIR, shared_dir)
    shared_meta = shared_dir / "problems.json"
    shared_runner = Path(__file__).parent / "oj_runner.py"
    render_oj_script(shared_dir / "OJ.sh", shared_runner, shared_meta)

    agent_dirs = {}
    prompts = {}
    labels = [e["label"] for e in agent_entries]
    for idx, entry in enumerate(agent_entries):
        agent_label = entry["label"]
        model_override = entry.get("model") or MINIMAL_DEFAULT_MODEL
        agent_dir = prepare_agent_workspace(agent_label, round_dir, shared_meta.resolve(), shared_runner.resolve())
        prev_dir = previous_round / agent_label if previous_round else None
        leaderboard_path = LEADERBOARD_ROOT / f"round_{round_id-1}" / "leaderboard.txt" if round_id > 1 else None
        prompt = build_prompt(agent_label, round_id, agent_dir, prev_dir, previous_round, leaderboard_path, labels)
        prompt_path = round_dir / f"{agent_label}-prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        agent_dirs[agent_label] = agent_dir
        prompts[agent_label] = (prompt, model_override, prompt_path)

    run_infos = {}
    with ThreadPoolExecutor(max_workers=len(agent_entries)) as executor:
        futures = {}
        for entry in agent_entries:
            label = entry["label"]
            prompt, model_override, prompt_path = prompts[label]
            log_path = round_dir / f"{label}-run.log"
            futures[
                executor.submit(
                    run_agent,
                    label,
                    agent_dirs[label],
                    prompt,
                    timeout_minutes,
                    model_override,
                    log_path,
                )
            ] = label
        for fut in as_completed(futures):
            ag = futures[fut]
            run_infos[ag] = fut.result()

    problem_ids = list(range(1, 11))
    agents_data = {}
    for entry in agent_entries:
        label = entry["label"]
        evals = evaluate_agent(agent_dirs[label], problem_ids)
        pass_count, avg_time, detail = summarize(evals)
        report = write_agent_report(label, round_id, round_dir, evals, run_infos.get(label, {}))
        agents_data[label] = {
            "pass_count": pass_count,
            "avg_time": avg_time,
            "detail": detail,
            "report": str(report),
            "agent_dir": str(agent_dirs[label]),
        }

    leaderboard_rows, lb_json, lb_txt = build_leaderboard(round_id, agents_data)
    print(f"\nLeaderboard (Round {round_id}):")
    print(lb_txt.read_text(encoding="utf-8"))

    summary = {
        "round": round_id,
        "leaderboard_json": str(lb_json),
        "leaderboard_txt": str(lb_txt),
        "agents": agents_data,
    }
    return summary


def parse_agents(agent_str: Optional[str]):
    entries = []
    if not agent_str:
        for name in AGENT_FLAGS:
            entries.append({"label": name, "name": name, "model": None})
        return entries

    parts = [p.strip() for p in agent_str.split(",") if p.strip()]
    for p in parts:
        lower = p.lower()
        if lower.startswith("minimal[") and lower.endswith("]"):
            model_key = lower[len("minimal["):-1]
            model = model_key if model_key in MINIMAL_MODELS else MINIMAL_DEFAULT_MODEL
            label = f"minimal[{model_key}]"
            entries.append({"label": label, "name": "minimal", "model": model})
        elif lower == "minimal":
            model = MINIMAL_DEFAULT_MODEL
            label = f"minimal[{model}]"
            entries.append({"label": label, "name": "minimal", "model": model})
        elif p in AGENT_FLAGS:
            entries.append({"label": p, "name": p, "model": None})
        else:
            raise ValueError(f"Agent '{p}' is not supported. Allowed: {', '.join(sorted(AGENT_FLAGS))} or minimal[...]")
    return entries


def run_competition(rounds: int, agents: List[dict], timeout_minutes: int = 30, base_seed: int = 2025):
    ensure_base_tasks(base_seed)
    history = {}
    prev_round_dir = None
    agent_entries = agents
    for r in range(1, rounds + 1):
        print(f"\n===== Starting Round {r} =====")
        summary = run_round(
            r,
            agent_entries,
            timeout_minutes=timeout_minutes,
            previous_round=prev_round_dir,
            base_seed=base_seed,
        )
        history[r] = summary
        prev_round_dir = RUNTIME_ROOT / f"round_{r}"

    final_lb = history[max(history.keys())]["leaderboard_txt"]
    print("\n===== Competition finished =====")
    print(f"Final leaderboard (Round {max(history.keys())}): {final_lb}")

    (LEADERBOARD_ROOT / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    plot_scores(history, agents)
    write_history_txt(history)


def main():
    parser = argparse.ArgumentParser(description="Run OIBench arena")
    parser.add_argument("--round", type=int, default=None, help="run a single specified round number")
    parser.add_argument("--rounds", type=int, default=None, help="run multiple rounds (default 4)")
    parser.add_argument("--agents", type=str, default=None, help="comma separated agent list, default all")
    parser.add_argument("--timeout", type=int, default=30, help="per-agent time budget in minutes")
    parser.add_argument("--base-seed", type=int, default=2025, help="seed used when generating base tasks")
    args = parser.parse_args()

    agents = parse_agents(args.agents)

    if args.round is not None and args.rounds is not None:
        print("Please specify either --round or --rounds, not both.")
        sys.exit(1)

    if args.round is not None:
        run_round(
            args.round,
            agents,
            timeout_minutes=args.timeout,
            base_seed=args.base_seed,
        )
    else:
        total_rounds = args.rounds or 4
        run_competition(
            total_rounds,
            agents,
            timeout_minutes=args.timeout,
            base_seed=args.base_seed,
        )


if __name__ == "__main__":
    main()
