import argparse
import json
import shutil
import subprocess
import sys
import time
import tempfile
import hashlib
from pathlib import Path
from statistics import mean

DEFAULT_MAX_TIME = 2.0
PRINT_MAX_LEN = 200  # characters to show before folding
CPP_COMPILE_TIMEOUT = 30


def load_meta(workdir: Path, meta_path: Path | None = None):
    meta_path = meta_path or (workdir / "problems.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"problems.json not found in {workdir}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "problems" in data:
        data = data["problems"]
    return {str(p["local_id"]): p for p in data}


def normalize_tests(problem):
    tests = []
    for key in ("public_tests", "hidden_tests"):
        for t in problem.get(key, []) or []:
            if isinstance(t, dict):
                inp = t.get("input") or t.get("in") or t.get("stdin") or ""
                out = t.get("output") or t.get("out") or t.get("stdout") or ""
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                inp, out = t[0], t[1]
            else:
                continue
            tests.append({"input": str(inp), "output": str(out)})
    return tests


def run_single(exec_cmd: list[str], test: dict, max_time: float):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            exec_cmd,
            input=test["input"],
            text=True,
            capture_output=True,
            timeout=max_time,
        )
        runtime = time.perf_counter() - start
    except subprocess.TimeoutExpired:
        return {"status": "TLE", "time": max_time, "stdout": "", "stderr": "timeout"}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {"status": "RE", "time": runtime, "stdout": stdout, "stderr": stderr}

    expected = (test.get("output") or "").strip()
    got = stdout.strip()
    if got == expected:
        status = "AC"
    else:
        status = "WA"
    return {"status": status, "time": runtime, "stdout": stdout, "stderr": stderr, "expected": expected, "got": got}


def _compile_cpp(source: Path, build_root: Path) -> Path:
    build_root.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()[:16]
    binary = build_root / f"{source.stem}_{h}"
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("C++ compiler not found (g++/clang++)")
    # macOS typically disallows static linking; keep portable flags
    cmd = [compiler, "-std=c++17", "-O2", "-pipe", str(source), "-o", str(binary)]
    try:
        subprocess.run(cmd, check=True, timeout=CPP_COMPILE_TIMEOUT, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"compile error: {e.stderr.decode('utf-8', 'ignore') if e.stderr else e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("compile timeout")
    return binary


def evaluate(problem_id: str, workdir: Path, max_time: float, program_path: Path | None = None, meta_path: Path | None = None):
    meta = load_meta(workdir, Path(meta_path) if meta_path else None)
    if str(problem_id) not in meta:
        raise KeyError(f"problem {problem_id} not found in problems.json")
    problem = meta[str(problem_id)]
    code_path = (workdir / program_path) if program_path else (workdir / f"{problem_id}.py")
    if not code_path.exists():
        return {
            "problem_id": problem_id,
            "status": "NO_SOLUTION",
            "avg_time": 0.0,
            "cases": [],
            "message": f"{code_path.name} not found",
        }

    tests = normalize_tests(problem)
    if not tests:
        return {
            "problem_id": problem_id,
            "status": "NO_TEST",
            "avg_time": 0.0,
            "cases": [],
            "message": "No tests provided in problems.json",
        }

    cases = []

    suffix = code_path.suffix.lower()
    exec_cmd: list[str]

    if suffix == ".py":
        exec_cmd = [sys.executable, str(code_path)]
        for t in tests:
            cases.append(run_single(exec_cmd, t, max_time))
    elif suffix == ".cpp":
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="oj_build_"))
            binary = _compile_cpp(code_path, tmp_dir)
            exec_cmd = [str(binary)]
            for t in tests:
                cases.append(run_single(exec_cmd, t, max_time))
        except Exception as e:
            return {
                "problem_id": problem_id,
                "status": "CE",
                "avg_time": 0.0,
                "cases": [],
                "message": str(e),
            }
    else:
        return {
            "problem_id": problem_id,
            "status": "UNSUPPORTED",
            "avg_time": 0.0,
            "cases": [],
            "message": f"Unsupported file type: {suffix}",
        }

    statuses = [c["status"] for c in cases]
    if all(s == "AC" for s in statuses):
        status = "AC"
    elif any(s == "TLE" for s in statuses):
        status = "TLE"
    elif any(s == "RE" for s in statuses):
        status = "RE"
    else:
        status = "WA"

    avg_time = mean([c.get("time", 0.0) for c in cases]) if cases else 0.0
    return {
        "problem_id": problem_id,
        "status": status,
        "avg_time": avg_time,
        "cases": cases,
    }


def _shorten(text: str, limit: int = PRINT_MAX_LEN) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " ...<folded>... " + text[-limit // 2 :]


def main():
    parser = argparse.ArgumentParser(description="Run local OJ for a single problem")
    parser.add_argument("--problem-id", required=True, help="problem id, e.g., 1")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="directory that contains problems.json and solution files")
    parser.add_argument("--program-path", type=Path, default=None, help="path to solution file (py or cpp)")
    parser.add_argument("--meta-path", type=Path, default=None, help="path to problems.json (if not in workdir)")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME, help="per-test timeout in seconds")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    args = parser.parse_args()

    result = evaluate(args.problem_id, args.workdir, args.max_time, args.program_path, args.meta_path)

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # human readable
    print(f"Problem {result['problem_id']} => {result['status']} (avg {result['avg_time']:.3f}s)")
    for i, c in enumerate(result.get("cases", [])):
        status = c.get("status")
        t = c.get("time", 0.0)
        print(f"  Case {i+1}: {status} ({t:.3f}s)")
        if status in {"WA", "RE"}:
            exp = c.get("expected", "")
            got = c.get("got", c.get("stdout", "")).strip()
            if status == "WA":
                print(f"    expected: {_shorten(exp)}")
                print(f"    got     : {_shorten(got)}")
            if c.get("stderr"):
                print(f"    stderr  : {_shorten(c['stderr'].strip())}")
    if result.get("message"):
        print(result["message"])


if __name__ == "__main__":
    main()
