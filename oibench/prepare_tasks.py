import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
BASE_TASK_DIR = ROOT / "runtime/base_tasks"


DEFAULT_TIME_LIMIT = 2.0


def _normalize_tests(test_case):
    """OIBench uses key `test_case`: list of {'input': str, 'output': str}."""
    tests = []
    for t in test_case or []:
        if not isinstance(t, dict):
            continue
        inp = str(t.get("input", ""))
        out = str(t.get("output", ""))
        tests.append({"input": inp, "output": out})
    return tests


def _load_hf(sample_size: int, seed: int):
    path = "AGI-Eval/OIBench"
    # Official release uses split "test"
    ds = load_dataset(path, split="test")
    if len(ds) < sample_size:
        sample_size = len(ds)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), sample_size)
    return [ds[int(i)] for i in idx], "huggingface"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _write_markdown(item: dict, local_id: int, target: Path):
    prob_en = item.get("prob_en") or "(no English statement provided)"
    prob_zh = item.get("prob_zh")
    desc = prob_en if prob_en.strip() else (prob_zh or "(no description)")
    tests_all = _normalize_tests(item.get("test_case"))
    public_tests = tests_all[:2]
    hidden_tests = tests_all[2:]
    time_limit = DEFAULT_TIME_LIMIT

    lines = [
        f"# Problem {local_id}",
        # f"Time Limit: {time_limit}s",
        "",
        desc,
    ]

    # if public_tests:
    #     lines.append("\n## Samples (public tests)")
    #     for i, t in enumerate(public_tests):
    #         lines.append(f"### Sample {i+1}\nInput:\n````\n{t['input']}\n````\nOutput:\n````\n{t['output']}\n````")
    # if hidden_tests:
    #     lines.append("\n## Hidden tests (for evaluation only)\nCount: %d" % len(hidden_tests))

    target.write_text("\n".join(lines), encoding="utf-8")


def _write_meta(problems, source: str, seed: int, out_path: Path):
    meta = {
        "source": source,
        "sample_seed": seed,
        "problem_count": len(problems),
        "problems": problems,
    }
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare OIBench tasks")
    parser.add_argument("--num-problems", type=int, default=11, help="total problems to sample (includes example 0)")
    parser.add_argument("--seed", type=int, default=2025, help="random seed for sampling")
    parser.add_argument("--output-dir", type=Path, default=BASE_TASK_DIR, help="where to write base tasks")
    args = parser.parse_args()

    sample_n = args.num_problems
    if sample_n < 1:
        print("num-problems must be >=1", file=sys.stderr)
        sys.exit(1)

    _ensure_dir(args.output_dir)

    items, source = _load_hf(sample_n, args.seed)
    print(f"Loaded {sample_n} tasks from OIBench (AGI-Eval/OIBench)")

    problems_meta = []

    for local_id, item in enumerate(items):
        tests_all = _normalize_tests(item.get("test_case"))
        public_tests = tests_all[:2]
        hidden_tests = tests_all[2:]
        canonical_solution = item.get("canonical_solution")
        time_limit = DEFAULT_TIME_LIMIT
        title = f"OIBench {item.get('id', local_id)}"
        desc = item.get("prob_en") or item.get("prob_zh") or "(no description)"
        hf_id = item.get("id") or item.get("hf_id") or f"sample-{local_id}"

        # write markdown in English
        md_path = args.output_dir / f"{local_id}.md"
        _write_markdown(
            {
                "id": hf_id,
                "prob_en": item.get("prob_en"),
                "prob_zh": item.get("prob_zh"),
                "test_case": tests_all,
            },
            local_id,
            md_path,
        )

        problems_meta.append({
            "local_id": local_id,
            "hf_id": hf_id,
            "title": title,
            "description": desc,
            "public_tests": public_tests,
            "hidden_tests": hidden_tests,
            "time_limit": time_limit,
            "canonical_solution": canonical_solution,
        })

    # write meta json
    _write_meta(problems_meta, source, args.seed, args.output_dir / "problems.json")

    # write example solution from dataset canonical_solution (default to C++)
    canonical = problems_meta[0].get("canonical_solution")
    if canonical:
        canonical_str = str(canonical)
        # simple heuristic: detect cpp vs python
        is_cpp = ("#include" in canonical_str) or ("int main" in canonical_str) or ("using namespace std" in canonical_str)
        ext = "cpp" if is_cpp else "py"
        example_path = args.output_dir / f"0.{ext}"
        example_path.write_text(canonical_str, encoding="utf-8")
    else:
        # fallback portable C++ echo
        example_cpp = args.output_dir / "0.cpp"
        example_cpp.write_text(
            r"""// Example C++ solution: echo input (portable headers)
#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string line, all;
    while (std::getline(std::cin, line)) {
        if (!all.empty()) all.push_back('\n');
        all += line;
    }
    if (all.empty()) std::cout << "0\n";
    else std::cout << all << "\n";
    return 0;
}
""",
            encoding="utf-8",
        )

    print(f"Tasks generated at: {args.output_dir}")
    print("Files:")
    for p in sorted(args.output_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
