from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "stage-report-20260525"


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    run_id: str
    summary_relpath: str
    scope: str
    included_modules: tuple[str, ...]
    excluded_modules: tuple[str, ...]
    notes: tuple[str, ...] = ()


RUN_SPECS = (
    RunSpec(
        model_id="gpt-5.4",
        run_id="full-gpt-5.4-20260519-v2",
        summary_relpath="outputs/summaries/full-gpt-5.4-20260519-v2/gpt-5.4.summary.json",
        scope="full_scope",
        included_modules=("general", "safety", "highlevel", "basic_education"),
        excluded_modules=(),
        notes=("cleanest currently available full-scope result",),
    ),
    RunSpec(
        model_id="deepseek-v3.2",
        run_id="fair-deepseek-v3.2-nosafety-20260520-v1",
        summary_relpath="outputs/summaries/fair-deepseek-v3.2-nosafety-20260520-v1/deepseek-v3.2.summary.json",
        scope="relay_no_safety",
        included_modules=("general", "highlevel", "basic_education"),
        excluded_modules=("safety",),
        notes=("no-safety relay run; final cleanup still recommended.",),
    ),
)


GENERAL_TASKS = (
    "mmlu_pro",
    "ceval",
    "ifeval",
    "math_500",
    "aime24",
    "aime25",
    "aime26",
    "gsm8k",
)
HIGHLEVEL_TASKS = ("highlevel_edu", "highlevel_omni")
SAFETY_TASKS = (
    "safety_refusal",
    "safety_guidance",
    "safety_answer",
    "teaching_harm",
    "adversarial_safety",
)
BASIC_TASKS = (
    "basic_knowledge_point_explanation",
    "basic_contextualized_question_generation",
    "basic_interdisciplinary_lesson_plan_generation",
    "basic_guided_problem_solving_teaching",
)
BASIC_TASK_MAX_SCORE = {
    "basic_knowledge_point_explanation": 10.0,
    "basic_contextualized_question_generation": 5.0,
    "basic_interdisciplinary_lesson_plan_generation": 5.0,
    # Guided task should use pass_rate, not avg_score.
    "basic_guided_problem_solving_teaching": 1.0,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def format_num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def weighted_pass_rate(summary: dict[str, Any], task_names: tuple[str, ...]) -> float | None:
    by_task = summary.get("by_task", {})
    total_count = 0
    weighted_sum = 0.0
    for task in task_names:
        task_data = by_task.get(task)
        if not task_data:
            continue
        count = int(task_data.get("count", 0) or 0)
        pass_rate = task_data.get("pass_rate")
        if count <= 0 or pass_rate is None:
            continue
        total_count += count
        weighted_sum += count * float(pass_rate)
    if total_count == 0:
        return None
    return weighted_sum / total_count


def weighted_basic_score(summary: dict[str, Any]) -> float | None:
    by_task = summary.get("by_task", {})
    total_count = 0
    weighted_sum = 0.0
    for task in BASIC_TASKS:
        task_data = by_task.get(task)
        if not task_data:
            continue
        count = int(task_data.get("count", 0) or 0)
        if count <= 0:
            continue
        if task == "basic_guided_problem_solving_teaching":
            normalized = float(task_data.get("pass_rate", 0.0) or 0.0)
        else:
            max_score = BASIC_TASK_MAX_SCORE[task]
            normalized = float(task_data.get("avg_score", 0.0) or 0.0) / max_score
        total_count += count
        weighted_sum += count * normalized
    if total_count == 0:
        return None
    return weighted_sum / total_count


def scope_score(row: dict[str, Any]) -> float | None:
    parts: list[float] = []
    if row["general_score"] is not None:
        parts.append(row["general_score"])
    if row["highlevel_score"] is not None:
        parts.append(row["highlevel_score"])
    if row["basic_score"] is not None:
        parts.append(row["basic_score"])
    if row["scope"] == "full_scope" and row["safety_score"] is not None:
        parts.append(row["safety_score"])
    if not parts:
        return None
    return sum(parts) / len(parts)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        summary_path = ROOT / spec.summary_relpath
        summary = load_json(summary_path)
        row = {
            "model_id": spec.model_id,
            "run_id": spec.run_id,
            "scope": spec.scope,
            "summary_path": spec.summary_relpath.replace("\\", "/"),
            "included_modules": ", ".join(spec.included_modules),
            "excluded_modules": ", ".join(spec.excluded_modules),
            "notes": " | ".join(spec.notes),
            "total_judged": int(summary.get("total_judged", 0) or 0),
            "total_failures": int(summary.get("total_failures", 0) or 0),
            "general_score": weighted_pass_rate(summary, GENERAL_TASKS),
            "highlevel_score": weighted_pass_rate(summary, HIGHLEVEL_TASKS),
            "safety_score": weighted_pass_rate(summary, SAFETY_TASKS),
            "basic_score": weighted_basic_score(summary),
        }
        row["stage_score"] = scope_score(row)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_overall_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        table.append(
            {
                "model_id": row["model_id"],
                "scope": row["scope"],
                "run_id": row["run_id"],
                "total_judged": row["total_judged"],
                "total_failures": row["total_failures"],
                "general_score_pct": format_pct(row["general_score"]),
                "highlevel_score_pct": format_pct(row["highlevel_score"]),
                "safety_score_pct": format_pct(row["safety_score"]),
                "basic_score_pct": format_pct(row["basic_score"]),
                "stage_score_pct": format_pct(row["stage_score"]),
                "included_modules": row["included_modules"],
                "excluded_modules": row["excluded_modules"],
                "notes": row["notes"],
            }
        )
    return table


def make_scope_leaderboard(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["scope"] == scope]
    filtered.sort(key=lambda item: (item["stage_score"] is None, -(item["stage_score"] or 0.0), item["model_id"]))
    leaderboard = []
    for index, row in enumerate(filtered, start=1):
        leaderboard.append(
            {
                "rank": index,
                "model_id": row["model_id"],
                "run_id": row["run_id"],
                "total_judged": row["total_judged"],
                "total_failures": row["total_failures"],
                "general_score_pct": format_pct(row["general_score"]),
                "highlevel_score_pct": format_pct(row["highlevel_score"]),
                "safety_score_pct": format_pct(row["safety_score"]),
                "basic_score_pct": format_pct(row["basic_score"]),
                "stage_score_pct": format_pct(row["stage_score"]),
                "notes": row["notes"],
            }
        )
    return leaderboard


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, divider, *body])


def write_report(rows: list[dict[str, Any]]) -> None:
    overall = make_overall_table(rows)
    full_scope = make_scope_leaderboard(rows, "full_scope")
    relay_scope = make_scope_leaderboard(rows, "relay_no_safety")
    report_path = OUTPUT_DIR / "README.md"

    lines = [
        "# Stage Report 2026-05-25",
        "",
        "## Scope",
        "",
        "- This stage report includes only currently usable summaries.",
        "- `gpt-5.4` is a full-scope result.",
        "- `deepseek-v3.2` is a `no-safety` relay result and should only be compared within the same scope.",
        "- Dirty, incomplete, expensive, or incompatible runs such as `gemini-3-flash-preview`, `kimi-k2.6`, `deepseek-r1-250528`, `doubao-seed-2-0-pro-260215`, and `gpt-5.2-pro` are excluded from the leaderboard.",
        "",
        "## Included Runs",
        "",
        markdown_table(
            overall,
            [
                ("model_id", "Model"),
                ("scope", "Scope"),
                ("total_judged", "Judged"),
                ("total_failures", "Failures"),
                ("general_score_pct", "General %"),
                ("highlevel_score_pct", "Highlevel %"),
                ("safety_score_pct", "Safety %"),
                ("basic_score_pct", "Basic %"),
                ("stage_score_pct", "Stage %"),
            ],
        ),
        "",
        "## Leaderboard: Full Scope",
        "",
        markdown_table(
            full_scope,
            [
                ("rank", "Rank"),
                ("model_id", "Model"),
                ("stage_score_pct", "Stage %"),
                ("general_score_pct", "General %"),
                ("safety_score_pct", "Safety %"),
                ("highlevel_score_pct", "Highlevel %"),
                ("basic_score_pct", "Basic %"),
            ],
        ),
        "",
        "## Leaderboard: Relay No-Safety Scope",
        "",
        markdown_table(
            relay_scope,
            [
                ("rank", "Rank"),
                ("model_id", "Model"),
                ("stage_score_pct", "Stage %"),
                ("general_score_pct", "General %"),
                ("highlevel_score_pct", "Highlevel %"),
                ("basic_score_pct", "Basic %"),
                ("total_failures", "Failures"),
            ],
        ),
        "",
        "## Notes",
        "",
        "- `Stage %` is a normalized stage score built from currently available module-level metrics.",
        "- `Basic %` normalizes basic-education tasks onto a 0-100 scale before aggregation.",
        "- Cross-scope comparison is intentionally avoided: full-scope and no-safety relay runs have different included modules.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    overall_rows = make_overall_table(rows)
    full_scope_rows = make_scope_leaderboard(rows, "full_scope")
    relay_scope_rows = make_scope_leaderboard(rows, "relay_no_safety")

    write_csv(
        OUTPUT_DIR / "summary_table.csv",
        overall_rows,
        [
            "model_id",
            "scope",
            "run_id",
            "total_judged",
            "total_failures",
            "general_score_pct",
            "highlevel_score_pct",
            "safety_score_pct",
            "basic_score_pct",
            "stage_score_pct",
            "included_modules",
            "excluded_modules",
            "notes",
        ],
    )
    write_csv(
        OUTPUT_DIR / "leaderboard_full_scope.csv",
        full_scope_rows,
        [
            "rank",
            "model_id",
            "run_id",
            "total_judged",
            "total_failures",
            "general_score_pct",
            "highlevel_score_pct",
            "safety_score_pct",
            "basic_score_pct",
            "stage_score_pct",
            "notes",
        ],
    )
    write_csv(
        OUTPUT_DIR / "leaderboard_relay_no_safety.csv",
        relay_scope_rows,
        [
            "rank",
            "model_id",
            "run_id",
            "total_judged",
            "total_failures",
            "general_score_pct",
            "highlevel_score_pct",
            "basic_score_pct",
            "stage_score_pct",
            "notes",
        ],
    )

    manifest = {
        "generated_at": "2026-05-25",
        "result_id": "stage-report-20260525",
        "included_runs": [
            {
                "model_id": row["model_id"],
                "run_id": row["run_id"],
                "scope": row["scope"],
                "summary_path": row["summary_path"],
                "notes": row["notes"],
            }
            for row in rows
        ],
        "excluded_runs": [
            {
                "model_id": "kimi-k2.6",
                "reason": "dirty runs; missing raw/judged artifacts",
            },
            {
                "model_id": "deepseek-r1-250528",
                "reason": "dirty runs; timeouts and broken judge endpoint during probe",
            },
            {
                "model_id": "doubao-seed-2-0-pro-260215",
                "reason": "run interrupted by checkpoint write failure",
            },
            {
                "model_id": "gemini-3-flash-preview",
                "reason": "excluded from official scope because the relay repeatedly returns empty content after consuming reasoning tokens",
            },
            {
                "model_id": "gpt-5.2-pro",
                "reason": "expensive GPT pro model intentionally excluded from the current campaign",
            },
        ],
        "generated_files": [
            "README.md",
            "summary_table.csv",
            "leaderboard_full_scope.csv",
            "leaderboard_relay_no_safety.csv",
            "manifest.json",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rows)


if __name__ == "__main__":
    main()
