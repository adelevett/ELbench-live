"""Persistent leaderboard, official ELBench results, comparison engine, and run recovery store."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from elbench.dashboard.english import area_label, plain_file
from elbench.persistence.writers import OutputPaths

DEFAULT_HF_LEADERBOARD_REPO = "adelevett/elbench-live-leaderboard"
OFFICIAL_DATA_PATH = Path(__file__).with_name("official_results_data.json")

GENERAL_FILES = (
    "mmlu_pro_sampled.jsonl",
    "ceval_sampled.jsonl",
    "ifeval_sampled.jsonl",
    "math_500_sampled.jsonl",
    "aime24_sampled.jsonl",
    "aime25.jsonl",
    "aime26.jsonl",
)
SAFETY_FILES = (
    "安全拒答.jsonl",
    "安全引导.jsonl",
    "安全回答.jsonl",
    "SATAs.xlsx",
    "adversarial_prompts.xlsx",
)
BASIC_FILES = (
    "knowledge.yaml",
    "question.yaml",
    "cross.yaml",
    "config_guided_task.yaml",
)
HIGHLEVEL_FILES = (
    "高阶育人-edu.jsonl",
    "高阶育人-omni.jsonl",
)

FILE_MAX_SCORE: dict[str, float] = {
    "knowledge.yaml": 10.0,
    "question.yaml": 5.0,
    "cross.yaml": 5.0,
    "config_guided_task.yaml": 5.0,
}

FILE_TO_MODULE: dict[str, str] = {
    **{fn: "通用模型" for fn in GENERAL_FILES},
    **{fn: "安全可信" for fn in SAFETY_FILES},
    **{fn: "基本教育" for fn in BASIC_FILES},
    **{fn: "高阶育人" for fn in HIGHLEVEL_FILES},
}

_STORE_LOCK = threading.Lock()

# Files keyed to China's national rules rather than to education: the general "should refuse" set
# (212 of 250 items labelled under China's "socialist core values" content rules) and C-Eval (Chinese
# national exams, including Marxism and Mao Zedong Thought). The published results are per file, so
# whole files are the finest cut that still re-scores the paper models. See D4 / T15.
CHINA_SPECIFIC_FILES = ("安全拒答.jsonl", "ceval_sampled.jsonl")
SCOPE_CORE = "core"
SCOPE_PAPER = "paper"
MODULE_SCORE_KEYS = {
    "overall": "overall_pct",
    "general": "general_pct",
    "safety": "safety_pct",
    "basic_education": "basic_education_pct",
    "high_level": "high_level_pct",
}


# The OpenRouter model closest to the paper's own hosted Qwen judge. Visitor runs it grades are ranked
# with the published models; runs graded by any other judge get a board of their own.
PAPER_JUDGE_SLUG = "qwen/qwen3-235b-a22b-2507"
PAPER_JUDGE_LABEL = "Qwen3-235B"
SYSTEM_ONE_PREFIX = "decision:"


def judge_board(run: dict[str, Any]) -> str | None:
    """Board a visitor run is ranked on: its judge, or None when no answer needed a judge.

    Runs saved before ``judge_used`` was recorded count as judged.
    """
    if run.get("judge_used") is False:
        return None
    judge = str(run.get("judge_model") or "").strip()
    if not judge or judge == "default":
        return PAPER_JUDGE_SLUG
    return f"{SYSTEM_ONE_PREFIX}{judge}" if run.get("use_system_one_judge") else judge


def on_judge_board(run: dict[str, Any], board: str) -> bool:
    """A run without judged answers scores the same under every judge, so it belongs on every board."""
    key = judge_board(run)
    return key is None or key == board


def judge_board_label(board: str) -> str:
    if board == PAPER_JUDGE_SLUG:
        return f"{PAPER_JUDGE_LABEL} (default, ranked with the published results)"
    if board.startswith(SYSTEM_ONE_PREFIX):
        return f"{board[len(SYSTEM_ONE_PREFIX):]} (decision model)"
    return board


def judge_board_choices(runs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The default board first, then every other judge a visitor run used, with its run count."""
    counts: dict[str, int] = {}
    for run in runs:
        key = judge_board(run) if isinstance(run, dict) else None
        if key is not None and key != PAPER_JUDGE_SLUG:
            counts[key] = counts.get(key, 0) + 1
    choices = [(judge_board_label(PAPER_JUDGE_SLUG), PAPER_JUDGE_SLUG)]
    for key in sorted(counts):
        choices.append((f"{judge_board_label(key)} · {counts[key]} run{'s' if counts[key] != 1 else ''}", key))
    return choices


def scope_files(scope: str) -> tuple[str, ...]:
    """Files scored in ``scope``: every file for the paper's scope, all but the region-specific ones for core."""
    if scope == SCOPE_PAPER:
        return tuple(FILE_TO_MODULE)
    return tuple(fn for fn in FILE_TO_MODULE if fn not in CHINA_SPECIFIC_FILES)


def scores_in_scope(per_file: dict[str, dict[str, Any]], scope: str) -> dict[str, Any]:
    """Module and overall scores over the files in ``scope``, by the paper's rules."""
    keep = set(scope_files(scope))
    return module_scores_from_per_file({fn: stats for fn, stats in per_file.items() if fn in keep})


def official_scores(item: dict[str, Any], scope: str) -> dict[str, Any]:
    """A published model's scores keyed like ``MODULE_SCORE_KEYS`` (plus ``per_file``) in ``scope``.

    The paper's scope uses the published numbers as printed; core recomputes them from per-file data.
    """
    per_file = item.get("per_file") or {}
    if scope == SCOPE_PAPER:
        return {**{key: item.get(key) for key in MODULE_SCORE_KEYS}, "per_file": per_file}
    norm = scores_in_scope(per_file, scope)
    # per_file stays whole: a single-task view of a region-specific file is an explicit choice.
    return {**{key: norm.get(field) for key, field in MODULE_SCORE_KEYS.items()}, "per_file": per_file}


def scope_item_count(scope: str) -> int:
    counts = full_file_counts()
    return sum(counts.get(fn, 0) for fn in scope_files(scope))


def load_official_models() -> list[dict[str, Any]]:
    if not OFFICIAL_DATA_PATH.exists():
        return []
    payload = json.loads(OFFICIAL_DATA_PATH.read_text(encoding="utf-8"))
    return list(payload.get("official_models") or [])


def official_leaderboard_rows(scope: str = SCOPE_PAPER) -> list[list[Any]]:
    """Published models ranked in ``scope``. Only the paper's scope has a published 95% range."""
    scored = [(item, official_scores(item, scope)) for item in load_official_models()]
    scored.sort(key=lambda pair: -float(pair[1]["overall"] or 0.0))
    items = scope_item_count(scope)
    rows: list[list[Any]] = []
    for rank, (item, scores) in enumerate(scored, start=1):
        overall = f"{scores['overall']:.1f}"
        if scope == SCOPE_PAPER:
            overall += f" {item['ci_95']}"
        rows.append(
            [
                rank if scope != SCOPE_PAPER else item["rank"],
                item["display_name"],
                overall,
                round(float(scores["general"]), 1),
                round(float(scores["safety"]), 1),
                round(float(scores["basic_education"]), 1),
                round(float(scores["high_level"]), 1),
                item.get("items", 2939) if scope == SCOPE_PAPER else items,
            ]
        )
    return rows


def official_per_file_rows(model_id_filter: str | None = None, scope: str = SCOPE_PAPER) -> list[list[Any]]:
    rows: list[list[Any]] = []
    models = load_official_models()
    for item in models:
        if model_id_filter and model_id_filter != "All Official Models":
            if item["model_id"] != model_id_filter and item["display_name"] != model_id_filter:
                continue
        per_file = item.get("per_file") or {}
        in_scope = set(scope_files(scope))
        for fn, stats in per_file.items():
            if fn not in in_scope:
                continue
            module_key = FILE_TO_MODULE.get(fn, "")
            rows.append(
                [
                    item["display_name"],
                    area_label(module_key),
                    plain_file(fn),
                    stats.get("count"),
                    stats.get("normalized_pct"),
                    stats.get("pass_rate_pct"),
                    stats.get("avg_score"),
                ]
            )
    return rows


def normalize_file_entry(source_file: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Score one file on 0-100 the way the paper does.

    General and High-Level use the pass rate. Safety uses the mean score, which
    gives partial credit on SATA and teaching-harm items. Basic Education uses
    the mean rubric score over that scenario's maximum.
    """
    count = int(stats.get("count") or 0)
    avg_score = stats.get("avg_score")
    pass_rate = stats.get("pass_rate")
    avg_val = float(avg_score) if avg_score is not None else 0.0
    pass_val = float(pass_rate) if pass_rate is not None else 0.0

    if source_file in FILE_MAX_SCORE:
        norm_pct = (avg_val / FILE_MAX_SCORE[source_file]) * 100.0
    elif source_file in SAFETY_FILES or pass_rate is None:
        norm_pct = avg_val * 100.0
    else:
        norm_pct = pass_val * 100.0

    return {
        "count": count,
        "avg_score": round(avg_val, 4) if avg_score is not None else None,
        "pass_rate_pct": round(pass_val * 100.0, 2) if pass_rate is not None else None,
        "normalized_pct": round(max(0.0, min(100.0, norm_pct)), 2),
    }


def basic_education_valid_means(judged_path: Path) -> dict[str, float]:
    """Mean rubric score per Basic Education file, skipping missing or out-of-range scores.

    The runtime occasionally emits a score above the rubric maximum (one 74.07 on a
    5-point rubric in the published runs). The audited paper scores leave those out.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    if not judged_path.exists():
        return {}
    with judged_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_file = record.get("source_file")
            score = record.get("score")
            max_score = FILE_MAX_SCORE.get(str(source_file))
            if max_score is None or not isinstance(score, (int, float)):
                continue
            if score < 0 or score > max_score:
                continue
            sums[source_file] = sums.get(source_file, 0.0) + float(score)
            counts[source_file] = counts.get(source_file, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def compute_normalized_scores(
    summary: dict[str, Any],
    *,
    basic_valid_means: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute 0-100 module, overall, and per-file scores from ``build_summary``.

    Matches the paper: General, Safety, and High-Level are item-weighted over their
    files; Basic Education is the plain mean of its scenarios; Overall is the mean
    of the modules present.
    """
    by_file_raw = summary.get("by_source_file") or {}
    per_file: dict[str, dict[str, Any]] = {}
    for source_file, stats in by_file_raw.items():
        if not isinstance(stats, dict):
            continue
        if basic_valid_means and source_file in basic_valid_means:
            stats = {**stats, "avg_score": basic_valid_means[source_file]}
        per_file[source_file] = normalize_file_entry(source_file, stats)
    return module_scores_from_per_file(per_file)


def module_scores_from_per_file(per_file: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def _present(files: tuple[str, ...]) -> list[dict[str, Any]]:
        return [per_file[fn] for fn in files if fn in per_file and int(per_file[fn]["count"] or 0) > 0]

    def _item_weighted(files: tuple[str, ...]) -> float | None:
        entries = _present(files)
        total_n = sum(int(entry["count"]) for entry in entries)
        if total_n == 0:
            return None
        return round(sum(int(entry["count"]) * float(entry["normalized_pct"]) for entry in entries) / total_n, 2)

    def _scenario_mean(files: tuple[str, ...]) -> float | None:
        entries = _present(files)
        if not entries:
            return None
        return round(sum(float(entry["normalized_pct"]) for entry in entries) / len(entries), 2)

    general_pct = _item_weighted(GENERAL_FILES)
    safety_pct = _item_weighted(SAFETY_FILES)
    basic_pct = _scenario_mean(BASIC_FILES)
    highlevel_pct = _item_weighted(HIGHLEVEL_FILES)

    present_modules = [
        val
        for val in (general_pct, safety_pct, basic_pct, highlevel_pct)
        if val is not None
    ]
    overall_pct = round(sum(present_modules) / len(present_modules), 2) if present_modules else None

    return {
        "overall_pct": overall_pct,
        "general_pct": general_pct,
        "safety_pct": safety_pct,
        "basic_education_pct": basic_pct,
        "high_level_pct": highlevel_pct,
        "per_file": per_file,
    }


def compute_plan_fingerprint(
    *,
    openrouter_model: str,
    judge_openrouter_model: str | None,
    use_system_one_judge: bool,
    use_multiturn_chat_model: bool,
    multiturn_openrouter_model: str | None,
    modules: set[str] | frozenset[str],
    source_files: set[str] | frozenset[str],
    dimensions: set[str] | frozenset[str],
    max_samples: int | None,
    max_samples_per_file: int | None = None,
) -> str:
    payload = {
        "target_model": openrouter_model.strip(),
        "judge_model": (judge_openrouter_model or "").strip(),
        "system_one_judge": bool(use_system_one_judge),
        "multiturn_chat": bool(use_multiturn_chat_model),
        "multiturn_model": (multiturn_openrouter_model or "").strip() if use_multiturn_chat_model else "",
        "modules": sorted(modules),
        "source_files": sorted(source_files),
        "dimensions": sorted(dimensions),
        "max_samples": max_samples,
    }
    if max_samples_per_file is not None:
        # Only when set, so fingerprints of earlier runs (and their resume) are unchanged.
        payload["max_samples_per_file"] = max_samples_per_file
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _resolve_hf_token() -> str:
    for env_key in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_API_KEY=") or line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val
        except OSError:
            pass
    return ""


class LeaderboardStore:
    """Reliable dual-layer (local atomic JSON + Hugging Face Dataset) leaderboard & checkpoint store."""

    def __init__(
        self,
        project_root: Path,
        *,
        dataset_repo_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.local_dir = project_root / "outputs" / "leaderboard"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.local_dir / "runs.json"
        self.dataset_repo_id = (
            dataset_repo_id
            or os.environ.get("ELBENCH_LEADERBOARD_REPO", "").strip()
            or DEFAULT_HF_LEADERBOARD_REPO
        )
        self.hf_token = hf_token if hf_token is not None else _resolve_hf_token()
        self._last_sync_status: str = "Local atomic store ready"

    @property
    def sync_status(self) -> str:
        return self._last_sync_status

    def load_runs(self) -> list[dict[str, Any]]:
        with _STORE_LOCK:
            return self._read_local_unlocked()

    def sync_from_hub(self) -> list[dict[str, Any]]:
        """Pull remote ``runs.json`` from Hugging Face Dataset repo and merge with local runs."""
        if not self.hf_token or not self.dataset_repo_id:
            self._last_sync_status = "Local disk persistence active (set HF_TOKEN for Hub Dataset sync)."
            return self.load_runs()
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.dataset_repo_id,
                repo_type="dataset",
                filename="runs.json",
                token=self.hf_token,
            )
            remote_data = json.loads(Path(downloaded).read_text(encoding="utf-8"))
            remote_runs = remote_data.get("runs") if isinstance(remote_data, dict) else remote_data
            if not isinstance(remote_runs, list):
                remote_runs = []
        except Exception as exc:  # noqa: BLE001
            self._last_sync_status = f"Local store active (Hub dataset `{self.dataset_repo_id}` will be initialized on first save)."
            return self.load_runs()

        with _STORE_LOCK:
            local_runs = self._read_local_unlocked()
            by_id: dict[str, dict[str, Any]] = {}
            for entry in remote_runs:
                if isinstance(entry, dict) and entry.get("run_id"):
                    by_id[str(entry["run_id"])] = entry
            for entry in local_runs:
                if isinstance(entry, dict) and entry.get("run_id"):
                    rid = str(entry["run_id"])
                    prev = by_id.get(rid)
                    if prev is None or str(entry.get("updated_at", "")) >= str(prev.get("updated_at", "")):
                        by_id[rid] = entry
            merged = sorted(
                by_id.values(),
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )
            self._write_local_unlocked(merged)
            self._last_sync_status = (
                f"Synced {len(merged)} run(s) with Hugging Face Dataset `{self.dataset_repo_id}`."
            )
            return merged

    def find_resumable_run(
        self,
        *,
        config_hash: str | None = None,
        explicit_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the most recent incomplete or failed run matching ``explicit_run_id`` or ``config_hash``."""
        runs = self.load_runs()
        if explicit_run_id and explicit_run_id.strip():
            wanted = explicit_run_id.strip()
            for entry in runs:
                if entry.get("run_id") == wanted:
                    return entry
            return None
        if not config_hash:
            return None
        for entry in runs:
            if entry.get("config_hash") != config_hash:
                continue
            status = str(entry.get("status") or "")
            failures = int(entry.get("total_failures") or 0)
            if status in {"partial", "interrupted", "running"} or failures > 0:
                return entry
        return None

    def prepare_resume_directory(
        self,
        *,
        run_id: str,
        model_id: str,
        output_root: Path,
    ) -> bool:
        """Ensure checkpoint and JSONL files exist in ``output_root`` (pulling from HF Hub if needed)."""
        paths = OutputPaths.build(output_root, run_id, model_id)
        if paths.checkpoint_path.exists() or paths.judged_path.exists():
            return True
        if not self.hf_token or not self.dataset_repo_id:
            return False
        restored_any = False
        try:
            from huggingface_hub import hf_hub_download

            mapping = {
                f"runs/{run_id}/checkpoint.json": paths.checkpoint_path,
                f"runs/{run_id}/judged.jsonl": paths.judged_path,
                f"runs/{run_id}/failures.jsonl": paths.failures_path,
                f"runs/{run_id}/summary.json": paths.summary_path,
            }
            for remote_name, dest_path in mapping.items():
                try:
                    cached = hf_hub_download(
                        repo_id=self.dataset_repo_id,
                        repo_type="dataset",
                        filename=remote_name,
                        token=self.hf_token,
                    )
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cached, dest_path)
                    restored_any = True
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            return False
        return restored_any

    def save_run(
        self,
        record: dict[str, Any],
        *,
        output_Paths: OutputPaths | None = None,
        push_to_hub: bool = True,
    ) -> dict[str, Any]:
        """Atomically save ``record`` to local disk and optionally push to Hugging Face Dataset repo."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = dict(record)
        entry.setdefault("created_at", now_iso)
        entry["updated_at"] = now_iso

        with _STORE_LOCK:
            runs = self._read_local_unlocked()
            by_id = {str(r["run_id"]): r for r in runs if isinstance(r, dict) and r.get("run_id")}
            rid = str(entry["run_id"])
            if rid in by_id and "created_at" in by_id[rid]:
                entry["created_at"] = by_id[rid]["created_at"]
            by_id[rid] = entry
            merged = sorted(
                by_id.values(),
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )
            self._write_local_unlocked(merged)

        if push_to_hub and self.hf_token and self.dataset_repo_id:
            self._push_to_hub(entry, output_Paths)
        return entry

    def _push_to_hub(self, entry: dict[str, Any], output_paths: OutputPaths | None) -> None:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi

            api = HfApi(token=self.hf_token)
            api.create_repo(
                repo_id=self.dataset_repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            operations = [
                CommitOperationAdd(
                    path_in_repo="runs.json",
                    path_or_fileobj=str(self.ledger_path),
                )
            ]
            run_id = str(entry["run_id"])
            record_bytes = json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"runs/{run_id}/record.json",
                    path_or_fileobj=record_bytes,
                )
            )
            if output_paths is not None:
                for remote_leaf, local_path in (
                    ("summary.json", output_paths.summary_path),
                    ("checkpoint.json", output_paths.checkpoint_path),
                    ("judged.jsonl", output_paths.judged_path),
                    ("failures.jsonl", output_paths.failures_path),
                ):
                    if local_path.exists() and local_path.stat().st_size > 0:
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"runs/{run_id}/{remote_leaf}",
                                path_or_fileobj=str(local_path),
                            )
                        )
            api.create_commit(
                repo_id=self.dataset_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Update ELBench-live run {run_id} ({entry.get('status', 'completed')})",
            )
            self._last_sync_status = (
                f"Saved locally and synced to Hugging Face Dataset `{self.dataset_repo_id}`."
            )
        except Exception as exc:  # noqa: BLE001
            self._last_sync_status = (
                f"Saved to local atomic store (Hugging Face Hub sync warning: {type(exc).__name__})."
            )

    def _read_local_unlocked(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        try:
            payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            rows = payload.get("runs") if isinstance(payload, dict) else payload
            return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        except Exception:  # noqa: BLE001
            return []

    def _write_local_unlocked(self, runs: list[dict[str, Any]]) -> None:
        tmp_path = self.ledger_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps({"runs": runs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.ledger_path)


def run_coverage_label(run: dict[str, Any], counts: dict[str, int] | None = None) -> str:
    if run_covers_metric(run, "overall", counts, scope=SCOPE_PAPER):
        return "Full benchmark"
    if run_covers_metric(run, "overall", counts, scope=SCOPE_CORE):
        return "Complete, without the region-specific sets"
    return "Part of the benchmark"


def custom_leaderboard_rows(runs: list[dict[str, Any]], scope: str = SCOPE_PAPER) -> list[list[Any]]:
    """Format persisted visitor runs for the visitor-run table.

    A score is shown only for areas the run covered in full in ``scope``, so a short run never shows a
    headline number next to complete ones.
    """
    counts = full_file_counts()
    rows: list[list[Any]] = []
    for run in runs:
        per_file = (run.get("normalized_scores") or {}).get("per_file") or {}
        norm = scores_in_scope(per_file, scope)

        def _cell(metric: str) -> Any:
            if not run_covers_metric(run, metric, counts, scope=scope):
                return None
            return norm.get(MODULE_SCORE_KEYS[metric])

        judge_desc = str(run.get("judge_model") or "default")
        if run.get("use_system_one_judge"):
            judge_desc += " (System 1)"
        rows.append(
            [
                run.get("run_id"),
                run.get("updated_at") or run.get("created_at"),
                run.get("target_model"),
                judge_desc,
                run.get("status", "completed"),
                run_coverage_label(run, counts),
                run.get("total_judged", 0),
                run.get("total_failures", 0),
                _cell("overall"),
                _cell("general"),
                _cell("safety"),
                _cell("basic_education"),
                _cell("high_level"),
            ]
        )
    return rows


def build_comparison_choices(
    runs: list[dict[str, Any]], scope: str = SCOPE_PAPER
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(primary_choices, baseline_choices)`` combining custom runs and official paper models."""
    counts = full_file_counts()
    custom_items: list[tuple[str, str]] = []
    for run in runs:
        rid = str(run.get("run_id") or "")
        model = str(run.get("target_model") or "unknown")
        status = str(run.get("status") or "completed")
        judged = int(run.get("total_judged") or 0)
        overall = None
        if run_covers_metric(run, "overall", counts, scope=scope):
            per_file = (run.get("normalized_scores") or {}).get("per_file") or {}
            overall = scores_in_scope(per_file, scope).get("overall_pct")
        score_str = f"{overall:.1f}" if isinstance(overall, (int, float)) else "partial"
        label = f"Visitor run: {model} · {rid} ({judged} items, {status}, {score_str})"
        custom_items.append((label, f"run:{rid}"))

    official_items: list[tuple[str, str]] = []
    for item in load_official_models():
        overall = official_scores(item, scope)["overall"]
        range_text = f" {item['ci_95']}" if scope == SCOPE_PAPER else ""
        label = f"Published: {item['display_name']} (Overall {overall:.1f}{range_text})"
        official_items.append((label, f"official:{item['model_id']}"))

    all_choices = custom_items + official_items
    return all_choices, all_choices


def _resolve_comparison_entry(
    selector: str,
    runs_by_id: dict[str, dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    scope: str = SCOPE_PAPER,
) -> dict[str, Any] | None:
    if not selector:
        return None
    if selector.startswith("run:"):
        rid = selector.split(":", 1)[1]
        run = runs_by_id.get(rid)
        if not run:
            return None
        norm = scores_in_scope((run.get("normalized_scores") or {}).get("per_file") or {}, scope)
        return {
            "label": f"{run.get('target_model')} ({rid})",
            "kind": "Custom Run",
            "judge": judge_board(run),
            **{key: norm.get(field) for key, field in MODULE_SCORE_KEYS.items()},
            "per_file": norm.get("per_file") or {},
        }
    if selector.startswith("official:"):
        mid = selector.split(":", 1)[1]
        off = official_by_id.get(mid)
        if not off:
            return None
        return {
            "label": f"{off['display_name']} (Official)",
            "kind": "Official Paper",
            "judge": PAPER_JUDGE_SLUG,
            **official_scores(off, scope),
        }
    return None


def build_comparison_table(
    primary_selector: str | None,
    baseline_selectors: list[str] | None,
    granularity: str,
    runs: list[dict[str, Any]],
    scope: str = SCOPE_PAPER,
) -> tuple[list[str], list[list[Any]], str]:
    """Build a side-by-side comparison table (headers, rows, note) between a primary run/model and baselines."""
    runs_by_id = {str(r["run_id"]): r for r in runs if isinstance(r, dict) and r.get("run_id")}
    official_by_id = {str(m["model_id"]): m for m in load_official_models()}

    primary = _resolve_comparison_entry(primary_selector or "", runs_by_id, official_by_id, scope)
    if primary is None:
        return (
            ["Metric", "Primary"],
            [],
            "Select a Primary Run or Official Model above to compare.",
        )

    baselines: list[dict[str, Any]] = []
    for sel in baseline_selectors or []:
        if sel == primary_selector:
            continue
        resolved = _resolve_comparison_entry(sel, runs_by_id, official_by_id, scope)
        if resolved is not None:
            baselines.append(resolved)

    headers = ["Category / Benchmark Slice", "Items in Primary", f"{primary['label']} (%)"]
    for b in baselines:
        headers.append(f"{b['label']} (%)")
        headers.append(f"Δ Primary vs {b['label']}")

    rows: list[list[Any]] = []
    primary_files = primary.get("per_file") or {}

    if granularity == "By Module (0–100 Normalized)":
        module_specs = [
            ("Overall (Published / Averaged Modules)", "overall", None),
            ("General Capability", "general", GENERAL_FILES),
            ("Safety & Trustworthiness", "safety", SAFETY_FILES),
            ("Basic Education", "basic_education", BASIC_FILES),
            ("High-Level Educational Cultivation", "high_level", HIGHLEVEL_FILES),
        ]
        for label, key, file_group in module_specs:
            if file_group is None:
                count_val: Any = sum(int((v or {}).get("count") or 0) for v in primary_files.values())
            else:
                count_val = sum(int((primary_files.get(fn) or {}).get("count") or 0) for fn in file_group)
            p_score = primary.get(key)
            row: list[Any] = [
                label,
                count_val,
                round(float(p_score), 2) if isinstance(p_score, (int, float)) else "—",
            ]
            for b in baselines:
                b_score = b.get(key)
                b_cell = round(float(b_score), 2) if isinstance(b_score, (int, float)) else "—"
                if isinstance(p_score, (int, float)) and isinstance(b_score, (int, float)):
                    diff = round(float(p_score) - float(b_score), 2)
                    diff_cell = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                else:
                    diff_cell = "—"
                row.extend([b_cell, diff_cell])
            rows.append(row)
        note = (
            "Module-level comparison on a 0–100 normalized scale. When comparing a partial/subset run "
            "against a full 2,939-item run, switch to **By Source File / Task (Matched Subset)** for an "
            "apples-to-apples comparison restricted to the exact files evaluated."
        )
        return headers, rows, note + _judge_note([primary, *baselines])

    # By Source File / Task (Matched Subset)
    in_scope = scope_files(scope)
    active_files = [fn for fn in in_scope if fn in primary_files and int(primary_files[fn].get("count") or 0) > 0]
    if not active_files:
        active_files = list(in_scope)

    total_primary_items = 0
    weighted_primary_sum = 0.0
    baseline_weighted_sums = [0.0 for _ in baselines]
    baseline_weighted_counts = [0 for _ in baselines]

    for fn in active_files:
        p_entry = primary_files.get(fn) or {}
        p_count = int(p_entry.get("count") or 0)
        p_norm = p_entry.get("normalized_pct")
        if isinstance(p_norm, (int, float)) and p_count > 0:
            total_primary_items += p_count
            weighted_primary_sum += p_count * float(p_norm)

        row = [
            f"{area_label(FILE_TO_MODULE.get(fn, ''))} · {plain_file(fn)}",
            p_count,
            round(float(p_norm), 2) if isinstance(p_norm, (int, float)) else "—",
        ]
        for idx, b in enumerate(baselines):
            b_entry = (b.get("per_file") or {}).get(fn) or {}
            b_norm = b_entry.get("normalized_pct")
            if isinstance(b_norm, (int, float)) and p_count > 0:
                baseline_weighted_sums[idx] += p_count * float(b_norm)
                baseline_weighted_counts[idx] += p_count
            b_cell = round(float(b_norm), 2) if isinstance(b_norm, (int, float)) else "—"
            if isinstance(p_norm, (int, float)) and isinstance(b_norm, (int, float)):
                diff = round(float(p_norm) - float(b_norm), 2)
                diff_cell = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            else:
                diff_cell = "—"
            row.extend([b_cell, diff_cell])
        rows.append(row)

    if total_primary_items > 0:
        p_matched_avg = round(weighted_primary_sum / total_primary_items, 2)
        summary_row: list[Any] = [
            "Matched Subset Weighted Average (Primary Files Only)",
            total_primary_items,
            p_matched_avg,
        ]
        for idx, _ in enumerate(baselines):
            if baseline_weighted_counts[idx] > 0:
                b_avg = round(baseline_weighted_sums[idx] / baseline_weighted_counts[idx], 2)
                diff = round(p_matched_avg - b_avg, 2)
                diff_cell = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                summary_row.extend([b_avg, diff_cell])
            else:
                summary_row.extend(["—", "—"])
        rows.insert(0, summary_row)

    note = (
        f"Matched-subset comparison across the **{len(active_files)}** source file(s) present in "
        f"`{primary['label']}`. The top row weights each baseline's score by the primary run's "
        "evaluated item counts on those exact files."
    )
    return headers, rows, note + _judge_note([primary, *baselines])


def _judge_note(entries: list[dict[str, Any]]) -> str:
    judges = sorted({entry["judge"] for entry in entries if entry.get("judge")})
    if len(judges) < 2:
        return ""
    return (
        f" **These results were graded by different judges** ({', '.join(judge_board_label(j) for j in judges)}), "
        "so scores on open-ended tasks are not directly comparable. Automatically checked tasks are."
    )


PUBLISHED_SOURCE = "Published result"
VISITOR_SOURCE = "Visitor run"
BEST_VALUE_STATUS = "★ Best value"

PARETO_SCORE_METRICS: tuple[tuple[str, str], ...] = (
    ("Overall", "overall"),
    ("Teaching", "basic_education"),
    ("Safety", "safety"),
    ("Knowledge & reasoning", "general"),
    ("Character & thinking skills", "high_level"),
)

PARETO_COST_METRICS: tuple[tuple[str, str], ...] = (
    ("Price per million tokens (3 parts input to 1 part output)", "blended_per_1m"),
    ("Input price per million tokens", "prompt_per_1m"),
    ("Output price per million tokens", "completion_per_1m"),
)

METRIC_SCOPE_FILES: dict[str, tuple[str, ...]] = {
    "overall": tuple(FILE_TO_MODULE),
    "general": GENERAL_FILES,
    "safety": SAFETY_FILES,
    "basic_education": BASIC_FILES,
    "high_level": HIGHLEVEL_FILES,
}


def pareto_score_metric_choices() -> list[tuple[str, str]]:
    choices = list(PARETO_SCORE_METRICS)
    for fn, mod_key in FILE_TO_MODULE.items():
        choices.append((f"{area_label(mod_key)} · {plain_file(fn)}", f"file:{fn}"))
    return choices


def pareto_cost_metric_choices() -> list[tuple[str, str]]:
    return list(PARETO_COST_METRICS)


def full_file_counts() -> dict[str, int]:
    """Item count per file in a full run, taken from the published runs."""
    models = load_official_models()
    if not models:
        return {}
    return {name: int(stats.get("count") or 0) for name, stats in (models[0].get("per_file") or {}).items()}


def run_covers_metric(
    run: dict[str, Any],
    score_metric: str,
    counts: dict[str, int] | None = None,
    *,
    scope: str = SCOPE_PAPER,
) -> bool:
    """True when the run scored every item behind ``score_metric`` in ``scope``.

    A run that skipped items (a sample cap, a dimension filter, failures, or
    unselected files) has a score over a different item set than the published
    models, so it is not plotted against them.
    """
    if score_metric.startswith("file:"):
        files: tuple[str, ...] = (score_metric.split(":", 1)[1],)
    else:
        in_scope = set(scope_files(scope))
        files = tuple(
            fn for fn in METRIC_SCOPE_FILES.get(score_metric, tuple(FILE_TO_MODULE)) if fn in in_scope
        )
    full = counts if counts is not None else full_file_counts()
    per_file = (run.get("normalized_scores") or {}).get("per_file") or {}
    for name in files:
        expected = full.get(name)
        judged = int((per_file.get(name) or {}).get("count") or 0)
        if not expected or judged < expected:
            return False
    return True


def seed_official_openrouter_prices() -> dict[str, tuple[float | None, float | None]]:
    """Return baseline OpenRouter token prices (USD/token) for official paper models."""
    seeded: dict[str, tuple[float | None, float | None]] = {}
    for item in load_official_models():
        slug = item.get("openrouter_slug")
        p_in = item.get("prompt_usd_per_token")
        p_out = item.get("completion_usd_per_token")
        if slug and isinstance(p_in, (int, float)) and isinstance(p_out, (int, float)):
            seeded[str(slug)] = (float(p_in), float(p_out))
    return seeded


def _compute_cost_metric_value(
    prompt_usd_per_token: float | None,
    completion_usd_per_token: float | None,
    cost_metric: str,
) -> float | None:
    if prompt_usd_per_token is None or completion_usd_per_token is None:
        return None
    if prompt_usd_per_token < 0 or completion_usd_per_token < 0:
        return None
    p_in_m = float(prompt_usd_per_token) * 1_000_000.0
    p_out_m = float(completion_usd_per_token) * 1_000_000.0
    if cost_metric == "prompt_per_1m":
        return round(p_in_m, 4)
    if cost_metric == "completion_per_1m":
        return round(p_out_m, 4)
    # Default: blended_per_1m (3:1 input-to-output token blend)
    return round((3.0 * p_in_m + 1.0 * p_out_m) / 4.0, 4)


def official_ci(entry: dict[str, Any], score_metric: str) -> tuple[float, float] | None:
    """Published 95% bootstrap interval for ``score_metric``, when the paper reports one."""
    if score_metric == "overall":
        raw = entry.get("overall_ci")
    else:
        raw = (entry.get("module_ci") or {}).get(score_metric)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return None


def statistically_tied_with_leader(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Models whose 95% interval overlaps the top scorer's interval, top scorer first."""
    with_ci = [c for c in candidates if c.get("ci") and c.get("score") is not None]
    if len(with_ci) < 2:
        return []
    leader = max(with_ci, key=lambda c: float(c["score"]))
    low = leader["ci"][0]
    tied = [c for c in with_ci if c is not leader and c["ci"][1] >= low]
    tied.sort(key=lambda c: -float(c["score"]))
    return [leader, *tied] if tied else []


def _extract_pareto_score(entry_data: dict[str, Any], score_metric: str) -> float | None:
    if score_metric.startswith("file:"):
        fn = score_metric.split(":", 1)[1]
        per_file = entry_data.get("per_file") or {}
        file_stats = per_file.get(fn) or {}
        val = file_stats.get("normalized_pct")
        return float(val) if isinstance(val, (int, float)) else None
    val = entry_data.get(score_metric)
    return float(val) if isinstance(val, (int, float)) else None


def build_pareto_frontier(
    runs: list[dict[str, Any]],
    prices: dict[str, tuple[float | None, float | None]] | None = None,
    *,
    score_metric: str = "overall",
    cost_metric: str = "blended_per_1m",
    include_custom_runs: bool = True,
    keep: Callable[[dict[str, Any]], bool] | None = None,
    scope: str = SCOPE_PAPER,
    judge: str = PAPER_JUDGE_SLUG,
) -> tuple[Any, list[list[Any]], str]:
    """Compute the cost-vs-score Pareto frontier and return ``(plotly_figure, table_rows, summary_md)``."""
    from elbench.dashboard.protocol import catalog_slug

    merged_prices = {**seed_official_openrouter_prices(), **(prices or {})}

    def _lookup_price(slug: str | None, fallback_in: Any, fallback_out: Any) -> tuple[float | None, float | None]:
        if slug:
            clean = str(slug).strip()
            raw = merged_prices.get(clean) or merged_prices.get(catalog_slug(clean))
            if raw and raw[0] is not None and raw[1] is not None:
                return float(raw[0]), float(raw[1])
        if isinstance(fallback_in, (int, float)) and isinstance(fallback_out, (int, float)):
            return float(fallback_in), float(fallback_out)
        return None, None

    candidates: list[dict[str, Any]] = []
    # The published results were graded by the paper's judge, so they sit on that board only.
    for off in load_official_models() if judge == PAPER_JUDGE_SLUG else []:
        slug = off.get("openrouter_slug")
        p_in, p_out = _lookup_price(slug, off.get("prompt_usd_per_token"), off.get("completion_usd_per_token"))
        score_val = _extract_pareto_score(official_scores(off, scope), score_metric)
        cost_val = _compute_cost_metric_value(p_in, p_out, cost_metric)
        candidates.append(
            {
                "name": str(off["display_name"]),
                "source": PUBLISHED_SOURCE,
                "slug": str(slug) if slug else "Not on OpenRouter",
                "prompt_per_1m": round(p_in * 1_000_000.0, 3) if p_in is not None else None,
                "completion_per_1m": round(p_out * 1_000_000.0, 3) if p_out is not None else None,
                "cost": cost_val,
                "score": round(score_val, 2) if score_val is not None else None,
                # The bootstrap ranges were published for the paper's scope only.
                "ci": official_ci(off, score_metric) if scope == SCOPE_PAPER else None,
            }
        )

    partial_runs = 0
    if include_custom_runs:
        counts = full_file_counts()
        for run in runs:
            if not isinstance(run, dict) or not on_judge_board(run, judge):
                continue
            if not run_covers_metric(run, score_metric, counts, scope=scope):
                partial_runs += 1
                continue
            target_slug = str(run.get("target_model") or "").strip()
            rid = str(run.get("run_id") or "")
            per_file = (run.get("normalized_scores") or {}).get("per_file") or {}
            norm = scores_in_scope(per_file, scope)
            mapped = {**{key: norm.get(field) for key, field in MODULE_SCORE_KEYS.items()}, "per_file": per_file}
            score_val = _extract_pareto_score(mapped, score_metric)
            if score_val is None:
                continue
            p_in, p_out = _lookup_price(
                target_slug,
                run.get("prompt_usd_per_token"),
                run.get("completion_usd_per_token"),
            )
            cost_val = _compute_cost_metric_value(p_in, p_out, cost_metric)
            candidates.append(
                {
                    "name": f"{target_slug} ({rid[:12]})",
                    "source": VISITOR_SOURCE,
                    "slug": target_slug or "unknown",
                    "prompt_per_1m": round(p_in * 1_000_000.0, 3) if p_in is not None else None,
                    "completion_per_1m": round(p_out * 1_000_000.0, 3) if p_out is not None else None,
                    "cost": cost_val,
                    "score": round(score_val, 2),
                    "ci": None,
                }
            )

    # The visitor's price and size filters; a hidden model also leaves the best-value line.
    filtered_out = 0
    if keep is not None:
        shown = []
        for candidate in candidates:
            if candidate["cost"] is not None and candidate["score"] is not None and not keep(candidate):
                filtered_out += 1
            else:
                shown.append(candidate)
        candidates = shown

    priced = [c for c in candidates if c["cost"] is not None and c["score"] is not None]
    unpriced = [c for c in candidates if c["cost"] is None or c["score"] is None]

    # Sort by ascending cost, then descending score to identify the Pareto frontier
    priced.sort(key=lambda item: (float(item["cost"]), -float(item["score"])))
    frontier: list[dict[str, Any]] = []
    best_score = -1.0
    last_frontier_cost: float | None = None
    for item in priced:
        c_val = float(item["cost"])
        s_val = float(item["score"])
        if s_val > best_score or (last_frontier_cost is not None and c_val == last_frontier_cost and s_val == best_score):
            item["pareto"] = True
            frontier.append(item)
            best_score = s_val
            last_frontier_cost = c_val
        else:
            item["pareto"] = False

    for item in unpriced:
        item["pareto"] = None

    score_label = {key: label for label, key in pareto_score_metric_choices()}.get(score_metric, score_metric)
    cost_label = {key: label for label, key in pareto_cost_metric_choices()}.get(cost_metric, cost_metric)

    fig = _render_pareto_figure(
        priced=priced,
        frontier=frontier,
        score_label=score_label,
        cost_label=cost_label,
    )

    # Order table rows: Pareto Optimal first (by cost asc), then Dominated (by score desc), then Unpriced
    dominated = [c for c in priced if not c.get("pareto")]
    dominated.sort(key=lambda item: (-float(item["score"]), float(item["cost"])))
    ordered = frontier + dominated + unpriced

    table_rows: list[list[Any]] = []
    for item in ordered:
        if item["pareto"] is True:
            status_cell = BEST_VALUE_STATUS
        elif item["pareto"] is False:
            status_cell = "A cheaper model scores as well or better"
        else:
            status_cell = "Not on OpenRouter"
        table_rows.append(
            [
                status_cell,
                item["name"],
                item["source"],
                item["slug"],
                f"${item['prompt_per_1m']:.3f}" if item["prompt_per_1m"] is not None else "—",
                f"${item['completion_per_1m']:.3f}" if item["completion_per_1m"] is not None else "—",
                f"${item['cost']:.3f}" if item["cost"] is not None else "—",
                item["score"] if item["score"] is not None else "—",
            ]
        )

    frontier_names = [f"**{item['name']}** ({item['score']:.1f} at ${item['cost']:.2f})" for item in frontier]
    unpriced_names = [item["name"] for item in unpriced]
    summary_lines = [] if judge == PAPER_JUDGE_SLUG else [
        f"**Graded by {judge_board_label(judge)}.** Only visitor runs with this judge are shown. The published "
        "results were graded by the paper's judge, so they are not on this board."
    ]
    if judge != PAPER_JUDGE_SLUG and not priced:
        summary_lines.append("No complete runs with this judge yet. A complete run graded by it starts this board.")
    else:
        summary_lines.append(
            "**Best value:** "
            + (" → ".join(frontier_names) if frontier_names else "none with a price yet")
            + ". Each of these scores higher than every model that costs less."
        )
    tied = statistically_tied_with_leader(candidates)
    if tied:
        summary_lines.append(
            f"**Statistically tied at the top:** {', '.join(item['name'] for item in tied)}. "
            "Their 95% ranges overlap, so the order among them could change on a re-run."
        )
    if partial_runs:
        summary_lines.append(
            f"*{partial_runs} visitor run{'s are' if partial_runs != 1 else ' is'} not shown because "
            f"{'they do' if partial_runs != 1 else 'it does'} not cover every item behind this score.*"
        )
    if filtered_out:
        summary_lines.append(
            f"*{filtered_out} model{'s' if filtered_out != 1 else ''} hidden by your price or size filter.*"
        )
    if unpriced_names:
        summary_lines.append(
            f"*Not on OpenRouter, so not plotted: {', '.join(unpriced_names)}.*"
        )
    return fig, table_rows, "\n\n".join(summary_lines)


# Chart tokens (dataviz reference palette, light surface). The chart sits on its own
# light card so it reads the same under Gradio's light and dark themes.
CHART_SURFACE = "#fcfcfb"
CHART_TEXT = "#0b0b0b"
CHART_TEXT_SECONDARY = "#52514e"
CHART_GRID = "#e4e3df"
SERIES_STYLE = {
    PUBLISHED_SOURCE: {"color": "#2a78d6", "whisker": "rgba(42, 120, 214, 0.45)", "symbol": "circle"},
    VISITOR_SOURCE: {"color": "#eb6834", "whisker": "rgba(235, 104, 52, 0.45)", "symbol": "diamond"},
}
# Rough plot-area size and text metrics, used only to keep point labels from overlapping.
LABEL_PLOT_PX = (900.0, 360.0)
LABEL_CHAR_PX = 7.0
LABEL_HEIGHT_PX = 16.0
LABEL_OFFSET_PX = 8.0


def _price_text(value: float) -> str:
    return f"${value:.2f}" if value >= 0.1 else f"${value:.3f}"


def _point_label(item: dict[str, Any]) -> str:
    return item["slug"] if item["source"] == VISITOR_SOURCE else item["name"]


def _place_labels(
    priced: list[dict[str, Any]], x_range: tuple[float, float], y_range: tuple[float, float], use_log: bool
) -> dict[int, str | None]:
    """Pick a label position per point so labels don't overlap; ``None`` leaves the point unlabelled.

    Best-value points are placed first and prefer sitting above their dot; the rest prefer
    below, away from the best-value line, which always runs above them.
    """
    width, height = LABEL_PLOT_PX

    def to_px(x: float, y: float) -> tuple[float, float]:
        if use_log:
            fx = (math.log10(x) - math.log10(x_range[0])) / (math.log10(x_range[1]) - math.log10(x_range[0]))
        else:
            fx = (x - x_range[0]) / ((x_range[1] - x_range[0]) or 1.0)
        fy = (y - y_range[0]) / ((y_range[1] - y_range[0]) or 1.0)
        return fx * width, fy * height

    # Every dot and its 95% whisker is an obstacle from the start, except to its own label.
    placed: list[tuple[int, tuple[float, float, float, float]]] = []
    for item in priced:
        px, py = to_px(float(item["cost"]), float(item["score"]))
        placed.append((id(item), (px - 7, py - 7, px + 7, py + 7)))
        if item.get("ci"):
            _, low = to_px(float(item["cost"]), item["ci"][0])
            _, high = to_px(float(item["cost"]), item["ci"][1])
            placed.append((id(item), (px - 2, low, px + 2, high)))
    # The best-value line: a horizontal run from each best-value point to the next, then up.
    steps = sorted((item for item in priced if item.get("pareto")), key=lambda item: float(item["cost"]))
    for current, following in zip(steps, steps[1:] + [None]):
        x0, y0 = to_px(float(current["cost"]), float(current["score"]))
        x1 = to_px(float(following["cost"]), 0.0)[0] if following else width
        placed.append((-1, (x0, y0 - 2, x1, y0 + 2)))
        if following:
            placed.append((-1, (x1 - 2, y0, x1 + 2, to_px(1.0, float(following["score"]))[1])))
    positions: dict[int, str | None] = {}
    order = sorted(priced, key=lambda item: (not item.get("pareto"), float(item["cost"])))
    for item in order:
        px, py = to_px(float(item["cost"]), float(item["score"]))
        w = LABEL_CHAR_PX * len(_point_label(item))
        boxes = {
            "top right": (px + LABEL_OFFSET_PX - 4, py + 4, px + LABEL_OFFSET_PX - 4 + w, py + 4 + LABEL_HEIGHT_PX),
            "top left": (px - LABEL_OFFSET_PX + 4 - w, py + 4, px - LABEL_OFFSET_PX + 4, py + 4 + LABEL_HEIGHT_PX),
            "bottom right": (px + LABEL_OFFSET_PX - 4, py - 4 - LABEL_HEIGHT_PX, px + LABEL_OFFSET_PX - 4 + w, py - 4),
            "bottom left": (px - LABEL_OFFSET_PX + 4 - w, py - 4 - LABEL_HEIGHT_PX, px - LABEL_OFFSET_PX + 4, py - 4),
        }
        preference = (
            ("top right", "top left", "bottom right", "bottom left")
            if item.get("pareto")
            else ("bottom right", "bottom left", "top right", "top left")
        )
        positions[id(item)] = None
        for name in preference:
            x0, y0, x1, y1 = boxes[name]
            if x0 < 0 or x1 > width or y0 < 0 or y1 > height:
                continue
            if any(
                owner != id(item) and x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1
                for owner, (bx0, by0, bx1, by1) in placed
            ):
                continue
            placed.append((-1, boxes[name]))
            positions[id(item)] = name
            break
    return positions


def _log_ticks(low: float, high: float) -> list[float]:
    """1-2-5 ticks across ``low``..``high`` so a log price axis reads in plain dollars."""
    ticks = []
    exponent = math.floor(math.log10(low))
    while 10**exponent <= high:
        for step in (1, 2, 5):
            value = step * 10**exponent
            if low <= value <= high:
                ticks.append(value)
        exponent += 1
    return ticks


def _render_pareto_figure(
    *,
    priced: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    score_label: str,
    cost_label: str,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font={"color": CHART_TEXT_SECONDARY, "size": 13},
        margin={"l": 64, "r": 24, "t": 64, "b": 56},
        height=480,
    )
    if not priced:
        fig.add_annotation(
            text="No model has both a price and a score for this view yet.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font={"color": CHART_TEXT_SECONDARY, "size": 14},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    costs = [float(item["cost"]) for item in priced]
    scores = [float(item["score"]) for item in priced]
    use_log = min(costs) > 0 and max(costs) / min(costs) > 3.0
    if use_log:
        x_min, x_max = min(costs) / 1.8, max(costs) * 2.5
    else:
        pad = 0.15 * ((max(costs) - min(costs)) or 1.0)
        x_min, x_max = max(0.0, min(costs) - pad), max(costs) + 2 * pad
    lowest = min(p["ci"][0] if p.get("ci") else float(p["score"]) for p in priced)
    y_min = max(0.0, lowest - 3.0)
    y_max = min(102.0, max(scores) + 5.0)
    label_at = _place_labels(priced, (x_min, x_max), (y_min, y_max), use_log)

    if frontier:
        fig.add_trace(
            go.Scatter(
                x=[float(item["cost"]) for item in frontier] + [x_max],
                y=[float(item["score"]) for item in frontier] + [float(frontier[-1]["score"])],
                mode="lines",
                line={"shape": "hv", "color": CHART_TEXT_SECONDARY, "width": 2, "dash": "dot"},
                name="Best value line",
                hoverinfo="skip",
            )
        )

    for source, style in SERIES_STYLE.items():
        points = [item for item in priced if item["source"] == source]
        if not points:
            continue
        ci_low = [float(p["score"]) - p["ci"][0] if p.get("ci") else 0.0 for p in points]
        ci_high = [p["ci"][1] - float(p["score"]) if p.get("ci") else 0.0 for p in points]
        detail = []
        for p in points:
            ci = f" (95% range {p['ci'][0]:.1f} to {p['ci'][1]:.1f})" if p.get("ci") else ""
            verdict = "Best value" if p.get("pareto") else "A cheaper model scores as well or better"
            detail.append([p["name"], ci, _price_text(float(p["cost"])), verdict])
        fig.add_trace(
            go.Scatter(
                x=[float(p["cost"]) for p in points],
                y=[float(p["score"]) for p in points],
                mode="markers+text",
                name=source,
                text=[_point_label(p) if label_at[id(p)] else "" for p in points],
                textposition=[label_at[id(p)] or "top right" for p in points],
                textfont={
                    "color": [CHART_TEXT if p.get("pareto") else CHART_TEXT_SECONDARY for p in points],
                    "size": 12,
                },
                marker={
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "size": [15 if p.get("pareto") else 11 for p in points],
                    "line": {"color": CHART_SURFACE, "width": 2},
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": ci_high,
                    "arrayminus": ci_low,
                    "color": style["whisker"],
                    "thickness": 1.5,
                    "width": 0,
                },
                customdata=detail,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Score %{y:.1f}%{customdata[1]}"
                    "<br>Price %{customdata[2]}<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": f"ELBench {score_label}: score vs. price", "font": {"color": CHART_TEXT, "size": 17}, "x": 0.01},
        hovermode="closest",
        hoverdistance=40,
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": CHART_GRID, "font": {"color": CHART_TEXT}},
        legend={"orientation": "h", "x": 0.0, "y": 1.02, "yanchor": "bottom", "font": {"color": CHART_TEXT_SECONDARY}},
    )
    axis_style = {
        "gridcolor": CHART_GRID,
        "zeroline": False,
        "linecolor": CHART_GRID,
        "tickfont": {"color": CHART_TEXT_SECONDARY},
        "title_font": {"color": CHART_TEXT_SECONDARY},
    }
    if use_log:
        ticks = _log_ticks(x_min, x_max)
        fig.update_xaxes(
            type="log",
            range=[math.log10(x_min), math.log10(x_max)],
            tickvals=ticks,
            ticktext=[_price_text(t).replace(".00", "") for t in ticks],
        )
    else:
        fig.update_xaxes(range=[x_min, x_max], tickprefix="$")
    fig.update_xaxes(title_text=cost_label + (" (log scale)" if use_log else ""), **axis_style)
    fig.update_yaxes(title_text="Score (0 to 100)", range=[y_min, y_max], **axis_style)
    return fig
