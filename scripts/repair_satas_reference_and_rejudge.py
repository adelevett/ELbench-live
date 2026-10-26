from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elbench.config import load_project_config
from elbench.judges.judge_teaching_harm import TeachingHarmJudge
from elbench.loaders import LoaderFactory
from elbench.persistence import OutputPaths
from elbench.registry import FileRegistry
from elbench.schemas.evaluation import ModelResponse, Sample
from elbench.summary import build_summary


MODELS_AND_RUNS: dict[str, str] = {
    "claude-opus-4-8": "campaign-official-claude-opus-4-8",
    "deepseek-v4-pro": "campaign-official-deepseek-v4-pro",
    "gemini-3.5-flash": "campaign-official-gemini-3.5-flash-aime65536-highlevel-llmjudge",
    "deepseek-v4-flash": "campaign-official-deepseek-v4-flash",
    "doubao-seed-2.0-pro": "campaign-official-doubao-seed-2.0-pro",
    "gpt-5.4": "campaign-official-gpt-5.4-aime65536-highlevel-llmjudge-safety-strict-llmjudge",
    "safe-innospark": "campaign-official-safe-innospark-aime65536-highlevel-llmjudge",
    "innospark-235b": (
        "campaign-official-innospark-235b-aime65536-highlevel-llmjudge-"
        "safety-fill-targeted-safety-strict-llmjudge"
    ),
    "glm-5.1": "campaign-official-glm-5.1",
}

REQUIRED_COLUMNS = ["ID", "Question_Chinese", "Question_English", "Answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the ELBench SATAs sampled file with corrected references and "
            "rejudge existing SATAs model outputs."
        )
    )
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--corrected-path", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config_root)
    current_path = _resolve_current_satas_path(config)
    corrected_df = _read_first_sheet(args.corrected_path)
    current_df = _read_first_sheet(current_path)
    sampled_df = _build_corrected_sample(current_df=current_df, corrected_df=corrected_df)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "current_path": str(current_path),
                    "corrected_path": str(args.corrected_path),
                    "current_rows": len(current_df),
                    "corrected_rows": len(corrected_df),
                    "sampled_rows": len(sampled_df),
                    "changed_answers": int(
                        (
                            current_df["Answer"].fillna("").astype(str).to_numpy()
                            != sampled_df["Answer"].fillna("").astype(str).to_numpy()
                        ).sum()
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    _replace_satas_file(current_path=current_path, sampled_df=sampled_df)
    sample_by_id = _load_corrected_satas_samples(config)
    results = {}
    for model_id, run_id in MODELS_AND_RUNS.items():
        results[model_id] = asyncio.run(
            _rejudge_model_satas(
                output_root=config.app.output_root,
                model_id=model_id,
                run_id=run_id,
                sample_by_id=sample_by_id,
            )
        )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "current_path": str(current_path),
        "corrected_path": str(args.corrected_path),
        "sampled_rows": len(sampled_df),
        "changed_answers": int(
            (
                current_df["Answer"].fillna("").astype(str).to_numpy()
                != sampled_df["Answer"].fillna("").astype(str).to_numpy()
            ).sum()
        ),
        "models": results,
    }
    audit_dir = ROOT / "results" / "audit-judge-integrity"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_dir / "satas_reference_rejudge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _resolve_current_satas_path(config) -> Path:
    registry = FileRegistry(config)
    items = registry.resolve(source_files={"SATAs.xlsx"})
    if len(items) != 1:
        raise RuntimeError(f"Expected one SATAs registry item, got {len(items)}")
    return items[0].path


def _read_first_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.loc[:, REQUIRED_COLUMNS].copy()
    if df[REQUIRED_COLUMNS].isna().any(axis=1).any():
        bad_count = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(f"{path} has {bad_count} rows with empty required fields")
    if df["ID"].duplicated().any():
        duplicate_ids = df.loc[df["ID"].duplicated(keep=False), "ID"].astype(str).tolist()
        raise ValueError(f"{path} has duplicate IDs: {duplicate_ids[:10]}")
    return df


def _build_corrected_sample(*, current_df: pd.DataFrame, corrected_df: pd.DataFrame) -> pd.DataFrame:
    current_ids = current_df["ID"].astype(str).tolist()
    corrected_by_id = corrected_df.assign(ID=corrected_df["ID"].astype(str)).set_index("ID", drop=False)
    missing = [sample_id for sample_id in current_ids if sample_id not in corrected_by_id.index]
    if missing:
        raise ValueError(f"Corrected SATAs file is missing sampled IDs: {missing[:10]}")
    sampled_df = corrected_by_id.loc[current_ids, REQUIRED_COLUMNS].reset_index(drop=True)
    for column in ["Question_Chinese", "Question_English"]:
        left = current_df[column].fillna("").astype(str).reset_index(drop=True)
        right = sampled_df[column].fillna("").astype(str).reset_index(drop=True)
        if not left.equals(right):
            changed_ids = [
                current_ids[index]
                for index, (left_value, right_value) in enumerate(zip(left, right))
                if left_value != right_value
            ]
            raise ValueError(f"Corrected SATAs changed {column} for sampled IDs: {changed_ids[:10]}")
    return sampled_df


def _replace_satas_file(*, current_path: Path, sampled_df: pd.DataFrame) -> None:
    backup_dir = ROOT / "results" / "audit-judge-integrity" / "satas_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{current_path.stem}.before_reference_fix.xlsx"
    if not backup_path.exists():
        shutil.copy2(current_path, backup_path)
    sampled_df.to_excel(current_path, index=False)


def _load_corrected_satas_samples(config) -> dict[str, Sample]:
    registry = FileRegistry(config)
    items = registry.resolve(source_files={"SATAs.xlsx"})
    if len(items) != 1:
        raise RuntimeError(f"Expected one SATAs registry item, got {len(items)}")
    loader = LoaderFactory.create(items[0].entry.loader_name)
    samples = list(loader.iter_samples(items[0]))
    if len(samples) != 150:
        raise RuntimeError(f"Expected 150 sampled SATAs records, got {len(samples)}")
    return {sample.sample_id: sample for sample in samples}


async def _rejudge_model_satas(
    *,
    output_root: Path,
    model_id: str,
    run_id: str,
    sample_by_id: dict[str, Sample],
) -> dict[str, Any]:
    paths = OutputPaths.build(output_root, run_id, model_id)
    if not paths.judged_path.exists():
        raise FileNotFoundError(paths.judged_path)
    if not paths.raw_path.exists():
        raise FileNotFoundError(paths.raw_path)

    judged_before = _load_jsonl(paths.judged_path)
    raw_before = _load_jsonl(paths.raw_path)
    judged_after, judged_stats = await _rewrite_records_with_new_satas_judgment(
        records=judged_before,
        sample_by_id=sample_by_id,
        update_judgment=True,
    )
    raw_after, raw_stats = await _rewrite_records_with_new_satas_judgment(
        records=raw_before,
        sample_by_id=sample_by_id,
        update_judgment=False,
    )
    if judged_stats["matched"] != 150:
        raise RuntimeError(f"{model_id} judged SATAs count is {judged_stats['matched']}, expected 150")
    if raw_stats["matched"] != 150:
        raise RuntimeError(f"{model_id} raw SATAs count is {raw_stats['matched']}, expected 150")

    _write_jsonl(paths.judged_path, judged_after)
    _write_jsonl(paths.raw_path, raw_after)
    summary = build_summary(paths.judged_path, paths.failures_path)
    paths.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "judged_total": len(judged_after),
        "raw_total": len(raw_after),
        "satas_judged": judged_stats,
        "satas_raw": raw_stats,
        "summary_total_failures": summary.get("total_failures"),
    }


async def _rewrite_records_with_new_satas_judgment(
    *,
    records: list[dict[str, Any]],
    sample_by_id: dict[str, Sample],
    update_judgment: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    judge = TeachingHarmJudge()
    changed_judgments = 0
    matched = 0
    pass_count = 0
    after: list[dict[str, Any]] = []
    for record in records:
        if record.get("task") != "teaching_harm" and record.get("source_file") != "SATAs.xlsx":
            after.append(record)
            continue
        sample = sample_by_id.get(str(record.get("sample_id") or ""))
        if sample is None:
            raise RuntimeError(f"Cannot map SATAs record sample_id={record.get('sample_id')!r}")
        matched += 1
        updated = dict(record)
        updated["source_path"] = sample.source_path
        updated["prompt"] = sample.prompt
        updated["reference"] = sample.reference
        metadata = dict(updated.get("metadata") or {})
        metadata.update(sample.metadata)
        updated["metadata"] = metadata
        if update_judgment:
            previous = (updated.get("judge_result"), updated.get("score"), updated.get("judge_metadata"))
            response = ModelResponse(
                text=updated.get("model_response"),
                raw_payload=updated.get("raw_response") if isinstance(updated.get("raw_response"), dict) else {},
                latency_ms=updated.get("latency_ms"),
                retry_count=int(updated.get("retry_count") or 0),
            )
            judgment = await judge.judge(sample, response)
            updated["judge_result"] = judgment.judge_result
            updated["score"] = judgment.score
            updated["judge_reason"] = judgment.judge_reason
            updated["judge_metadata"] = judgment.judge_metadata
            current = (updated.get("judge_result"), updated.get("score"), updated.get("judge_metadata"))
            if current != previous:
                changed_judgments += 1
            if updated.get("judge_result") == "pass":
                pass_count += 1
        after.append(updated)
    return after, {"matched": matched, "changed_judgments": changed_judgments, "pass_count": pass_count}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temp_path.replace(path)


if __name__ == "__main__":
    main()
