from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime

from elbench.config import load_project_config
from elbench.execution import BenchmarkRunner, PreflightError, PreflightRunner, RunOptions, run_campaign_sync
from elbench.execution.campaign import build_default_api_pools, clean_model_artifacts
from elbench.registry import FileRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ELBench benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect resolved benchmark files")
    inspect_parser.add_argument("--config-root", default="configs")

    preflight_parser = subparsers.add_parser("preflight", help="Validate whether a run can produce official scores")
    preflight_parser.add_argument("--config-root", default="configs")
    preflight_parser.add_argument("--model-id", required=True)
    preflight_parser.add_argument("--module", action="append")
    preflight_parser.add_argument("--subset", action="append")
    preflight_parser.add_argument("--source-file", action="append")
    preflight_parser.add_argument("--dimension", action="append")
    preflight_parser.add_argument("--max-samples", type=int)
    preflight_parser.add_argument("--no-judge", action="store_true")
    preflight_parser.add_argument("--allow-mock-judge", action="store_true")
    preflight_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument("--config-root", default="configs")
    run_parser.add_argument("--model-id", required=True)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--module", action="append")
    run_parser.add_argument("--subset", action="append")
    run_parser.add_argument("--source-file", action="append")
    run_parser.add_argument("--dimension", action="append")
    run_parser.add_argument("--max-samples", type=int)
    run_parser.add_argument("--max-concurrency", type=int)
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--no-judge", action="store_true")
    run_parser.add_argument("--skip-preflight", action="store_true")
    run_parser.add_argument("--allow-mock-judge", action="store_true")
    run_parser.add_argument("--progress", action="store_true")

    campaign_parser = subparsers.add_parser("campaign", help="Run the default two-API benchmark campaign")
    campaign_parser.add_argument("--config-root", default="configs")
    campaign_parser.add_argument("--run-prefix", default=None)
    campaign_parser.add_argument("--max-concurrency", type=int)
    campaign_parser.add_argument("--module", action="append")
    campaign_parser.add_argument("--progress", action="store_true")
    campaign_parser.add_argument("--list-models", action="store_true")

    clean_parser = subparsers.add_parser("clean-model-results", help="Remove local result artifacts for models")
    clean_parser.add_argument("--config-root", default="configs")
    clean_parser.add_argument("--model-id", action="append", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_project_config(args.config_root)

    if args.command == "inspect":
        registry = FileRegistry(config)
        resolved = registry.resolve()
        for item in resolved:
            print(
                f"module={item.entry.module} subset={item.entry.subset} file={item.entry.canonical_name} path={item.path}"
            )
        return

    if args.command == "preflight":
        runner = PreflightRunner(config)
        report = runner.run(
            model_id=args.model_id,
            modules=set(args.module) if args.module else None,
            subsets=set(args.subset) if args.subset else None,
            source_files=set(args.source_file) if args.source_file else None,
            dimensions=set(args.dimension) if args.dimension else None,
            max_samples=args.max_samples,
            judge_enabled=not args.no_judge,
            require_real_judges=not args.allow_mock_judge,
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"ok={report.ok}")
            print(f"selected_modules={','.join(sorted(report.selection.selected_modules))}")
            print(f"total_samples={report.total_samples}")
            for module_name, count in sorted(report.by_module.items()):
                print(f"module[{module_name}]={count}")
            for issue in report.issues:
                print(f"{issue.level.upper()} {issue.code}: {issue.message}")
        raise SystemExit(0 if report.ok else 2)

    if args.command == "campaign":
        if args.list_models:
            for pool in build_default_api_pools(config):
                print(f"{pool.pool_id}: {', '.join(pool.model_ids)}")
            return
        result = run_campaign_sync(
            config_root=args.config_root,
            run_prefix=args.run_prefix,
            max_concurrency=args.max_concurrency,
            modules=set(args.module) if args.module else None,
            progress_enabled=args.progress,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "clean-model-results":
        deleted = clean_model_artifacts(config.app.output_root, args.model_id)
        for path in deleted:
            print(path)
        print(f"deleted={len(deleted)}")
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.skip_preflight:
        runner = PreflightRunner(config)
        try:
            runner.ensure_ok(
                model_id=args.model_id,
                modules=set(args.module) if args.module else None,
                subsets=set(args.subset) if args.subset else None,
                source_files=set(args.source_file) if args.source_file else None,
                dimensions=set(args.dimension) if args.dimension else None,
                max_samples=args.max_samples,
                judge_enabled=not args.no_judge,
                require_real_judges=not args.allow_mock_judge,
            )
        except PreflightError as exc:
            raise SystemExit(f"Preflight failed: {exc}") from exc

    runner = BenchmarkRunner(config)
    result = asyncio.run(
        runner.run(
            RunOptions(
                model_id=args.model_id,
                run_id=run_id,
                modules=set(args.module) if args.module else None,
                subsets=set(args.subset) if args.subset else None,
                source_files=set(args.source_file) if args.source_file else None,
                dimensions=set(args.dimension) if args.dimension else None,
                max_samples=args.max_samples,
                max_concurrency=args.max_concurrency,
                resume=not args.no_resume,
                judge_enabled=not args.no_judge,
                progress_enabled=args.progress,
            )
        )
    )
    print(result)


if __name__ == "__main__":
    main()

