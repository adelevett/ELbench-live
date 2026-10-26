import click

from pathlib import Path
from typing import Dict, Any

from elbench.basic_education_runtime.langchain_compat import set_debug
from tenacity import RetryError


@click.command(help="Evaluate the performance of a model on a dataset")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration file",
)
@click.option("--debug", default=False, help="Debug Mode", is_flag=True)
@click.option("--avg/--no-avg", default=True, help="Calculate the average score")
def eval(config: Path, debug: bool, avg: bool):
    set_debug(debug)
    from elbench.basic_education_runtime.config import load_conf

    load_conf(config)
    eval_logic(avg)


def eval_logic(avg: bool):
    from elbench.basic_education_runtime.config import CONFIG

    input_dir = CONFIG.globals.memory.path
    import asyncio
    from elbench.basic_education_runtime.evaluation import evaluate
    from elbench.basic_education_runtime.model import init_chat_model_from_dict

    from elbench.basic_education_runtime.entity import ExportFormat
    from tqdm.asyncio import tqdm
    import json

    input_dir = Path(input_dir)

    eval_path = input_dir / "eval"
    eval_path.mkdir(exist_ok=True)

    sem = asyncio.Semaphore(CONFIG.globals.concurrency)

    async def eval_task(model, file: Path) -> Dict[str, Any]:
        async with sem:
            ef = ExportFormat.from_json_file(file)
            try:
                eval = await evaluate(model, ef)
                with open(eval_path / file.name, "w", encoding="utf8") as f:
                    json.dump(eval, f, ensure_ascii=False, indent=4)
                return eval
            except RetryError as e:
                print(f"Error evaluating {file}", e.last_attempt.exception())
                return {}

    async def main():
        assert CONFIG.evaluation
        model = init_chat_model_from_dict(CONFIG.models[CONFIG.evaluation.model])

        to_eval_files = list(input_dir.glob("*.json"))
        task_ids = [file.stem for file in to_eval_files]
        eval_tasks = []
        for file in to_eval_files:
            eval_tasks.append(eval_task(model, file))

        evals = await tqdm.gather(*eval_tasks)

        csv_utf8 = open(
            eval_path / f"{CONFIG.evaluation.name}.csv", "w", encoding="utf-8"
        )
        # csv_gbk = open(eval_path / f"{CONFIG.evaluation.name}-gbk.csv", "w", encoding="gbk")

        metric_fields = _metric_fields_from_evals(evals)
        if not metric_fields:
            csv_utf8.write("task_id\n")
            csv_utf8.close()
            raise RuntimeError("No successful evaluation outputs were produced.")

        title = ["task_id"]
        for field in metric_fields:
            title.append(field)

        if avg:
            title.append("avg")

        csv_utf8.write(",".join(title) + "\n")
        # csv_gbk.write(",".join(title) + "\n")

        if avg:
            row = len(task_ids) + 1
            metric_count = len(metric_fields)
            col = metric_count + 1

            matrix = [[0.0] * col for _ in range(row)]

            # 缁熻鏁版嵁骞惰绠楁瘡琛屽钩鍧囧€?
            for idx, (task_id, eval) in enumerate(zip(task_ids, evals)):
                contents = [task_id]
                for sub_idx, field in enumerate(metric_fields):
                    raw_value = eval.get(field, 0)
                    v = float(raw_value)
                    matrix[idx][sub_idx] = v
                    contents.append(f"{raw_value}")
                sum = 0
                for i in matrix[idx][:-1]:
                    sum += i
                # 鏈€鍚庝竴鍒楃殑鏁板瓧 = 姣忓垪鐨勫拰闄や互鍒楁暟-1
                matrix[idx][col - 1] = sum / metric_count
                contents.append(f"{matrix[idx][col - 1]:.2f}")
                csv_utf8.write(",".join(contents) + "\n")
                # csv_gbk.write(",".join(contents) + "\n")
            # 璁＄畻姣忓垪鐨勫钩鍧囧€?
            for col_idx in range(col):
                # print(matrix)
                sum = 0
                # 璁＄畻姣忎竴鍒楅櫎鍘绘渶鍚庝竴涓厓绱犵殑鍜?
                for row_idx in range(row - 1):
                    sum += matrix[row_idx][col_idx]
                matrix[-1][col_idx] = sum / (row - 1)

            write_str = ["%.2f" % i for i in matrix[-1]]
            write_str.insert(0, "Avg")
            # 鍐欏叆鏈€鍚庝竴琛岀殑骞冲潎鍊?
            csv_utf8.write(",".join(write_str) + "\n")
            # csv_gbk.write(",".join(write_str) + "\n")
        else:
            for task_id, eval in zip(task_ids, evals):
                contents = [task_id]
                for field in metric_fields:
                    contents.append(f"{eval.get(field, 0)}")
                csv_utf8.write(",".join(contents) + "\n")
                # csv_gbk.write(",".join(contents) + "\n")

        csv_utf8.close()
        # csv_gbk.close()

    asyncio.run(main())


def _metric_fields_from_evals(evals: list[Dict[str, Any]]) -> list[str]:
    for item in evals:
        if item:
            return list(item.keys())
    return []

