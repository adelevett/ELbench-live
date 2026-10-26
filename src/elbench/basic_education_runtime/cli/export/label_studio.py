import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from typing import Dict, Any

import json as jsonmodule
import asyncio
from tqdm.asyncio import tqdm

import click

from pathlib import Path

from elbench.basic_education_runtime.langchain_compat import set_debug


def export_label_studio_logic():
    from elbench.basic_education_runtime.config import CONFIG

    input = CONFIG.globals.memory.path
    output = input

    dbfiles = []
    files = input.iterdir()
    for file in files:
        if file.suffix == ".db":
            dbfiles.append(file.absolute())

    from elbench.basic_education_runtime.cli.export.exporter.label_studio_ import aexport_label_studio
    from elbench.basic_education_runtime.cli.export.const.label_studio import generate_label_studio_interface

    if CONFIG.evaluation is None:
        raise ValueError("Evaluation瀛楁鏈厤缃?)

    tasks = []
    for dbfile in dbfiles:
        task = aexport_label_studio(dbfile)
        tasks.append(task)

    results = asyncio.run(tqdm.gather(*tasks))
    with open(output / f"label_studio.json", "w", encoding="utf-8") as f:
        jsonmodule.dump(results, f, ensure_ascii=False, indent=4)

    template = generate_label_studio_interface(CONFIG.evaluation.format)
    with open(output / "label_studio.txt", "w", encoding="utf-8") as f:
        f.write(template)



@click.command(help="Export chat databases to Label Studio Data format")
@click.option(
    "--config", default="config.yaml", help="Directory containing chat databases"
)
@click.option("--debug", default=False, help="Debug Mode", is_flag=True)
def label_studio(config: str, debug: bool):
    set_debug(debug)
    from elbench.basic_education_runtime.config import load_conf

    path = Path(config)
    load_conf(path)
    export_label_studio_logic()
