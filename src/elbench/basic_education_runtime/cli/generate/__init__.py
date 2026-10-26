import click

from pathlib import Path
from elbench.basic_education_runtime.langchain_compat import set_debug

# set_debug(True)


@click.command("generate", help="Generate conversions for all tasks")
@click.option("--config", default="config.yaml", help="Path to the configuration file.")
@click.option("--debug", default=False, help="Debug Mode", is_flag=True)
def generate(config: str, debug: bool):
    set_debug(debug)
    from elbench.basic_education_runtime.config import load_conf

    path = Path(config)
    load_conf(path)
    generate_logic()


def generate_logic():
    from elbench.basic_education_runtime.run import run
    import asyncio

    asyncio.run(run())

