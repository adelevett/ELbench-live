import click

from elbench.basic_education_runtime.langchain_compat import set_debug

from elbench.basic_education_runtime.cli.generate import generate, generate_logic
from elbench.basic_education_runtime.cli.eval import eval, eval_logic
from elbench.basic_education_runtime.cli.visualize import visualize
from elbench.basic_education_runtime.cli.draw import draw
from elbench.basic_education_runtime.cli.export import export
from elbench.basic_education_runtime.cli.export.json_ import export_json_logic


@click.command(
    help="Run the pipeline to generate, export JSON files, and evaluate the results."
)
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the configuration file"
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
def pipeline(config, debug=False):
    set_debug(debug)
    from elbench.basic_education_runtime.config import load_conf

    load_conf(config)
    generate_logic()
    export_json_logic()
    eval_logic(avg=True)


@click.group()
def main():
    pass


main.add_command(generate)
main.add_command(export)
main.add_command(eval)
main.add_command(pipeline)

main.add_command(visualize)

main.add_command(draw)


if __name__ == "__main__":
    main()
