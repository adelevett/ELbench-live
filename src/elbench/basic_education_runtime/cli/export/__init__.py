import click

from elbench.basic_education_runtime.cli.export.json_ import json

try:
    from elbench.basic_education_runtime.cli.export.label_studio import label_studio
except Exception:  # optional command; do not block core pipeline/export-json flow
    label_studio = None


@click.group(help="Export chat databases")
def export():
    pass


export.add_command(json)
if label_studio is not None:
    export.add_command(label_studio)

