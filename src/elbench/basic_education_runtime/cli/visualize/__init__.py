import click

from pathlib import Path


@click.command(help="Visualize the results in all CSV file in the specified directory.")
@click.argument(
    "input_dir",
    type=click.Path(exists=True),
)
@click.option(
    "--x-rotation",
    type=int,
    default=30,
)
def visualize(input_dir: str, x_rotation: int):
    visualize_logic(input_dir, x_rotation)


def visualize_logic(input_dir: str, x_rotation: int):
    color_palette = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    from importlib.resources import files

    font_path = files("elbench.basic_education_runtime.assets.fonts").joinpath("sarasa-mono-sc-regular.ttf")
    font_path = str(font_path)

    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = "Sarasa Mono SC"
    plt.rcParams["axes.unicode_minus"] = False  # 瑙ｅ喅璐熷彿鏄剧ず闂

    input_path = Path(input_dir)
    csvs = input_path.rglob("*.csv")

    task_name = ""
    keys = []
    models = []
    values = {}

    for csv in csvs:
        stem_split = csv.stem.rsplit("_", 1)
        if task_name == "":
            task_name = stem_split[0]
        elif task_name != stem_split[0]:
            raise ValueError(
                f"Multiple task names found in CSV files. {task_name} and {stem_split[0]} are different."
            )

        model = stem_split[1]
        models.append(model)

        data = pd.read_csv(csv)
        data = data.drop(columns=["task_id", "avg"])
        data = data.iloc[-1].to_dict()

        if not keys:
            keys = list(data.keys())
            for k in keys:
                values[k] = []
        elif keys != list(data.keys()):
            raise ValueError(
                f"Data keys do not match across CSV files. [{','.join(keys)}] and [{','.join(data.keys())}] are different."
            )

        for k in keys:
            values[k].append(data[k])

    # 鏋勫缓 DataFrame
    df_dict = {"": models}
    for k in keys:
        df_dict[k] = values[k]

    df = pd.DataFrame(df_dict)

    ncol = min(len(keys), 5)

    # ==== 鉁?鑷€傚簲鐢诲竷瀹藉害 ====
    fig_width = max(8, len(df) * 0.8, ncol * 2)  # 姣忎釜妯″瀷 0.8 鑻卞锛屾渶灏忓搴︿负 8
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    df.set_index("").plot(kind="bar", stacked=True, ax=ax, color=color_palette)
    ax.set_xticklabels(df[""], rotation=x_rotation)
    ax.set_title(f"{task_name}")

    # 鉁?璁剧疆鍥句緥浣嶇疆鍒板浘琛ㄤ笅鏂癸紝鎵撴暎涓哄鍒?
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=ncol,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(input_path / f"stack_{task_name}.png", dpi=300)

    # ==== 鉁?闆疯揪鍥?====
    # 鍑嗗鏁版嵁
    num_vars = len(keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闂悎鍥惧舰

    # 璁＄畻鎵€鏈夋暟鍊肩殑鏈€灏忓€?
    all_scores = [values[k] for k in keys]
    min_value = min([min(score_list) for score_list in all_scores])

    # 璁剧疆闆疯揪鍥剧殑鏈€灏忓€煎師鐐?
    min_value -= 1  # 鏈€灏忓€煎噺鍘?1

    # 缁樺埗闆疯揪鍥?
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, model in enumerate(models):
        scores = [values[k][idx] for k in keys]
        scores += scores[:1]  # 闂悎鍥惧舰
        ax.plot(
            angles,
            scores,
            label=model,
            linewidth=2,
            color=color_palette[idx % len(color_palette)],
        )
        ax.fill(angles, scores, alpha=0.1)

    # 璋冩暣闆疯揪鍥剧殑鍗婂緞鑼冨洿锛岀‘淇濅粠 (min_value - 1) 寮€濮?
    ax.set_ylim(min_value, max([max(score_list) for score_list in all_scores]) + 1)

    # 璁剧疆鏍囩鍜屾牱寮?
    ax.set_thetagrids(np.degrees(angles[:-1]), keys)  # type: ignore
    ax.set_title(f"{task_name}", size=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(input_path / f"radar_{task_name}.png", dpi=300)

