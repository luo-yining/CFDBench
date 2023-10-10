from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_output_dir, load_json
from args import Args


MODEL_TO_LABEL = {
    "ffn": "FFN",
    "deeponet": "DeepONet",
    "auto_ffn": "Auto-FFN",
    "auto_deeponet": "Auto-DeepONet",
    "auto_edeeponet": "Auto-EDeepONet",
    "auto_deeponet_cnn": "Auto-DeepONetCNN",
    "fno": "FNO",
    "unet": "U-Net",
    "resnet": "ResNet",
}


def plot(scores: list, models: list, out_path: str):
    # plt.ylim([1e-5, 1e0])
    labels = [MODEL_TO_LABEL[model] for model in models]
    for i in range(len(models)):
        score = scores[i]
        label = labels[i]
        model = models[i]

        marker = dict(
            ffn="s",
            deeponet="s",
            auto_ffn="^",
            auto_deeponet="v",
            auto_edeeponet="<",
            auto_deeponet_cnn=">",
            fno="o",
            unet="o",
        )[model]
        if model in ["fno", "unet", "resnet"]:
            linestyle = "dashed"
        else:
            linestyle = None

        colors = dict(
            ffn=tuple(c / 255 for c in (157, 225, 206)),
            deeponet=tuple(c / 255 for c in (157, 225, 206)),
            auto_ffn=tuple(c / 255 for c in (254, 204, 166)),
            auto_deeponet=tuple(c / 255 for c in (255, 128, 33)),
            auto_edeeponet=tuple(c / 255 for c in (249, 179, 167)),
            auto_deeponet_cnn=tuple(c / 255 for c in (241, 65, 36)),
            fno=tuple(c / 255 for c in (94, 204, 243)),
            unet=tuple(c / 255 for c in (94, 204, 243)),
        )

        if model in ["unet", "ffn"]:
            markerfacecolor = "w"
        else:
            markerfacecolor = None
        xs = list(range(1, len(score) + 1))
        plt.plot(
            xs,
            score,
            label=label,
            color=colors[model],
            marker=marker,
            linestyle=linestyle,
            # fillstyle=fill_style,
            markerfacecolor=markerfacecolor,
        )
    plt.xlabel("Forward Propagation Steps")
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlim((0, 20))
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.legend(ncol=2)
    print("Saving to", out_path)
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches='tight')
    # plt.show()
    plt.clf()


def get_scores(problem_name: str, models: List[str]) -> List[List[float]]:
    args = Args()
    args.data_name = problem_name + "_prop"
    scores = []
    for model in models:
        args.model = model
        if problem_name != "cylinder" and model in ["fno", "unet", "resnet"]:
            subset_name = "prop_bc_geo"
        else:
            subset_name = "prop"

        args.data_name = problem_name + "_" + subset_name

        if args.model in ["ffn", "deeponet"]:
            output_dir = get_output_dir(args, is_auto=False)
        else:
            output_dir = get_output_dir(args, is_auto=True)
        metrics_path = output_dir / "multistep_metrics.json"
        metrics = load_json(metrics_path)
        nmse = [x["nmse"] for x in metrics][:20]
        scores.append(nmse)
    return scores


def main():
    MODELS = [
        "ffn",
        "deeponet",
        "auto_ffn",
        "auto_deeponet",
        "auto_edeeponet",
        "auto_deeponet_cnn",
        "fno",
        "unet",
        # "resnet",
    ]

    problems = [
        "cavity",
        "tube",
        "dam",
        "cylinder",
    ]
    for problem_name in problems:
        scores = get_scores(problem_name, MODELS)
        out_path = f"figs/multistep_infer_{problem_name}.pdf"
        sns.set_style('whitegrid')
        plot(scores, MODELS, out_path)
        plt.clf()


if __name__ == "__main__":
    main()
