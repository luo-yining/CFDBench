import matplotlib.pyplot as plt

from utils import get_output_dir, load_json
from args import Args


args = Args()
args.data_name = "cavity_prop_bc_geo"
models = [
    "auto_deeponet",
    "auto_edeeponet",
    "fno",
    "unet",
]

model_to_labels = {
    "auto_deeponet": "Auto-DeepONet",
    "auto_edeeponet": "Auto-EDeepONet",
    "fno": "FNO",
    "unet": "U-Net",
}

output_dir = get_output_dir(args)
print(output_dir)

for model in models:
    args.model = model
    output_dir = get_output_dir(args, is_auto=True)
    metrics_path = output_dir / "multistep_metrics.json"
    metrics = load_json(metrics_path)
    nmse = [x["nmse"] for x in metrics][:10]
    plt.plot(nmse, label=model_to_labels[model])
plt.xlabel('Step')
plt.yscale('log')
plt.legend()
plt.savefig('multistep_inference.pdf', bbox_inches='tight')
plt.show()
