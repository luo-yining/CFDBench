"""
Utility script to plot training and validation losses from loss_history.json

Usage:
    cd scripts/visualization
    python plot_losses.py --result_dir ../../result/auto/cavity_geo/dt0.1/unet/...

    Or with absolute path:
    python plot_losses.py --result_dir /data/CFDBench/result/auto/cavity_geo/dt0.1/unet/...

    Or run from project root:
    python scripts/visualization/plot_losses.py --result_dir result/auto/cavity_geo/dt0.1/unet/...
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_loss_history(loss_history_path: Path, output_path: Path = None):
    """
    Plot training and validation losses from loss_history.json

    Args:
        loss_history_path: Path to loss_history.json file
        output_path: Path to save the plot (optional, defaults to same dir as json)
    """
    # Load loss history
    with open(loss_history_path, 'r') as f:
        loss_history = json.load(f)

    train_losses = loss_history["train_losses"]
    dev_losses_data = loss_history["dev_losses"]
    epochs = loss_history["epochs"]

    # Extract dev loss epochs and values
    dev_epochs = [item["epoch"] for item in dev_losses_data]
    dev_losses = [item["dev_loss"] for item in dev_losses_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training loss (every epoch)
    ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4, linewidth=2)

    # Plot dev loss (only evaluated epochs)
    ax.plot(dev_epochs, dev_losses, label='Dev Loss', marker='s', markersize=6, linewidth=2)

    # Formatting
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (NMSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set x-axis to integers
    ax.set_xticks(range(0, len(epochs), max(1, len(epochs) // 10)))

    # Add best dev loss annotation
    if dev_losses:
        best_dev_loss = min(dev_losses)
        best_epoch = dev_epochs[dev_losses.index(best_dev_loss)]
        ax.axhline(y=best_dev_loss, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(
            0.02, 0.98,
            f'Best Dev Loss: {best_dev_loss:.6f} (Epoch {best_epoch})',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = loss_history_path.parent / "loss_curve.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to: {output_path}")

    # Also save a summary
    summary_path = loss_history_path.parent / "loss_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Loss Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total epochs: {len(epochs)}\n")
        f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
        if dev_losses:
            f.write(f"Final dev loss: {dev_losses[-1]:.6f}\n")
            f.write(f"Best dev loss: {best_dev_loss:.6f} (Epoch {best_epoch})\n")
        f.write(f"\nTrain loss improvement: {train_losses[0]:.6f} → {train_losses[-1]:.6f}\n")
        if len(dev_losses) > 1:
            f.write(f"Dev loss improvement: {dev_losses[0]:.6f} → {dev_losses[-1]:.6f}\n")

    print(f"Loss summary saved to: {summary_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation losses")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to result directory containing loss_history.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot (default: <result_dir>/loss_curve.png)"
    )

    args = parser.parse_args()

    result_path = Path(args.result_dir)

    # Resolve to absolute path for clearer error messages
    if not result_path.is_absolute():
        result_path = result_path.resolve()

    # Handle both directory path and direct file path
    if result_path.is_file() and result_path.name == "loss_history.json":
        # User provided the JSON file directly
        loss_history_path = result_path
        result_dir = result_path.parent
    elif result_path.is_dir():
        # User provided a directory
        result_dir = result_path
        loss_history_path = result_dir / "loss_history.json"
    else:
        print(f"Error: Path not found: {result_path}")
        print("\nTip: Provide either:")
        print("  1. A directory containing loss_history.json")
        print("  2. The direct path to loss_history.json")
        print(f"\nCurrent directory: {Path.cwd()}")
        return

    if not loss_history_path.exists():
        print(f"Error: {loss_history_path} not found!")
        print("Make sure you've run training with train_auto_v2.py first.")
        print(f"\nLooking in directory: {result_dir}")
        print("Expected file: loss_history.json")

        # Show what files are actually in the directory
        if result_dir.exists():
            files = list(result_dir.glob("*.json"))
            if files:
                print(f"\nJSON files found in directory:")
                for f in files:
                    print(f"  - {f.name}")
        return

    output_path = Path(args.output) if args.output else None

    plot_loss_history(loss_history_path, output_path)


if __name__ == "__main__":
    main()
