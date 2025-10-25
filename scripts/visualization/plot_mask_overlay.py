"""
Plot the mask channel overlaid on the u velocity channel to verify cylinder position.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def plot_mask_overlay(case_dir, timestep=500, save_path=None):
    """
    Plot mask channel over u velocity channel.

    Args:
        case_dir: Path to case directory
        timestep: Which timestep to visualize
        save_path: Optional path to save figure
    """
    case_dir = Path(case_dir)

    # Load data
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")

    with open(case_dir / "case.json", "r") as f:
        params = json.load(f)

    # Get domain bounds
    x_min, x_max = params['x_min'], params['x_max']
    y_min, y_max = params['y_min'], params['y_max']
    radius = params['radius']

    # Generate mask using the same logic as load_case_data_fix
    H, W = u.shape[1], u.shape[2]
    dx = (x_max - x_min) / W
    dy = (y_max - y_min) / H

    # Calculate cylinder center
    if "center_x" in params and "center_y" in params:
        center_x = params["center_x"]
        center_y = params["center_y"]
    else:
        # Cylinder is at the origin (0, 0) of the coordinate system
        center_x = 0.0
        center_y = 0.0

    # Create mask
    mask = np.ones((H, W))
    cylinder_points = 0

    for i in range(H):
        for j in range(W):
            # Physical coordinates of grid cell center
            x = x_min + (j + 0.5) * dx
            y = y_min + (i + 0.5) * dy

            # Distance from cylinder center
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2

            if dist_sq <= radius ** 2:
                mask[i, j] = 0
                cylinder_points += 1

    # Set boundary cells to 0
    mask[0, :] = 0    # Top boundary
    mask[-1, :] = 0   # Bottom boundary
    mask[:, 0] = 0    # Left boundary (inlet)

    # Select timestep
    t_idx = min(timestep, u.shape[0] - 1)
    u_frame = u[t_idx]

    # Create coordinate grids
    x_coords = x_min + (np.arange(W) + 0.5) * dx
    y_coords = y_min + (np.arange(H) + 0.5) * dy

    # Print info
    print(f"Case: {case_dir.name}")
    print(f"Domain: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"Cylinder: center=({center_x:.6f}, {center_y:.6f}), radius={radius:.4f}")
    print(f"Grid: {H}x{W}, dx={dx:.6f}, dy={dy:.6f}")
    print(f"Cylinder mask: {cylinder_points} grid points")
    print(f"Timestep: {t_idx}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: u velocity
    im0 = axes[0].imshow(u_frame, cmap='RdBu_r', origin='lower',
                         extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[0].set_title(f'u velocity (t={t_idx})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # Draw expected cylinder
    circle0 = plt.Circle((center_x, center_y), radius, color='lime', fill=False,
                         linewidth=2, linestyle='--', label='Expected cylinder')
    axes[0].add_patch(circle0)
    axes[0].axhline(center_y, color='lime', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].axvline(center_x, color='lime', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    # Plot 2: mask
    im1 = axes[1].imshow(mask, cmap='gray', origin='lower',
                         extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[1].set_title('Mask (0=cylinder/boundary, 1=fluid)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    # Draw expected cylinder
    circle1 = plt.Circle((center_x, center_y), radius, color='red', fill=False,
                         linewidth=2, linestyle='--', label='Expected cylinder')
    axes[1].add_patch(circle1)
    axes[1].axhline(center_y, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].axvline(center_x, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1])

    # Plot 3: u velocity with mask overlay
    im2 = axes[2].imshow(u_frame, cmap='RdBu_r', origin='lower',
                         extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    # Overlay mask as contour
    mask_contour = axes[2].contour(x_coords, y_coords, mask, levels=[0.5],
                                    colors='lime', linewidths=3)

    axes[2].set_title('u velocity + mask overlay')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    # Draw expected cylinder
    circle2 = plt.Circle((center_x, center_y), radius, color='yellow', fill=False,
                         linewidth=2, linestyle='--', label='Expected cylinder')
    axes[2].add_patch(circle2)
    axes[2].axhline(center_y, color='white', linestyle='--', alpha=0.5, linewidth=1)
    axes[2].axvline(center_x, color='white', linestyle='--', alpha=0.5, linewidth=1)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', linewidth=3, label='Mask boundary'),
        Line2D([0], [0], color='yellow', linewidth=2, linestyle='--', label='Expected cylinder')
    ]
    axes[2].legend(handles=legend_elements)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()

    return mask


if __name__ == "__main__":
    # Test on case0001
    case_path = Path(r"F:\Users\Ricardo\Documents\GitHub\CFDBench\data\cylinder\geo\case0001")

    if case_path.exists():
        mask = plot_mask_overlay(case_path, timestep=500,
                                 save_path='mask_overlay_verification.png')
    else:
        print(f"Case not found at {case_path}")
