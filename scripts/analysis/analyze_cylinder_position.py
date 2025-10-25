"""
Analyze the actual cylinder position by looking at flow patterns.
The cylinder creates a wake and stagnation point, not just zero velocity.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_flow_pattern(case_dir):
    """Analyze flow pattern to find cylinder location."""
    case_dir = Path(case_dir)

    # Load data
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")

    with open(case_dir / "case.json", "r") as f:
        case_params = json.load(f)

    print(f"\nCase: {case_dir.name}")
    print(f"Data shape: {u.shape}")
    print(f"Domain: x=[{case_params['x_min']}, {case_params['x_max']}], "
          f"y=[{case_params['y_min']}, {case_params['y_max']}]")
    print(f"Specified radius: {case_params['radius']}")

    # Use a late timestep to see developed flow
    t_idx = min(500, u.shape[0] - 1)
    u_t = u[t_idx]
    v_t = v[t_idx]

    H, W = u_t.shape
    x_min, x_max = case_params['x_min'], case_params['x_max']
    y_min, y_max = case_params['y_min'], case_params['y_max']

    dx = (x_max - x_min) / W
    dy = (y_max - y_min) / H

    print(f"Grid: {H}x{W}, dx={dx:.6f}, dy={dy:.6f}")

    # Create coordinate grids
    x_coords = x_min + (np.arange(W) + 0.5) * dx
    y_coords = y_min + (np.arange(H) + 0.5) * dy

    print(f"x range: [{x_coords[0]:.6f}, {x_coords[-1]:.6f}]")
    print(f"y range: [{y_coords[0]:.6f}, {y_coords[-1]:.6f}]")

    # Look for the cylinder by checking different criteria
    vel_mag = np.sqrt(u_t**2 + v_t**2)

    # 1. Find regions with very low u velocity (stagnation)
    print("\n=== Method 1: Looking for low u velocity (stagnation) ===")
    u_threshold = 0.05
    low_u = u_t < u_threshold
    print(f"Points with u < {u_threshold}: {low_u.sum()}")

    if low_u.sum() > 0:
        rows, cols = np.where(low_u)
        print(f"  Grid row range: [{rows.min()}, {rows.max()}]")
        print(f"  Grid col range: [{cols.min()}, {cols.max()}]")
        center_i = (rows.min() + rows.max()) / 2
        center_j = (cols.min() + cols.max()) / 2
        phys_x = x_min + (center_j + 0.5) * dx
        phys_y = y_min + (center_i + 0.5) * dy
        print(f"  Center (grid): i={center_i:.1f}, j={center_j:.1f}")
        print(f"  Center (physical): x={phys_x:.6f}, y={phys_y:.6f}")

    # 2. Check if there are NaN or masked values
    print("\n=== Method 2: Checking for NaN/inf values ===")
    nan_u = np.isnan(u_t) | np.isinf(u_t)
    nan_v = np.isnan(v_t) | np.isinf(v_t)
    print(f"NaN/inf in u: {nan_u.sum()}, in v: {nan_v.sum()}")

    # 3. Look at flow statistics per row to find obstruction
    print("\n=== Method 3: Flow statistics per row (y-coordinate) ===")
    mean_u_per_row = np.mean(u_t, axis=1)
    std_u_per_row = np.std(u_t, axis=1)

    # Find rows with high variation (indicates obstruction)
    high_var_rows = np.where(std_u_per_row > 0.3)[0]
    if len(high_var_rows) > 0:
        print(f"Rows with high u variation: {high_var_rows}")
        center_i = np.median(high_var_rows)
        phys_y = y_min + (center_i + 0.5) * dy
        print(f"  Center row: i={center_i:.1f}, y={phys_y:.6f}")

    # 4. Look for minimum velocity in interior (not boundaries)
    print("\n=== Method 4: Minimum velocity in interior ===")
    # Exclude boundaries
    interior_vel = vel_mag[5:-5, 5:-5]
    if interior_vel.size > 0:
        min_idx = np.unravel_index(np.argmin(interior_vel), interior_vel.shape)
        i_interior, j_interior = min_idx[0] + 5, min_idx[1] + 5
        phys_x = x_min + (j_interior + 0.5) * dx
        phys_y = y_min + (i_interior + 0.5) * dy
        print(f"  Min velocity at: i={i_interior}, j={j_interior}")
        print(f"  Physical: x={phys_x:.6f}, y={phys_y:.6f}")
        print(f"  Velocity: {vel_mag[i_interior, j_interior]:.6f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Velocity fields
    im0 = axes[0, 0].imshow(u_t, cmap='RdBu_r', origin='lower',
                             extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[0, 0].set_title(f'u velocity (t={t_idx})')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].axhline(0, color='green', linestyle='--', alpha=0.5, label='y=0')
    axes[0, 0].axvline(0, color='green', linestyle='--', alpha=0.5, label='x=0')
    axes[0, 0].legend()
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(v_t, cmap='RdBu_r', origin='lower',
                             extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[0, 1].set_title(f'v velocity (t={t_idx})')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].axhline(0, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(0, color='green', linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(vel_mag, cmap='viridis', origin='lower',
                             extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[0, 2].set_title('Velocity magnitude')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Mark expected cylinder position
    expected_x = 0.0
    expected_y = 0.0
    radius = case_params['radius']
    circle = plt.Circle((expected_x, expected_y), radius, color='red', fill=False,
                        linewidth=2, label='Expected cylinder')
    axes[0, 2].add_patch(circle)
    axes[0, 2].legend()
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Analysis
    axes[1, 0].plot(mean_u_per_row, y_coords, 'b-', label='mean u')
    axes[1, 0].axhline(0, color='green', linestyle='--', alpha=0.5, label='y=0')
    axes[1, 0].set_xlabel('Mean u velocity')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Mean u per row')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(std_u_per_row, y_coords, 'r-', label='std u')
    axes[1, 1].axhline(0, color='green', linestyle='--', alpha=0.5, label='y=0')
    axes[1, 1].set_xlabel('Std u velocity')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Std u per row')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Low velocity mask
    low_vel_mask = vel_mag < 0.1
    axes[1, 2].imshow(low_vel_mask, cmap='gray', origin='lower',
                      extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    axes[1, 2].set_title('Low velocity mask (<0.1)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
    circle2 = plt.Circle((expected_x, expected_y), radius, color='red', fill=False, linewidth=2)
    axes[1, 2].add_patch(circle2)

    plt.tight_layout()
    output_path = f'flow_analysis_{case_dir.name}.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")
    plt.show()


if __name__ == "__main__":
    case_path = Path(r"F:\Users\Ricardo\Documents\GitHub\CFDBench\data\cylinder\geo\case0001")

    if case_path.exists():
        analyze_flow_pattern(case_path)
    else:
        print(f"Case not found at {case_path}")
