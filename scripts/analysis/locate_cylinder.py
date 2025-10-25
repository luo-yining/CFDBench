"""
Script to locate the cylinder position in the velocity field data.
The cylinder should appear as a region where u ≈ 0 and v ≈ 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def locate_cylinder(case_dir):
    """Find cylinder location from velocity field data."""
    case_dir = Path(case_dir)

    # Load velocity data
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")

    # Load case parameters
    with open(case_dir / "case.json", "r") as f:
        case_params = json.load(f)

    print(f"Case: {case_dir.name}")
    print(f"Data shape: u={u.shape}, v={v.shape}")
    print(f"Domain: x=[{case_params['x_min']:.3f}, {case_params['x_max']:.3f}], "
          f"y=[{case_params['y_min']:.3f}, {case_params['y_max']:.3f}]")
    print(f"Radius: {case_params['radius']:.3f}")

    # Use first timestep
    u_frame = u[0]
    v_frame = v[0]

    # Calculate velocity magnitude
    vel_mag = np.sqrt(u_frame**2 + v_frame**2)

    # Find points with very low velocity (likely inside/near cylinder)
    threshold = 0.01  # Adjust if needed
    mask = vel_mag < threshold

    print(f"\nVelocity stats:")
    print(f"  u: min={u_frame.min():.4f}, max={u_frame.max():.4f}, mean={u_frame.mean():.4f}")
    print(f"  v: min={v_frame.min():.4f}, max={v_frame.max():.4f}, mean={v_frame.mean():.4f}")
    print(f"  |v|: min={vel_mag.min():.4f}, max={vel_mag.max():.4f}, mean={vel_mag.mean():.4f}")

    print(f"\nLow velocity points (|v| < {threshold}): {mask.sum()} grid points")

    if mask.sum() == 0:
        print("⚠️  No low-velocity points found! Cylinder may not be in domain or threshold too low.")
        # Try finding minimum velocity region
        min_vel_idx = np.unravel_index(np.argmin(vel_mag), vel_mag.shape)
        print(f"Minimum velocity at grid index: {min_vel_idx}")

        # Expand around minimum
        i_min, j_min = min_vel_idx
        window = 5
        i_start = max(0, i_min - window)
        i_end = min(vel_mag.shape[0], i_min + window + 1)
        j_start = max(0, j_min - window)
        j_end = min(vel_mag.shape[1], j_min + window + 1)

        local_vel = vel_mag[i_start:i_end, j_start:j_end]
        print(f"Local velocities around minimum:\n{local_vel}")

        return None

    # Find bounding box of low-velocity region
    rows, cols = np.where(mask)
    i_min, i_max = rows.min(), rows.max()
    j_min, j_max = cols.min(), cols.max()

    # Center in grid coordinates
    i_center = (i_min + i_max) / 2.0
    j_center = (j_min + j_max) / 2.0

    print(f"\nCylinder in grid coordinates:")
    print(f"  i (row) range: [{i_min}, {i_max}], center: {i_center:.1f}")
    print(f"  j (col) range: [{j_min}, {j_max}], center: {j_center:.1f}")

    # Convert to physical coordinates
    grid_height, grid_width = u_frame.shape
    x_min, x_max = case_params['x_min'], case_params['x_max']
    y_min, y_max = case_params['y_min'], case_params['y_max']

    dx = (x_max - x_min) / grid_width
    dy = (y_max - y_min) / grid_height

    # Physical coordinates (using cell centers)
    x_center = x_min + (j_center + 0.5) * dx
    y_center = y_min + (i_center + 0.5) * dy

    print(f"\nCylinder in physical coordinates:")
    print(f"  center_x: {x_center:.6f}")
    print(f"  center_y: {y_center:.6f}")

    # Estimate radius from masked region
    radius_i = (i_max - i_min) / 2.0
    radius_j = (j_max - j_min) / 2.0
    radius_phys_y = radius_i * dy
    radius_phys_x = radius_j * dx
    radius_avg = (radius_phys_x + radius_phys_y) / 2.0

    print(f"  Estimated radius (physical): {radius_avg:.6f}")
    print(f"  Expected radius: {case_params['radius']:.6f}")
    print(f"  Ratio: {radius_avg / case_params['radius']:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # u velocity
    im0 = axes[0, 0].imshow(u_frame, cmap='RdBu_r', origin='lower')
    axes[0, 0].set_title('u velocity')
    axes[0, 0].plot(j_center, i_center, 'go', markersize=10, label='Center')
    axes[0, 0].legend()
    plt.colorbar(im0, ax=axes[0, 0])

    # v velocity
    im1 = axes[0, 1].imshow(v_frame, cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('v velocity')
    axes[0, 1].plot(j_center, i_center, 'go', markersize=10, label='Center')
    axes[0, 1].legend()
    plt.colorbar(im1, ax=axes[0, 1])

    # velocity magnitude
    im2 = axes[1, 0].imshow(vel_mag, cmap='viridis', origin='lower')
    axes[1, 0].set_title('Velocity magnitude')
    axes[1, 0].plot(j_center, i_center, 'ro', markersize=10, label='Center')
    axes[1, 0].legend()
    plt.colorbar(im2, ax=axes[1, 0])

    # mask
    im3 = axes[1, 1].imshow(mask, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'Low velocity mask (|v| < {threshold})')
    axes[1, 1].plot(j_center, i_center, 'ro', markersize=10, label='Center')
    axes[1, 1].legend()
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f'cylinder_location_{case_dir.name}.png', dpi=150)
    print(f"\nSaved visualization to: cylinder_location_{case_dir.name}.png")
    plt.show()

    return {
        'center_x': x_center,
        'center_y': y_center,
        'radius_estimated': radius_avg,
        'grid_center_i': i_center,
        'grid_center_j': j_center
    }


if __name__ == "__main__":
    # Test on a few cases
    data_root = Path(r"F:\Users\Ricardo\Documents\GitHub\CFDBench\data\cylinder\geo")

    test_cases = ['case0001', 'case0002', 'case0003']

    results = {}
    for case_name in test_cases:
        case_path = data_root / case_name
        if case_path.exists():
            print("\n" + "="*70)
            result = locate_cylinder(case_path)
            results[case_name] = result
            print("="*70)
        else:
            print(f"Case {case_name} not found at {case_path}")

    print("\n\nSUMMARY:")
    print("="*70)
    for case_name, result in results.items():
        if result:
            print(f"{case_name}: center=({result['center_x']:.6f}, {result['center_y']:.6f}), "
                  f"radius={result['radius_estimated']:.6f}")
