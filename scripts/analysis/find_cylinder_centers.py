"""
Quick script to find cylinder centers across multiple cases.
"""
import numpy as np
from pathlib import Path
import json

def find_cylinder_center(case_dir):
    """Find cylinder by looking at interior minimum velocity."""
    case_dir = Path(case_dir)

    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")

    with open(case_dir / "case.json", "r") as f:
        params = json.load(f)

    # Use late timestep
    t_idx = min(500, u.shape[0] - 1)
    u_t = u[t_idx]
    v_t = v[t_idx]

    H, W = u_t.shape
    x_min, x_max = params['x_min'], params['x_max']
    y_min, y_max = params['y_min'], params['y_max']

    dx = (x_max - x_min) / W
    dy = (y_max - y_min) / H

    # Find minimum velocity in interior (exclude boundaries)
    vel_mag = np.sqrt(u_t**2 + v_t**2)
    interior_vel = vel_mag[5:-5, 5:-5]

    min_idx = np.unravel_index(np.argmin(interior_vel), interior_vel.shape)
    i_interior, j_interior = min_idx[0] + 5, min_idx[1] + 5

    phys_x = x_min + (j_interior + 0.5) * dx
    phys_y = y_min + (i_interior + 0.5) * dy

    return {
        'center_x': phys_x,
        'center_y': phys_y,
        'x_min': x_min,
        'y_min': y_min,
        'radius': params['radius']
    }

if __name__ == "__main__":
    data_root = Path(r"F:\Users\Ricardo\Documents\GitHub\CFDBench\data\cylinder\geo")

    results = []
    for case_dir in sorted(data_root.glob("case*"))[:5]:  # First 5 cases
        result = find_cylinder_center(case_dir)
        results.append(result)
        print(f"{case_dir.name}: center=({result['center_x']:.6f}, {result['center_y']:.6f}), "
              f"x_min={result['x_min']:.3f}, radius={result['radius']:.4f}")

    # Check if there's a pattern
    print("\n=== Analysis ===")
    if len(results) > 1:
        # Check if x_center - x_min is constant
        offsets_x = [r['center_x'] - r['x_min'] for r in results]
        offsets_y = [r['center_y'] - r['y_min'] for r in results]

        print(f"x_center - x_min: {offsets_x}")
        print(f"  Mean: {np.mean(offsets_x):.6f}, Std: {np.std(offsets_x):.6f}")

        print(f"y_center - y_min: {offsets_y}")
        print(f"  Mean: {np.mean(offsets_y):.6f}, Std: {np.std(offsets_y):.6f}")

        # Alternative: check if it's a ratio
        ratios_x = [(r['center_x'] - r['x_min']) / (r['x_max'] - r['x_min'])
                    for r in results if 'x_max' in locals()]

        print(f"\nRecommendation:")
        if np.std(offsets_x) < 0.001:
            print(f"  Use: center_x = x_min + {np.mean(offsets_x):.6f}")
        else:
            print(f"  Cylinder x position varies across cases")

        if np.std(offsets_y) < 0.001:
            print(f"  Use: center_y = y_min + {np.mean(offsets_y):.6f}")
        else:
            print(f"  Cylinder y position varies across cases")
