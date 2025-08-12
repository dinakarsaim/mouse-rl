import numpy as np
import math
from scipy.ndimage import gaussian_filter1d

def analyze_mouse_path(points):
    """
    Analyze mouse movement path for robotic-like patterns.
    
    Args:
        points: list of (t, x, y) tuples, where t = time in ms
    
    Returns:
        dict with metrics for:
        - linearity_ratio
        - speed_variance
        - acceleration_variance
        - direction_change_variance
        - jitter_magnitude
        - pause_count
        - robotic_score (0-10, higher = more robotic)
    """
    if len(points) < 3:
        return {}
    
    # Unpack points
    times = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    
    # Distances & times
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_path_length = np.sum(dists)
    direct_distance = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    
    # 1. Linearity Ratio
    linearity_ratio = direct_distance / total_path_length if total_path_length > 0 else 0
    
    # Speeds & time deltas
    dt = np.diff(times) / 1000.0  # convert ms to sec
    dt[dt == 0] = 1e-6  # prevent division by zero
    speeds = dists / dt
    
    # 2. Speed Variance
    speed_variance = np.var(speeds)
    
    # 3. Acceleration Profile (variance)
    accelerations = np.diff(speeds) / dt[1:]  # acceleration in px/s^2
    acceleration_variance = np.var(accelerations) if len(accelerations) > 0 else 0
    
    # 4. Direction Change Variance
    angles = np.arctan2(np.diff(ys), np.diff(xs))
    direction_changes = np.diff(angles)
    direction_change_variance = np.var(direction_changes) if len(direction_changes) > 0 else 0
    
    # 5. Jitter Magnitude
    smooth_x = gaussian_filter1d(xs, sigma=2)
    smooth_y = gaussian_filter1d(ys, sigma=2)
    jitter_magnitude = np.mean(np.sqrt((xs - smooth_x)**2 + (ys - smooth_y)**2))
    
    # 6. Pause / Hesitation Count
    pause_count = np.sum(speeds < 1)  # counts points where speed < 1 px/s
    
    # --- Robotic Score (0-10) ---
    score = 0
    if linearity_ratio > 0.98: score += 2       # very straight
    if speed_variance < 0.001: score += 2       # constant speed
    if acceleration_variance < 0.001: score += 2  # no acceleration profile
    if direction_change_variance < 0.001: score += 2  # no wobble
    if jitter_magnitude < 0.5: score += 1       # too smooth
    if pause_count == 0: score += 1              # no hesitations
    
    return {
        "linearity_ratio": linearity_ratio,
        "speed_variance": speed_variance,
        "acceleration_variance": acceleration_variance,
        "direction_change_variance": direction_change_variance,
        "jitter_magnitude": jitter_magnitude,
        "pause_count": int(pause_count),
        "robotic_score": score
    }