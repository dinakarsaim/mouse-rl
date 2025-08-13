import numpy as np
import math
from scipy.ndimage import gaussian_filter1d

def analyze_mouse_path(points):
    if len(points) < 3:
        return {}
    
    # Unpack
    times = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    
    # Distances
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_path_length = np.sum(dists)
    direct_distance = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    linearity_ratio = direct_distance / total_path_length if total_path_length > 0 else 0
    
    # Speeds
    dt = np.diff(times) / 1000.0
    dt[dt == 0] = 1e-6
    speeds = dists / dt
    speed_cv = np.std(speeds) / (np.mean(speeds) + 1e-6)  # scale-free
    
    # Accelerations
    accelerations = np.diff(speeds) / dt[1:]
    accel_cv = np.std(accelerations) / (np.mean(np.abs(accelerations)) + 1e-6) if len(accelerations) > 0 else 0
    
    # Direction change
    angles = np.unwrap(np.arctan2(np.diff(ys), np.diff(xs)))
    dir_change = np.diff(angles)
    dir_std = np.std(dir_change) if len(dir_change) > 0 else 0
    
    # Jitter
    smooth_x = gaussian_filter1d(xs, sigma=2)
    smooth_y = gaussian_filter1d(ys, sigma=2)
    jitter_magnitude = np.mean(np.sqrt((xs - smooth_x)**2 + (ys - smooth_y)**2))
    
    # Pause count (count segments, not points)
    pause_mask = speeds < 0.05  # px/s
    pause_count = np.sum((~pause_mask[:-1] & pause_mask[1:]))  # count pause start
    
    # --- Robotic Score ---
    score = 0
    if linearity_ratio > 0.95: score += 2       # very straight
    if speed_cv < 0.15: score += 2              # too constant speed
    if accel_cv < 0.40: score += 2              # too constant accel profile
    if dir_std < 0.15: score += 2               # too little wobble
    if jitter_magnitude < 0.003: score += 1     # too smooth
    if pause_count == 0: score += 1             # no hesitations
    
    return {
        "linearity_ratio": linearity_ratio,
        "speed_cv": speed_cv,
        "accel_cv": accel_cv,
        "dir_std": dir_std,
        "jitter_magnitude": jitter_magnitude,
        "pause_count": int(pause_count),
        "robotic_score": score
    }