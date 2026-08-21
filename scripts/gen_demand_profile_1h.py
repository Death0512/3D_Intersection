#!/usr/bin/env python3
"""Generate a realistic 1-hour time-varying demand profile JSON covering empty, sparse, moderate, dense."""
import json
import random

def generate_natural_1hour_profile(output_path="demand_profile_1h.json", seed=42):
    rng = random.Random(seed)
    total_seconds = 3600.0  # 1 hour
    
    # Natural traffic flow pattern:
    # 1. Early / Off-peak lull: Empty -> Sparse (0 -> ~0.3-1.0) ~600s (10 min)
    # 2. Building up: Sparse -> Moderate (1.0 -> 3.0) ~800s (13.3 min)
    # 3. Peak / Rush hour surge: Moderate -> Dense -> Peak (3.0 -> 6.0-6.5 -> 4.5) ~1200s (20 min)
    # 4. Cooling down: Dense -> Moderate -> Sparse (4.5 -> 2.0 -> 0.8) ~700s (11.7 min)
    # 5. Night / Off-peak drop: Sparse -> Empty (0.5 -> 0.0) ~300s (5 min)

    # Base target curve keypoints (time_s, base_scale, label)
    keypoints = [
        (0.0, 0.0),       # empty
        (180.0, 0.2),     # nearly empty
        (450.0, 0.8),     # sparse start
        (750.0, 1.2),     # sparse
        (1100.0, 2.2),    # light-moderate
        (1450.0, 3.2),    # moderate
        (1750.0, 4.5),    # heavy moderate
        (2050.0, 6.0),    # dense (peak)
        (2350.0, 6.3),    # dense peak maximum (in 6.0-6.5 range)
        (2650.0, 5.0),    # dense easing
        (2950.0, 2.8),    # moderate cooldown
        (3250.0, 1.0),    # sparse return
        (3500.0, 0.3),    # tapering off
        (3600.0, 0.0),    # empty finish
    ]

    # Subdivide into continuous segments of 90s - 240s with natural jitter
    windows = []
    t = 0.0
    
    def get_interpolated_scale(time_sec):
        for i in range(len(keypoints) - 1):
            t0, s0 = keypoints[i]
            t1, s1 = keypoints[i+1]
            if t0 <= time_sec <= t1:
                ratio = (time_sec - t0) / (t1 - t0)
                return s0 + ratio * (s1 - s0)
        return keypoints[-1][1]

    while t < total_seconds:
        seg_dur = rng.uniform(90.0, 210.0)
        t_next = min(total_seconds, t + seg_dur)
        if total_seconds - t_next < 60.0:
            t_next = total_seconds
        
        mid_t = (t + t_next) / 2.0
        base_s = get_interpolated_scale(mid_t)
        
        # Add slight natural randomness (+-10% to 15%) while respecting boundaries
        if base_s > 0.1:
            jitter = rng.uniform(-0.15, 0.15) * base_s
            scale = max(0.0, min(6.5, base_s + jitter))
        else:
            scale = 0.0 if base_s == 0.0 else max(0.0, base_s + rng.uniform(-0.05, 0.05))
            
        scale = round(scale, 2)
        windows.append({
            "start": round(t, 1),
            "end": round(t_next, 1),
            "scale": scale
        })
        t = t_next

    # Ensure clean 0.0 start and end coverage
    windows[0]["start"] = 0.0
    windows[-1]["end"] = total_seconds

    with open(output_path, "w") as f:
        json.dump(windows, f, indent=2)
        
    print(f"Generated {len(windows)} intervals covering 0 -> {total_seconds}s to {output_path}")
    return windows

if __name__ == "__main__":
    generate_natural_1hour_profile()
