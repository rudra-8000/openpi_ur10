uv run python3 << 'EOF'
import subprocess, pandas as pd
from pathlib import Path

SNAP = Path("/home_local/rudra_1/.cache/huggingface/hub/datasets--rudy8k--grasp_place/snapshots/4726d2b0b41c28fd06e60bc738e1e99514e86785")
OUT  = Path("/home_local/rudra_1/rudra/data/grasp_place_v21")

ep_files = sorted((SNAP / "meta/episodes").rglob("*.parquet"))
ep_df = pd.concat([pd.read_parquet(p) for p in ep_files]).sort_values("episode_index").reset_index(drop=True)

for cam_key in ["observation.images.cam_high", "observation.images.cam_right_wrist"]:
    dst_dir = OUT / "videos/chunk-000" / cam_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    for _, row in ep_df.iterrows():
        ep_idx  = int(row["episode_index"])
        ci      = int(row[f"videos/{cam_key}/chunk_index"])
        fi      = int(row[f"videos/{cam_key}/file_index"])
        from_ts = float(row[f"videos/{cam_key}/from_timestamp"])
        to_ts   = float(row[f"videos/{cam_key}/to_timestamp"])

        # dash not underscore
        src = SNAP / "videos" / cam_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        dst = dst_dir / f"episode_{ep_idx:06d}.mp4"

        r = subprocess.run([
            "ffmpeg", "-y", "-ss", f"{from_ts:.6f}", "-i", str(src),
            "-t", f"{to_ts - from_ts:.6f}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", str(dst)
        ], capture_output=True)
        if r.returncode != 0:
            print(f"  ✗ ep{ep_idx:02d} {cam_key}: {r.stderr.decode()[-200:]}")
        else:
            print(f"  ✓ ep{ep_idx:02d} {cam_key}", end="\r")

print("\n✓ All videos done!")
print(f'\nSet in config.py: repo_id="{OUT}"')
EOF