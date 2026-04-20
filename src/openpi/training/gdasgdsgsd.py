uv run python3 << 'EOF'
import json, shutil, pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download

SNAP = Path("/home_local/rudra_1/.cache/huggingface/hub/datasets--rudy8k--grasp_place/snapshots/4726d2b0b41c28fd06e60bc738e1e99514e86785")
OUT  = Path("/home_local/rudra_1/rudra/data/grasp_place_v21")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# ── 1. info.json ──────────────────────────────────────────────────────────────
info = json.loads((SNAP / "meta/info.json").read_text())
fps  = info["fps"]
info["codebase_version"] = "v2.1"
info["data_path"]        = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
info["video_path"]       = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
info["tolerance_s"]      = round(1 / fps, 6)
# remove v3.0-only keys
for k in ["data_files_size_in_mb", "video_files_size_in_mb"]:
    info.pop(k, None)
# fix List -> Sequence in features
def fix_features(features):
    for key, val in features.items():
        if isinstance(val, dict):
            if val.get("_type") == "List":
                val["_type"] = "Sequence"
            fix_features(val)
fix_features(info["features"])
(OUT / "meta").mkdir()
(OUT / "meta/info.json").write_text(json.dumps(info, indent=2))
print("✓ info.json written")

# ── 2. episodes.jsonl + tasks.jsonl from v3.0 meta ───────────────────────────
# v3.0 episodes are in meta/episodes/chunk-000/file_000.parquet
ep_parquets = sorted((SNAP / "meta/episodes").rglob("*.parquet"))
ep_df = pd.concat([pd.read_parquet(p) for p in ep_parquets]).sort_values("episode_index").reset_index(drop=True)

# tasks.jsonl
tasks_src = SNAP / "meta/tasks.parquet"
if tasks_src.exists():
    tasks_df = pd.read_parquet(tasks_src)
    with open(OUT / "meta/tasks.jsonl", "w") as f:
        for _, row in tasks_df.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": row["task"]}) + "\n")
    print("✓ tasks.jsonl written")
else:
    # fallback: tasks.jsonl directly
    shutil.copy(SNAP / "meta/tasks.jsonl", OUT / "meta/tasks.jsonl")
    print("✓ tasks.jsonl copied")

# episodes.jsonl — map task_index back from task string
tasks_map = {}
with open(OUT / "meta/tasks.jsonl") as f:
    for line in f:
        t = json.loads(line)
        tasks_map[t["task"]] = t["task_index"]

with open(OUT / "meta/episodes.jsonl", "w") as f:
    for _, row in ep_df.iterrows():
        tasks_list = row["tasks"] if isinstance(row["tasks"], list) else [row["tasks"]]
        task_indices = [tasks_map[t] for t in tasks_list]
        f.write(json.dumps({
            "episode_index": int(row["episode_index"]),
            "tasks": tasks_list,
            "task_index": task_indices[0],
            "length": int(row["length"]),
        }) + "\n")
print(f"✓ episodes.jsonl written ({len(ep_df)} episodes)")

# ── 3. episodes_stats.jsonl ───────────────────────────────────────────────────
import numpy as np

stat_cols = [c for c in ep_df.columns if "/" in c and any(
    c.endswith(s) for s in ["/min", "/max", "/mean", "/std", "/count"])]

with open(OUT / "meta/episodes_stats.jsonl", "w") as f:
    for _, row in ep_df.iterrows():
        stats = {}
        for col in stat_cols:
            parts = col.split("/")
            feat  = "/".join(parts[:-1])
            stat  = parts[-1]
            val   = row[col]
            if feat not in stats:
                stats[feat] = {}
            stats[feat][stat] = val.tolist() if hasattr(val, "tolist") else val
        f.write(json.dumps({"episode_index": int(row["episode_index"]), "stats": stats}) + "\n")
print("✓ episodes_stats.jsonl written")

# stats.json — copy if exists
stats_src = SNAP / "meta/stats.json"
if stats_src.exists():
    shutil.copy(stats_src, OUT / "meta/stats.json")
    print("✓ stats.json copied")

# ── 4. Data parquet files: rename file_NNN.parquet → episode_NNNNNN.parquet ──
data_out = OUT / "data/chunk-000"
data_out.mkdir(parents=True)

# v3.0: data/chunk-000/file_000.parquet (all episodes merged)
# v2.1: data/chunk-000/episode_000000.parquet (one per episode)
data_parquets = sorted((SNAP / "data").rglob("*.parquet"))
all_data = pd.concat([pd.read_parquet(p) for p in data_parquets]).sort_values(
    ["episode_index", "index"]).reset_index(drop=True)

for ep_idx, group in all_data.groupby("episode_index"):
    out_path = data_out / f"episode_{ep_idx:06d}.parquet"
    group.reset_index(drop=True).to_parquet(out_path, index=False)

print(f"✓ data parquets written ({all_data['episode_index'].nunique()} episodes)")

# ── 5. Videos: reorganize from videos/CAMERA/chunk-000/file_000.mp4
#              to           videos/chunk-000/CAMERA/episode_000000.mp4 ─────────
# v3.0 layout: videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4
# v2.1 layout: videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
#
# Since each v3.0 file may contain MULTIPLE episodes concatenated, we need
# to split them. Check if ffmpeg split is needed or if it's 1 ep per file:

import subprocess

videos_src = SNAP / "videos"
ep_lengths = dict(zip(ep_df["episode_index"], ep_df["length"]))

for cam_dir in sorted(videos_src.iterdir()):
    if not cam_dir.is_dir():
        continue
    cam_key = cam_dir.name  # e.g. observation.images.cam_high

    # Collect all mp4 files for this camera sorted
    mp4_files = sorted(cam_dir.rglob("*.mp4"))

    # Build per-episode assignment from ep_df
    # ep_df has columns: videos/{cam_key}/chunk_index, videos/{cam_key}/file_index
    # videos/{cam_key}/from_timestamp, videos/{cam_key}/to_timestamp
    ci_col  = f"videos/{cam_key}/chunk_index"
    fi_col  = f"videos/{cam_key}/file_index"
    ft_col  = f"videos/{cam_key}/from_timestamp"
    tt_col  = f"videos/{cam_key}/to_timestamp"

    if ci_col not in ep_df.columns:
        print(f"  ⚠ No metadata for {cam_key}, skipping")
        continue

    for _, row in ep_df.iterrows():
        ep_idx   = int(row["episode_index"])
        ci       = int(row[ci_col])
        fi       = int(row[fi_col])
        from_ts  = float(row[ft_col])
        to_ts    = float(row[tt_col])

        src_mp4  = cam_dir / f"chunk-{ci:03d}" / f"file_{fi:03d}.mp4"
        dst_dir  = OUT / "videos" / "chunk-000" / cam_key
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_mp4  = dst_dir / f"episode_{ep_idx:06d}.mp4"

        duration = to_ts - from_ts
        cmd = [
            "ffmpeg", "-y", "-ss", str(from_ts), "-i", str(src_mp4),
            "-t", str(duration), "-c", "copy", str(dst_mp4)
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  ✗ ffmpeg failed for ep {ep_idx} cam {cam_key}")
        else:
            print(f"  ✓ ep {ep_idx:02d} {cam_key}", end="\r")

print("\n✓ videos converted")
print(f"\nDone! Dataset written to {OUT}")
print("Now set in config.py:")
print(f'  repo_id="{OUT}"')
EOF