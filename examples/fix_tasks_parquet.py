# run from anywhere with: uv run python fix_tasks_parquet.py
# (inside openpi dir so you use the openpi venv)
import pandas as pd
from huggingface_hub import HfApi
from io import BytesIO

df = pd.DataFrame([
    {"task_index": 0, "task": "grasp the object and place it in the box"}
])

buf = BytesIO()
df.to_parquet(buf, index=False)
buf.seek(0)

api = HfApi()
api.upload_file(
    path_or_fileobj=buf,
    path_in_repo="meta/tasks.parquet",
    repo_id="rudy8k/grasp_place",
    repo_type="dataset",
    commit_message="Fix tasks.parquet schema for lerobot pinned commit compatibility",
)
print("Uploaded. Schema:")
print(df.dtypes)