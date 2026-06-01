"""Download MiniLM into models/all-MiniLM-L6-v2 (uses HF mirror when HF_ENDPOINT unset)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    root = _ensure_src_on_path()
    from depression_ml import config
    from depression_ml.embedding_baseline import configure_hf_download

    configure_hf_download()
    endpoint = os.environ.get("HF_ENDPOINT", "(default huggingface.co)")
    out_dir = Path(config.EMBEDDING_MODEL_LOCAL_DIR or (root / "models" / "all-MiniLM-L6-v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"HF_ENDPOINT: {endpoint}", flush=True)
    print(f"Target folder: {out_dir}", flush=True)
    print("Downloading (about 90MB). If this fails, set mirror first:", flush=True)
    print('  $env:HF_ENDPOINT = "https://hf-mirror.com"', flush=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...", flush=True)
        os.system(f'"{sys.executable}" -m pip install huggingface_hub')
        from huggingface_hub import snapshot_download

    repo = config.EMBEDDING_MODEL_NAME
    if repo.startswith("sentence-transformers/"):
        repo_id = repo
    else:
        repo_id = config.EMBEDDING_MODEL_NAME

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Model saved under: {path}", flush=True)
    print("Run task2:", flush=True)
    print("  python scripts/run_task2.py", flush=True)


if __name__ == "__main__":
    main()
