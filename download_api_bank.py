#!/usr/bin/env python3
"""
Download API-Bank Level 1/2 and the correct Level 3 files (no more 404).
"""
import json
import shutil
import ssl
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path

# Disable SSL verification for corporate networks
ssl._create_default_https_context = ssl._create_unverified_context

REVISION = "main"
# Pin to a specific commit if you want reproducibility:
# REVISION = "12e8158b7628c168f07e8f31fbbe3445e99f44cf"

BASE = f"https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/{REVISION}/test-data"

def download(url: str, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest.stat().st_size > 0
    except HTTPError as e:
        print(f"  HTTP {e.code} for {url}")
    except URLError as e:
        print(f"  URL error for {url}: {e}")
    except Exception as e:
        print(f"  Failed for {url}: {e}")
    return False

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def download_api_bank():
    print("=" * 60)
    print("Downloading API-Bank Data")
    print("=" * 60)

    data_dir = Path("data/api_bank")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Level 1/2 files (as before)
    files = {
        "level-1-api.json": f"{BASE}/level-1-api.json",
        "level-2-api.json": f"{BASE}/level-2-api.json",

        # Level 3 options (these exist; level-3-api.json does NOT)
        "level-3.json": f"{BASE}/level-3.json",  # full traces
        "level-3-batch-inf.json": f"{BASE}/level-3-batch-inf.json",  # step-by-step API-Request generation
        # "level-3-batch-inf-icl.json": f"{BASE}/level-3-batch-inf-icl.json",  # optional ICL variant
    }

    for fname, url in files.items():
        fpath = data_dir / fname
        print(f"\n{fname}:")
        if fpath.exists():
            print(f"  Already exists ({fpath.stat().st_size/1024:.1f} KB)")
            continue

        print(f"  Downloading from {url} ...")
        ok = download(url, fpath)
        if ok:
            print(f"  Downloaded ({fpath.stat().st_size/1024:.1f} KB)")
        else:
            print("  Download failed.")

    # Optional: if your code expects level-3-api.json, alias the batch-inf file
    batch_inf = data_dir / "level-3-batch-inf.json"
    alias = data_dir / "level-3-api.json"
    if batch_inf.exists() and not alias.exists():
        shutil.copyfile(batch_inf, alias)
        print("\nAliased level-3-batch-inf.json -> level-3-api.json (for pipeline compatibility)")

    # Quick sanity checks / summaries
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Level 1/2: straightforward counts
    for lvl in [1, 2]:
        p = data_dir / f"level-{lvl}-api.json"
        if p.exists():
            data = load_json(p)
            print(f"Level {lvl}: {len(data)} items")
        else:
            print(f"Level {lvl}: NOT FOUND")

    # Level 3 trace file
    p3 = data_dir / "level-3.json"
    if p3.exists():
        data = load_json(p3)
        # each item has requirement/response/apis
        print(f"Level 3 (traces): {len(data)} scenarios")

    # Level 3 batch inference file
    p3b = data_dir / "level-3-batch-inf.json"
    if p3b.exists():
        data = load_json(p3b)
        sample_ids = {x.get("sample_id") for x in data if "sample_id" in x}
        print(f"Level 3 (batch-inf): {len(data)} steps across {len(sample_ids)} scenarios")

if __name__ == "__main__":
    download_api_bank()
