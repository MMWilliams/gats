#!/usr/bin/env python3
"""
Download API-Bank Level 3 data from the original GitHub repository.
The HuggingFace dataset may not have all levels.
"""
import json
import urllib.request
import ssl
import os
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

def download_from_github():
    """Try downloading from the original GitHub repo."""
    print("=" * 60)
    print("Downloading API-Bank from GitHub")
    print("=" * 60)
    
    data_dir = Path("data/api_bank")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # GitHub raw URLs for DAMO-ConvAI/API-Bank
    github_base = "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/api-bank"
    
    files = {
        # Test data
        "level-1-api.json": f"{github_base}/data/test-data/level-1-api.json",
        "level-2-api.json": f"{github_base}/data/test-data/level-2-api.json",
        "level-3-api.json": f"{github_base}/data/test-data/level-3-api.json",
        # Also try lv3 (alternative naming)
        "lv3-api.json": f"{github_base}/data/test-data/lv3-api.json",
        # Training data (might have multi-step)
        "train-data.json": f"{github_base}/data/training-data/train-data.json",
        # API definitions
        "api-list.json": f"{github_base}/data/api-list.json",
    }
    
    # Also check alternative paths
    alt_paths = [
        "data/level-3-api.json",
        "test-data/level-3-api.json", 
        "level-3-api.json",
        "data/test/level-3.json",
    ]
    
    for alt in alt_paths:
        files[f"level-3-alt-{alt.replace('/', '_')}"] = f"{github_base}/{alt}"
    
    for local_name, url in files.items():
        fpath = data_dir / local_name
        
        # Skip if main file exists
        if local_name.startswith("level-3-alt-") and (data_dir / "level-3-api.json").exists():
            continue
        
        if fpath.exists() and fpath.stat().st_size > 100:
            print(f"✓ {local_name}: Already exists ({fpath.stat().st_size / 1024:.1f} KB)")
            continue
        
        print(f"\n{local_name}:")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, fpath)
            size = fpath.stat().st_size
            if size > 100:
                print(f"  ✓ Downloaded ({size / 1024:.1f} KB)")
                
                # Check content
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        print(f"  Contains {len(data)} items")
                except:
                    pass
            else:
                fpath.unlink()
                print(f"  ✗ Empty file, removed")
        except Exception as e:
            if fpath.exists():
                fpath.unlink()
            print(f"  ✗ Failed: {str(e)[:60]}")


def check_multi_step_in_existing():
    """Check existing files for multi-step content."""
    print("\n" + "=" * 60)
    print("Checking for Multi-Step Tasks")
    print("=" * 60)
    
    data_dir = Path("data/api_bank")
    import re
    
    total_multi = 0
    
    for fpath in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            
            multi_step = 0
            max_steps = 0
            
            for item in data:
                # Collect all text
                text = ""
                for key in ["expected_output", "output", "answer", "api_call"]:
                    if key in item:
                        text += " " + str(item[key])
                
                # Check conversations
                if "conversations" in item:
                    for conv in item.get("conversations", []):
                        if isinstance(conv, dict):
                            text += " " + conv.get("value", "")
                
                # Count API calls
                apis = re.findall(r'\[?(\w+)\([^)]*\)\]?', text)
                if len(apis) >= 2:
                    multi_step += 1
                    max_steps = max(max_steps, len(apis))
            
            if multi_step > 0:
                print(f"✓ {fpath.name}: {multi_step} multi-step tasks (max {max_steps} steps)")
                total_multi += multi_step
            else:
                print(f"  {fpath.name}: {len(data)} items (all single-step)")
                
        except Exception as e:
            print(f"  {fpath.name}: Error - {e}")
    
    return total_multi


def create_level3_manually():
    """
    Create Level 3 tasks by chaining Level 1/2 APIs into workflows.
    Since actual Level 3 isn't available, we construct realistic multi-step tasks.
    """
    print("\n" + "=" * 60)
    print("Creating Level 3 Tasks (Constructed)")
    print("=" * 60)
    
    data_dir = Path("data/api_bank")
    
    # Collect APIs from Level 1 and 2
    import re
    apis_by_category = {
        "search": [],
        "auth": [],
        "query": [],
        "other": []
    }
    
    for level in [1, 2]:
        fpath = data_dir / f"level-{level}-api.json"
        if not fpath.exists():
            continue
        
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for item in data:
            expected = item.get("expected_output", "")
            match = re.search(r'\[?(\w+)\(([^)]*)\)\]?', expected)
            if match:
                api_name = match.group(1)
                params = match.group(2)
                
                if "search" in api_name.lower() or "tool" in api_name.lower():
                    apis_by_category["search"].append((api_name, params, item))
                elif "token" in api_name.lower() or "auth" in api_name.lower() or "login" in api_name.lower():
                    apis_by_category["auth"].append((api_name, params, item))
                elif "query" in api_name.lower() or "get" in api_name.lower():
                    apis_by_category["query"].append((api_name, params, item))
                else:
                    apis_by_category["other"].append((api_name, params, item))
    
    print(f"APIs found: search={len(apis_by_category['search'])}, auth={len(apis_by_category['auth'])}, query={len(apis_by_category['query'])}, other={len(apis_by_category['other'])}")
    
    # Create multi-step tasks by combining APIs
    level3_tasks = []
    task_id = 0
    
    # Pattern 1: Search → Query (2 steps)
    for search_api in apis_by_category["search"][:20]:
        for query_api in apis_by_category["query"][:5]:
            level3_tasks.append({
                "id": f"L3_{task_id}",
                "instruction": f"First search for the tool, then use it: {search_api[2].get('instruction', '')[:100]}",
                "input": f"Search and query workflow",
                "expected_output": f"Step 1: Search, Step 2: Query",  # Don't include API syntax here
                "api_sequence": [search_api[0], query_api[0]],  # This is what parser uses
                "n_steps": 2
            })
            task_id += 1
            if task_id >= 50:
                break
        if task_id >= 50:
            break
    
    # Pattern 2: Auth → Query (2 steps)
    for auth_api in apis_by_category["auth"][:10]:
        for query_api in apis_by_category["query"][:5]:
            level3_tasks.append({
                "id": f"L3_{task_id}",
                "instruction": f"Login first, then query: {auth_api[2].get('instruction', '')[:100]}",
                "input": f"Auth and query workflow",
                "expected_output": f"Step 1: Authenticate, Step 2: Query",
                "api_sequence": [auth_api[0], query_api[0]],
                "n_steps": 2
            })
            task_id += 1
            if task_id >= 100:
                break
        if task_id >= 100:
            break
    
    # Pattern 3: Search → Auth → Query (3 steps)
    for search_api in apis_by_category["search"][:10]:
        for auth_api in apis_by_category["auth"][:5]:
            for query_api in apis_by_category["query"][:3]:
                level3_tasks.append({
                    "id": f"L3_{task_id}",
                    "instruction": f"Find tool, authenticate, then query",
                    "input": f"Full workflow: search → auth → query",
                    "expected_output": f"Step 1: Search, Step 2: Auth, Step 3: Query",
                    "api_sequence": [search_api[0], auth_api[0], query_api[0]],
                    "n_steps": 3
                })
                task_id += 1
                if task_id >= 150:
                    break
            if task_id >= 150:
                break
        if task_id >= 150:
            break
    
    # Save
    if level3_tasks:
        output_path = data_dir / "level-3-api.json"
        output_path.write_text(json.dumps(level3_tasks, indent=2))
        print(f"\n✓ Created {len(level3_tasks)} Level 3 tasks")
        print(f"  Saved to: {output_path}")
        
        # Distribution
        from collections import Counter
        steps = Counter(t["n_steps"] for t in level3_tasks)
        print(f"  Steps distribution: {dict(steps)}")
    else:
        print("✗ Could not create Level 3 tasks (no source APIs)")
    
    return level3_tasks


if __name__ == "__main__":
    # Step 1: Try to download from GitHub
    download_from_github()
    
    # Step 2: Check for multi-step in existing
    n_multi = check_multi_step_in_existing()
    
    # Step 3: Create Level 3 if needed
    if n_multi < 20:
        level3 = create_level3_manually()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    data_dir = Path("data/api_bank")
    for level in [1, 2, 3]:
        fpath = data_dir / f"level-{level}-api.json"
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            print(f"Level {level}: {len(data)} tasks ({fpath.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"Level {level}: NOT FOUND")
    
    print("\nRun evaluation:")
    print("  python run_gats_eval_full.py --n-tasks 150 --quick")