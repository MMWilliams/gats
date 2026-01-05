#!/usr/bin/env python3
"""
Check API-Bank and ToolBench data availability and format.
Run this first to verify data is present.
"""
import json
from pathlib import Path

def check_data():
    print("=" * 60)
    print("Checking benchmark data...")
    print("=" * 60)
    
    data_dir = Path("data")
    
    # Check API-Bank
    print("\n[API-Bank]")
    api_bank_dir = data_dir / "api_bank"
    
    for level in [1, 2, 3]:
        fpath = api_bank_dir / f"level-{level}-api.json"
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                print(f"  Level {level}: {len(data)} items ({fpath.stat().st_size / 1024:.1f} KB)")
                
                # Show example
                if data:
                    item = data[0]
                    print(f"    Keys: {list(item.keys())}")
                    
                    # Show expected output format
                    expected = item.get("expected_output", item.get("output", ""))
                    if expected:
                        print(f"    Expected output sample: {expected[:100]}...")
                    
            except Exception as e:
                print(f"  Level {level}: ERROR - {e}")
        else:
            print(f"  Level {level}: NOT FOUND at {fpath}")
    
    # Check ToolBench
    print("\n[ToolBench]")
    toolbench_dir = data_dir / "toolbench"
    
    if toolbench_dir.exists():
        files = list(toolbench_dir.glob("*.json"))
        if files:
            for f in files[:3]:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    print(f"  {f.name}: {len(data) if isinstance(data, list) else 'dict'} items")
                except Exception as e:
                    print(f"  {f.name}: ERROR - {e}")
        else:
            print("  No JSON files found")
    else:
        print(f"  NOT FOUND at {toolbench_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    api_bank_found = any((api_bank_dir / f"level-{l}-api.json").exists() for l in [1, 2, 3])
    toolbench_found = toolbench_dir.exists() and list(toolbench_dir.glob("*.json"))
    
    if api_bank_found:
        print("✓ API-Bank data found")
    else:
        print("✗ API-Bank data missing - run: python scripts/download_datasets.py")
    
    if toolbench_found:
        print("✓ ToolBench data found")
    else:
        print("✗ ToolBench data missing (optional)")
    
    print("\nTo run evaluation:")
    print("  python run_gats_eval_full.py --n-tasks 100 --quick")


if __name__ == "__main__":
    check_data()