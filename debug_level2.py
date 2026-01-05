#!/usr/bin/env python3
"""
Debug API-Bank Level 2 parsing issue.
"""
import json
import re
from pathlib import Path

def debug_level2():
    fpath = Path("data/api_bank/level-2-api.json")
    
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return
    
    print(f"Loading {fpath}...")
    data = json.loads(fpath.read_text(encoding="utf-8"))
    print(f"Found {len(data)} items\n")
    
    # Analyze first 5 items
    for i, item in enumerate(data[:5]):
        print(f"=" * 60)
        print(f"Item {i}")
        print(f"=" * 60)
        
        print(f"Keys: {list(item.keys())}")
        
        # Check all possible output fields
        for key in ["expected_output", "output", "answer", "response", "api_calls"]:
            if key in item:
                val = item[key]
                print(f"\n{key}:")
                print(f"  Type: {type(val)}")
                print(f"  Value: {str(val)[:200]}")
        
        # Try to find API patterns
        expected = item.get("expected_output", "") or item.get("output", "") or item.get("answer", "")
        if expected:
            # Various patterns to try
            patterns = [
                r'\[(\w+)\(([^)]*)\)\]',  # [ApiName(params)]
                r'(\w+)\(([^)]*)\)',       # ApiName(params)
                r'API-Request:\s*\[?(\w+)',  # API-Request: [ApiName
                r'"api":\s*"(\w+)"',        # JSON format
            ]
            
            print(f"\nPattern matching on expected_output:")
            for p in patterns:
                matches = re.findall(p, str(expected))
                if matches:
                    print(f"  {p[:30]}: {matches[:3]}")
        
        print()
    
    # Count how many have parseable API calls
    print("=" * 60)
    print("PARSING ANALYSIS")
    print("=" * 60)
    
    parseable = 0
    multi_step = 0
    
    for item in data:
        expected = str(item.get("expected_output", "") or item.get("output", "") or item.get("answer", ""))
        
        # Try to find any API pattern
        apis = re.findall(r'(\w+)\([^)]*\)', expected)
        
        if apis:
            parseable += 1
            if len(apis) > 1:
                multi_step += 1
    
    print(f"Total items: {len(data)}")
    print(f"Parseable (has API calls): {parseable}")
    print(f"Multi-step (2+ APIs): {multi_step}")
    
    if parseable == 0:
        print("\n[ISSUE] No API patterns found!")
        print("Checking raw content of first item...")
        print(json.dumps(data[0], indent=2)[:1000])


if __name__ == "__main__":
    debug_level2()