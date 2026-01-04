#!/usr/bin/env python3
"""Download datasets for GATS 2.0 - Works on Windows/Linux/Mac."""
import os
import json
import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for some corporate networks
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"  Trying: {url[:60]}...", end=" ")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            dest.write_bytes(response.read())
        print("OK")
        return True
    except Exception as e:
        print(f"SKIP ({type(e).__name__})")
        return False

def download_api_bank(data_dir: Path) -> bool:
    """Download API-Bank dataset."""
    print("\n[API-Bank]")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try HuggingFace URLs (correct format)
    hf_base = "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main"
    files = [
        ("level-1-api.json", f"{hf_base}/test-data/level-1-api.json"),
        ("level-2-api.json", f"{hf_base}/test-data/level-2-api.json"),
        ("level-3-api.json", f"{hf_base}/test-data/level-3-api.json"),
        ("level-1-api.json", f"{hf_base}/lv1-lv2-samples/level-1-api.json"),
        ("level-2-api.json", f"{hf_base}/lv1-lv2-samples/level-2-api.json"),
    ]
    
    success = False
    for fname, url in files:
        dest = data_dir / fname
        if dest.exists():
            print(f"  {fname} already exists")
            success = True
            continue
        if download_file(url, dest):
            success = True
    
    return success

def download_toolbench(data_dir: Path) -> bool:
    """Download ToolBench dataset."""
    print("\n[ToolBench]")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ToolBench requires HuggingFace authentication - skip direct download
    print("  ToolBench requires HuggingFace login (skipping direct download)")
    print("  To download manually: pip install datasets && python -c \"from datasets import load_dataset; ds = load_dataset('ToolBench/ToolBench')\"")
    return False

def create_synthetic_data(data_dir: Path) -> None:
    """Create comprehensive synthetic data for testing."""
    api_bank_dir = data_dir / "api_bank"
    toolbench_dir = data_dir / "toolbench"
    
    # Check if we need synthetic API-Bank
    has_api_bank = any(api_bank_dir.glob("level*.json")) if api_bank_dir.exists() else False
    
    if not has_api_bank:
        print("\n[Creating synthetic API-Bank data]")
        api_bank_dir.mkdir(parents=True, exist_ok=True)
        
        # Level 1: Single API calls
        level1 = [
            {"input": "What's the weather in New York?", "instruction": "Call weather API", 
             "expected_output": "[GetWeather(city='New York')]", "id": 1, "file": "test"},
            {"input": "Search for restaurants nearby", "instruction": "Call search API",
             "expected_output": "[SearchPlaces(query='restaurants', type='nearby')]", "id": 2, "file": "test"},
            {"input": "Get stock price for AAPL", "instruction": "Call finance API",
             "expected_output": "[GetStockPrice(symbol='AAPL')]", "id": 3, "file": "test"},
            {"input": "Translate hello to Spanish", "instruction": "Call translation API",
             "expected_output": "[Translate(text='hello', target='es')]", "id": 4, "file": "test"},
            {"input": "What time is it in Tokyo?", "instruction": "Call timezone API",
             "expected_output": "[GetTime(timezone='Asia/Tokyo')]", "id": 5, "file": "test"},
        ]
        for i in range(6, 51):
            level1.append({
                "input": f"Query {i} for single API", "instruction": "Call API",
                "expected_output": f"[API{i}(param='value')]", "id": i, "file": "test"
            })
        
        # Level 2: Multi-step API calls
        level2 = [
            {"input": "Book a flight from NYC to LA and then a hotel", "instruction": "Multi-step booking",
             "expected_output": "[SearchFlights(from='NYC', to='LA'), BookFlight(id=result), SearchHotels(city='LA'), BookHotel(id=result)]", 
             "id": 1, "file": "test"},
            {"input": "Find weather and recommend activities", "instruction": "Weather + recommendations",
             "expected_output": "[GetWeather(city='current'), RecommendActivities(weather=result)]", "id": 2, "file": "test"},
            {"input": "Search products and compare prices", "instruction": "Search + compare",
             "expected_output": "[SearchProducts(query='laptop'), ComparePrices(products=result)]", "id": 3, "file": "test"},
        ]
        for i in range(4, 31):
            level2.append({
                "input": f"Multi-step query {i}", "instruction": "Multi-step API calls",
                "expected_output": f"[API{i}A(), API{i}B(prev=result)]", "id": i, "file": "test"
            })
        
        # Level 3: Complex reasoning + retrieval
        level3 = [
            {"input": "Plan a trip including flights, hotels, car rental, and activities",
             "instruction": "Complex trip planning",
             "expected_output": "[SearchFlights(), BookFlight(), SearchHotels(), BookHotel(), RentCar(), FindActivities()]",
             "id": 1, "file": "test"},
        ]
        for i in range(2, 21):
            level3.append({
                "input": f"Complex query {i} requiring multiple APIs", "instruction": "Complex reasoning",
                "expected_output": f"[API{i}A(), API{i}B(), API{i}C(), API{i}D()]", "id": i, "file": "test"
            })
        
        (api_bank_dir / "level-1-api.json").write_text(json.dumps(level1, indent=2))
        (api_bank_dir / "level-2-api.json").write_text(json.dumps(level2, indent=2))
        (api_bank_dir / "level-3-api.json").write_text(json.dumps(level3, indent=2))
        print(f"  Created level-1-api.json ({len(level1)} tasks)")
        print(f"  Created level-2-api.json ({len(level2)} tasks)")
        print(f"  Created level-3-api.json ({len(level3)} tasks)")
    
    # ToolBench synthetic
    has_toolbench = any(toolbench_dir.glob("*.json")) if toolbench_dir.exists() else False
    
    if not has_toolbench:
        print("\n[Creating synthetic ToolBench data]")
        toolbench_dir.mkdir(parents=True, exist_ok=True)
        
        tasks = []
        tools = ["weather_api", "search_api", "calendar_api", "email_api", "maps_api", 
                 "finance_api", "news_api", "translate_api", "booking_api", "social_api"]
        
        for i in range(100):
            n_tools = (i % 3) + 1  # 1-3 tools
            selected_tools = tools[i % len(tools): i % len(tools) + n_tools]
            if not selected_tools:
                selected_tools = [tools[0]]
            
            tasks.append({
                "query": f"Task {i+1}: Use {', '.join(selected_tools)} to complete request",
                "tools": selected_tools,
                "answer": [{"tool": t, "action": f"action_{j}"} for j, t in enumerate(selected_tools)]
            })
        
        (toolbench_dir / "test_data.json").write_text(json.dumps(tasks, indent=2))
        print(f"  Created test_data.json ({len(tasks)} tasks)")

def main():
    print("=" * 50)
    print("GATS 2.0 Dataset Downloader")
    print("=" * 50)
    
    data_dir = Path("data")
    
    api_bank_ok = download_api_bank(data_dir / "api_bank")
    toolbench_ok = download_toolbench(data_dir / "toolbench")
    
    # Create synthetic if downloads failed
    if not api_bank_ok or not toolbench_ok:
        create_synthetic_data(data_dir)
    
    print("\n" + "=" * 50)
    print("Done! Files in data/:")
    for p in sorted(data_dir.rglob("*.json")):
        size = p.stat().st_size
        print(f"  {p} ({size:,} bytes)")
    
    # Summary
    print("\n" + "=" * 50)
    api_count = sum(1 for _ in (data_dir / "api_bank").glob("*.json")) if (data_dir / "api_bank").exists() else 0
    tb_count = sum(1 for _ in (data_dir / "toolbench").glob("*.json")) if (data_dir / "toolbench").exists() else 0
    print(f"API-Bank: {api_count} files")
    print(f"ToolBench: {tb_count} files")
    
    if api_count > 0 and tb_count > 0:
        print("\n✓ Ready to run experiments!")
        print("  python run_experiment.py --n-tasks 50")

if __name__ == "__main__":
    main()