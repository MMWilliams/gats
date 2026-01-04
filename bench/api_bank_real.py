"""API-Bank Real Dataset Loader - Handles multiple data formats."""
from __future__ import annotations
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class APIBankTask:
    """Single API-Bank task."""
    query: str
    api_calls: List[Dict[str, Any]]
    level: int = 1
    task_id: str = ""
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"api_{hash(self.query) % 10000:04d}"

def parse_api_string(api_str: str) -> List[Dict[str, Any]]:
    """Parse API call string like '[GetWeather(city='NYC'), BookFlight(id=result)]'."""
    api_calls = []
    # Match patterns like ApiName(param='value', ...)
    pattern = r'(\w+)\(([^)]*)\)'
    for match in re.finditer(pattern, api_str):
        api_name = match.group(1)
        params_str = match.group(2)
        params = {}
        if params_str:
            # Parse params like key='value' or key=value
            param_pattern = r"(\w+)=['\"]?([^,'\")]+)['\"]?"
            for pm in re.finditer(param_pattern, params_str):
                params[pm.group(1)] = pm.group(2)
        api_calls.append({"api": api_name, "params": params})
    return api_calls

@dataclass
class RealAPIBankDataset:
    """Loader for real API-Bank dataset with synthetic fallback."""
    data_dir: str = "data/api_bank"
    tasks: List[APIBankTask] = field(default_factory=list)
    
    def load(self) -> "RealAPIBankDataset":
        """Load from JSON files or generate synthetic."""
        data_path = Path(self.data_dir)
        
        # Try level files (real data format)
        for level in [1, 2, 3]:
            for pattern in [f"level-{level}-api.json", f"level_{level}.json"]:
                fpath = data_path / pattern
                if fpath.exists():
                    self._load_file(fpath, level)
        
        # Try any other JSON files
        for fpath in data_path.glob("*.json"):
            if "level" not in fpath.name.lower() and "synthetic" not in fpath.name.lower():
                self._load_file(fpath)
        
        # Fallback to synthetic
        if not self.tasks:
            self._generate_synthetic()
        
        return self
    
    def _load_file(self, fpath: Path, default_level: int = 1) -> None:
        """Load tasks from a JSON file."""
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    self._parse_item(item, default_level)
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
    
    def _parse_item(self, item: Dict[str, Any], default_level: int = 1) -> None:
        """Parse a single item into APIBankTask - handles multiple formats."""
        # Get query from various possible fields
        query = (
            item.get("query") or 
            item.get("input") or 
            item.get("question") or
            item.get("instruction") or
            ""
        )
        if not query:
            return
        
        # Try to get API calls from various formats
        api_calls = []
        
        # Format 1: Direct api_list
        if "api_list" in item:
            api_calls = item["api_list"]
        # Format 2: api_calls field
        elif "api_calls" in item:
            api_calls = item["api_calls"]
        # Format 3: answer field (ToolBench style)
        elif "answer" in item:
            api_calls = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        # Format 4: expected_output as string (HuggingFace API-Bank format)
        elif "expected_output" in item:
            api_calls = parse_api_string(item["expected_output"])
        # Format 5: output field
        elif "output" in item:
            api_calls = parse_api_string(item["output"])
        
        if not api_calls:
            # Create a placeholder API call from the query
            api_calls = [{"api": "execute_task", "params": {"query": query[:50]}}]
        
        level = item.get("level", default_level)
        task_id = str(item.get("id", item.get("task_id", "")))
        
        self.tasks.append(APIBankTask(
            query=query,
            api_calls=api_calls if isinstance(api_calls, list) else [api_calls],
            level=int(level) if isinstance(level, (int, str)) and str(level).isdigit() else default_level,
            task_id=task_id
        ))
    
    def _generate_synthetic(self, n: int = 50) -> None:
        """Generate synthetic tasks for testing."""
        templates = [
            (1, "What is the weather in {city}?", [{"api": "get_weather", "params": {"city": "{city}"}}]),
            (1, "Get stock price for {symbol}", [{"api": "get_stock", "params": {"symbol": "{symbol}"}}]),
            (1, "Search for {query}", [{"api": "search", "params": {"q": "{query}"}}]),
            (2, "Book a flight from {src} to {dst}", [
                {"api": "search_flights", "params": {"from": "{src}", "to": "{dst}"}},
                {"api": "book_flight", "params": {"flight_id": "result"}}
            ]),
            (2, "Order {item} and track delivery", [
                {"api": "place_order", "params": {"item": "{item}"}},
                {"api": "track_order", "params": {"order_id": "result"}}
            ]),
            (3, "Plan trip to {city} with hotel and activities", [
                {"api": "search_flights", "params": {"to": "{city}"}},
                {"api": "search_hotels", "params": {"city": "{city}"}},
                {"api": "search_activities", "params": {"city": "{city}"}},
                {"api": "create_itinerary", "params": {}}
            ]),
        ]
        
        cities = ["NYC", "London", "Tokyo", "Paris", "Berlin"]
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        items = ["laptop", "phone", "headphones"]
        
        for i in range(n):
            level, template, apis = random.choice(templates)
            query = template.format(
                city=random.choice(cities),
                symbol=random.choice(symbols),
                src=random.choice(cities),
                dst=random.choice(cities),
                query=f"topic_{i}",
                item=random.choice(items)
            )
            self.tasks.append(APIBankTask(query=query, api_calls=apis, level=level))
    
    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        by_level: Dict[int, int] = {}
        for t in self.tasks:
            by_level[t.level] = by_level.get(t.level, 0) + 1
        return {"total": len(self.tasks), "by_level": by_level}
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)