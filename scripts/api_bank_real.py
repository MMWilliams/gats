"""API-Bank Real Dataset Loader - No external LLM dependencies."""
from __future__ import annotations
import json
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

@dataclass
class RealAPIBankDataset:
    """Loader for real API-Bank dataset with synthetic fallback."""
    data_dir: str = "data/api_bank"
    tasks: List[APIBankTask] = field(default_factory=list)
    
    def load(self) -> "RealAPIBankDataset":
        """Load from JSON files or generate synthetic."""
        data_path = Path(self.data_dir)
        
        # Try level files
        for level in [1, 2, 3]:
            for pattern in [f"level-{level}-api.json", f"level_{level}.json"]:
                fpath = data_path / pattern
                if fpath.exists():
                    self._load_file(fpath, level)
        
        # Try combined files
        for fpath in data_path.glob("*.json"):
            if "level" not in fpath.name.lower():
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
        except Exception:
            pass
    
    def _parse_item(self, item: Dict[str, Any], default_level: int = 1) -> None:
        """Parse a single item into APIBankTask."""
        query = item.get("query", item.get("input", item.get("question", "")))
        if not query:
            return
        
        api_calls = (
            item.get("api_list", []) or
            item.get("api_calls", []) or
            item.get("answer", []) or
            item.get("apis", [])
        )
        
        level = item.get("level", default_level)
        
        self.tasks.append(APIBankTask(
            query=query,
            api_calls=api_calls if isinstance(api_calls, list) else [api_calls],
            level=int(level) if isinstance(level, (int, str)) else default_level
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