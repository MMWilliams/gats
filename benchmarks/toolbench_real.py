"""ToolBench Real Dataset Loader - No external LLM dependencies."""
from __future__ import annotations
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ToolBenchTask:
    """Single ToolBench task."""
    query: str
    tools: List[str]
    steps: List[Dict[str, Any]]
    split: str = "G1"
    task_id: str = ""
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"tb_{hash(self.query) % 10000:04d}"

@dataclass
class RealToolBenchDataset:
    """Loader for real ToolBench dataset with synthetic fallback."""
    data_dir: str = "data/toolbench"
    tasks: List[ToolBenchTask] = field(default_factory=list)
    
    def load(self) -> "RealToolBenchDataset":
        """Load from JSON files or generate synthetic."""
        data_path = Path(self.data_dir)
        
        # Try split files
        for split in ["G1_instruction", "G2_category", "G3_instruction"]:
            fpath = data_path / f"{split}.json"
            if fpath.exists():
                self._load_file(fpath, split.split("_")[0])
        
        # Try any JSON files
        for fpath in data_path.glob("*.json"):
            if not any(s in fpath.name for s in ["G1", "G2", "G3"]):
                self._load_file(fpath)
        
        # Fallback to synthetic
        if not self.tasks:
            self._generate_synthetic()
        
        return self
    
    def _load_file(self, fpath: Path, split: str = "G1") -> None:
        """Load tasks from a JSON file."""
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    self._parse_item(item, split)
        except Exception:
            pass
    
    def _parse_item(self, item: Dict[str, Any], split: str = "G1") -> None:
        """Parse a single item into ToolBenchTask."""
        query = item.get("query", item.get("input", item.get("instruction", "")))
        if not query:
            return
        
        tools = item.get("tools", item.get("tool_list", []))
        if isinstance(tools, str):
            tools = [tools]
        
        steps = item.get("answer", item.get("steps", item.get("api_calls", [])))
        if not isinstance(steps, list):
            steps = [steps] if steps else []
        
        self.tasks.append(ToolBenchTask(
            query=query,
            tools=tools,
            steps=steps,
            split=split
        ))
    
    def _generate_synthetic(self, n: int = 50) -> None:
        """Generate synthetic tasks for testing."""
        templates = [
            ("Get weather for {city}", ["weather_api"], [{"tool": "weather_api", "action": "get_current"}]),
            ("Search for {topic}", ["search_api"], [{"tool": "search_api", "action": "search"}]),
            ("Send email to {person}", ["email_api"], [{"tool": "email_api", "action": "send"}]),
            ("Book flight to {city}", ["flight_api", "booking_api"], [
                {"tool": "flight_api", "action": "search"},
                {"tool": "booking_api", "action": "reserve"}
            ]),
            ("Get stock info for {symbol}", ["finance_api"], [{"tool": "finance_api", "action": "get_quote"}]),
            ("Translate {text} to {lang}", ["translate_api"], [{"tool": "translate_api", "action": "translate"}]),
        ]
        
        cities = ["NYC", "London", "Tokyo", "Paris"]
        topics = ["AI", "climate", "sports", "tech"]
        people = ["Alice", "Bob", "Charlie"]
        symbols = ["AAPL", "GOOGL", "TSLA"]
        langs = ["Spanish", "French", "German"]
        
        for i in range(n):
            template, tools, steps = random.choice(templates)
            query = template.format(
                city=random.choice(cities),
                topic=random.choice(topics),
                person=random.choice(people),
                symbol=random.choice(symbols),
                text=f"text_{i}",
                lang=random.choice(langs)
            )
            split = random.choice(["G1", "G2", "G3"])
            self.tasks.append(ToolBenchTask(query=query, tools=tools, steps=steps, split=split))
    
    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        by_split: Dict[str, int] = {}
        total_steps = 0
        for t in self.tasks:
            by_split[t.split] = by_split.get(t.split, 0) + 1
            total_steps += len(t.steps)
        avg_steps = total_steps / len(self.tasks) if self.tasks else 0
        return {"total": len(self.tasks), "by_split": by_split, "avg_steps": avg_steps}
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)