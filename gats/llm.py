"""GATS 2.0 LLM Predictors - Compact implementation."""
from __future__ import annotations
import json, re, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class LLMResponse:
    text: str
    effects: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 0.5
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool: return len(self.effects) > 0

@dataclass
class LLMConfig:
    model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 100
    timeout: float = 30.0
    base_url: str = ""

def parse_json_list(text: str) -> list[str]:
    """Extract JSON list from text."""
    text = text.strip()
    if text.startswith('['):
        try: return json.loads(text[:text.find(']')+1])
        except: pass
    match = re.search(r'\[([^\]]*)\]', text)
    if match:
        try: return json.loads(f"[{match.group(1)}]")
        except: return re.findall(r'"([^"]+)"', match.group(1))
    return re.findall(r'"([^"]+)"', text) or re.findall(r'\b(\w+_(?:data|result|list|id|status))\b', text)

class LLMPredictor(ABC):
    @abstractmethod
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse: ...
    @abstractmethod
    def is_available(self) -> bool: ...

class MockPredictor(LLMPredictor):
    """Mock predictor for testing."""
    def is_available(self) -> bool: return True
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        # Heuristic effects based on action name
        if action.startswith("get_"): effects = frozenset([f"{action[4:]}_data"])
        elif action.startswith("search_"): effects = frozenset([f"{action[7:]}_results"])
        elif action.startswith("create_"): effects = frozenset([f"{action[7:]}_id"])
        else: effects = frozenset([f"{action}_result"])
        return LLMResponse(str(list(effects)), effects, 0.6, 1.0)

class OllamaPredictor(LLMPredictor):
    """Ollama local model predictor."""
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(base_url="http://localhost:11434")
        self._available: bool | None = None
    
    def is_available(self) -> bool:
        if self._available is not None: return self._available
        try:
            import requests
            self._available = requests.get(f"{self.config.base_url}/api/tags", timeout=2).ok
        except: self._available = False
        return self._available
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available(): return LLMResponse("", frozenset())
        import requests
        prompt = f'Act:{action} Has:{",".join(sorted(inventory)[:5])}\n→ Adds? JSON list:'
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.config.base_url}/api/generate", json={
                "model": self.config.model, "prompt": prompt, "stream": False,
                "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}
            }, timeout=self.config.timeout)
            lat = (time.perf_counter() - start) * 1000
            if not resp.ok: return LLMResponse("", latency_ms=lat)
            text = resp.json().get("response", "")
            effects = parse_json_list(text)
            return LLMResponse(text, frozenset(effects), 0.5 if effects else 0.0, lat, resp.json())
        except Exception as e:
            return LLMResponse(str(e), latency_ms=(time.perf_counter() - start) * 1000)

class VLLMPredictor(LLMPredictor):
    """vLLM server predictor."""
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(model="meta-llama/Llama-3.2-3B-Instruct", 
                                          base_url="http://localhost:8000/v1")
        self._available: bool | None = None
    
    def is_available(self) -> bool:
        if self._available is not None: return self._available
        try:
            import requests
            self._available = requests.get(f"{self.config.base_url}/models", timeout=2).ok
        except: self._available = False
        return self._available
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available(): return LLMResponse("", frozenset())
        import requests
        prompt = f'Act:{action} Has:{",".join(sorted(inventory)[:5])}\n→ Adds? JSON list:['
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.config.base_url}/completions", json={
                "model": self.config.model, "prompt": prompt, "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature, "stop": ["\n", "]"]
            }, timeout=self.config.timeout)
            lat = (time.perf_counter() - start) * 1000
            if not resp.ok: return LLMResponse("", latency_ms=lat)
            text = "[" + resp.json()["choices"][0]["text"] + "]"
            effects = parse_json_list(text)
            return LLMResponse(text, frozenset(effects), 0.5 if effects else 0.0, lat)
        except Exception as e:
            return LLMResponse(str(e), latency_ms=(time.perf_counter() - start) * 1000)

class CachedPredictor(LLMPredictor):
    """Caching wrapper."""
    def __init__(self, predictor: LLMPredictor, maxsize: int = 1000):
        self._pred, self._cache, self._max = predictor, {}, maxsize
        self._hits = self._misses = 0
    
    def is_available(self) -> bool: return self._pred.is_available()
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        key = (action, tuple(sorted(inventory)))
        if key in self._cache: self._hits += 1; return self._cache[key]
        self._misses += 1
        r = self._pred.predict_effects(action, inventory, goal)
        if len(self._cache) >= self._max: self._cache.pop(next(iter(self._cache)))
        self._cache[key] = r
        return r
    
    @property
    def hit_rate(self) -> float: return self._hits / max(1, self._hits + self._misses)

def create_predictor(backend: str = "auto", **kw) -> LLMPredictor:
    """Create predictor with auto-detection."""
    if backend == "ollama": return OllamaPredictor(LLMConfig(**kw) if kw else None)
    if backend == "vllm": return VLLMPredictor(LLMConfig(**kw) if kw else None)
    if backend == "mock": return MockPredictor()
    if backend == "auto":
        for cls in [OllamaPredictor, VLLMPredictor]:
            try:
                p = cls()
                if p.is_available(): return p
            except: pass
        return MockPredictor()
    raise ValueError(f"Unknown backend: {backend}")

class HFPredictor:
    """Stub HuggingFace predictor - placeholder for future implementation."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
    
    def is_available(self) -> bool:
        return False
    
    def predict_effects(self, action: str, inventory: frozenset) -> "PredictionResponse":
        return PredictionResponse(
            effects=frozenset(),
            confidence=0.0,
            success=False
        )