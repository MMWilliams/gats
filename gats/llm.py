"""GATS 2.0 Open Model Layer.

Unified interface for LLM-based predictions:
- Ollama (local)
- vLLM (server)
- HuggingFace Transformers (local)
- OpenAI-compatible APIs

Token-efficient prompting with structured output parsing.
"""
from __future__ import annotations
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import lru_cache


# =============================================================================
# Core Types
# =============================================================================

@dataclass
class LLMResponse:
    """Structured LLM response."""
    text: str
    effects: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 0.5
    latency_ms: float = 0.0
    tokens_used: int = 0
    raw: dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return len(self.effects) > 0


@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 100
    timeout: float = 30.0
    base_url: str = ""


# =============================================================================
# Abstract Base
# =============================================================================

class LLMPredictor(ABC):
    """Abstract base for LLM predictors."""
    
    @abstractmethod
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        """Predict effects of action given current state."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available."""
        pass


# =============================================================================
# Prompt Templates
# =============================================================================

EFFECT_PROMPT = """Action: {action}
State: {inventory}
Goal: {goal}

What items does this action add? Reply ONLY with a JSON list like ["item1", "item2"].
List:"""

EFFECT_PROMPT_MINIMAL = """Act:{action} Has:{inv}
→ Adds? JSON list:"""

REASONING_PROMPT = """Action: {action}
Current items: {inventory}
Goal items: {goal}

1. What does "{action}" likely do?
2. What items will it produce?
3. Confidence (0-1)?

Reply as JSON: {{"reasoning": "...", "effects": ["item1"], "confidence": 0.X}}"""


# =============================================================================
# Response Parsing
# =============================================================================

def parse_json_list(text: str) -> list[str]:
    """Extract JSON list from text, handling common LLM quirks."""
    # Try direct JSON parse
    text = text.strip()
    if text.startswith('['):
        try:
            end = text.find(']') + 1
            return json.loads(text[:end])
        except json.JSONDecodeError:
            pass
    
    # Find bracketed content
    match = re.search(r'\[([^\]]*)\]', text)
    if match:
        try:
            return json.loads(f"[{match.group(1)}]")
        except json.JSONDecodeError:
            # Try with quotes
            items = re.findall(r'"([^"]+)"', match.group(1))
            if items:
                return items
            # Try unquoted items
            items = [s.strip().strip('"\'') for s in match.group(1).split(',')]
            return [i for i in items if i]
    
    # Fallback: extract quoted strings
    items = re.findall(r'"([^"]+)"', text)
    if items:
        return items
    
    # Last resort: look for item-like patterns
    items = re.findall(r'\b(\w+_(?:data|result|list|id|status|done|complete))\b', text)
    return items


def parse_json_object(text: str) -> dict:
    """Extract JSON object from text."""
    text = text.strip()
    
    # Find JSON object
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    
    return {}


# =============================================================================
# Ollama Predictor
# =============================================================================

class OllamaPredictor(LLMPredictor):
    """Ollama local model predictor."""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(
            model="llama3.2",
            base_url="http://localhost:11434"
        )
        self._available: bool | None = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import requests
            resp = requests.get(f"{self.config.base_url}/api/tags", timeout=2)
            self._available = resp.ok
        except Exception:
            self._available = False
        
        return self._available
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(text="", effects=frozenset())
        
        import requests
        
        prompt = EFFECT_PROMPT_MINIMAL.format(
            action=action,
            inv=",".join(sorted(inventory)[:5])  # Truncate for token efficiency
        )
        
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=self.config.timeout
            )
            latency = (time.perf_counter() - start) * 1000
            
            if not resp.ok:
                return LLMResponse(text="", latency_ms=latency)
            
            data = resp.json()
            text = data.get("response", "")
            effects = parse_json_list(text)
            
            return LLMResponse(
                text=text,
                effects=frozenset(effects),
                confidence=0.5 if effects else 0.0,
                latency_ms=latency,
                raw=data
            )
            
        except Exception as e:
            return LLMResponse(text=str(e), latency_ms=(time.perf_counter() - start) * 1000)


# =============================================================================
# vLLM Predictor
# =============================================================================

class VLLMPredictor(LLMPredictor):
    """vLLM server predictor (OpenAI-compatible API)."""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(
            model="meta-llama/Llama-3.2-3B-Instruct",
            base_url="http://localhost:8000/v1"
        )
        self._available: bool | None = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import requests
            resp = requests.get(f"{self.config.base_url}/models", timeout=2)
            self._available = resp.ok
        except Exception:
            self._available = False
        
        return self._available
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(text="", effects=frozenset())
        
        import requests
        
        prompt = EFFECT_PROMPT_MINIMAL.format(
            action=action,
            inv=",".join(sorted(inventory)[:5])
        )
        
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.config.base_url}/completions",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stop": ["\n", "]"]
                },
                timeout=self.config.timeout
            )
            latency = (time.perf_counter() - start) * 1000
            
            if not resp.ok:
                return LLMResponse(text="", latency_ms=latency)
            
            data = resp.json()
            text = data.get("choices", [{}])[0].get("text", "")
            
            # vLLM may return partial list, complete it
            if not text.endswith(']'):
                text = "[" + text + "]"
            
            effects = parse_json_list(text)
            
            return LLMResponse(
                text=text,
                effects=frozenset(effects),
                confidence=0.5 if effects else 0.0,
                latency_ms=latency,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                raw=data
            )
            
        except Exception as e:
            return LLMResponse(text=str(e), latency_ms=(time.perf_counter() - start) * 1000)


# =============================================================================
# HuggingFace Transformers Predictor
# =============================================================================

class HFPredictor(LLMPredictor):
    """HuggingFace Transformers local predictor."""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self._model = None
        self._tokenizer = None
        self._available: bool | None = None
    
    def _load_model(self):
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self._available = True
        except Exception:
            self._available = False
    
    def is_available(self) -> bool:
        if self._available is None:
            self._load_model()
        return self._available
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(text="", effects=frozenset())
        
        import torch
        
        prompt = EFFECT_PROMPT_MINIMAL.format(
            action=action,
            inv=",".join(sorted(inventory)[:5])
        )
        
        start = time.perf_counter()
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text[len(prompt):]  # Remove prompt
            latency = (time.perf_counter() - start) * 1000
            
            effects = parse_json_list(text)
            
            return LLMResponse(
                text=text,
                effects=frozenset(effects),
                confidence=0.5 if effects else 0.0,
                latency_ms=latency,
                tokens_used=len(outputs[0])
            )
            
        except Exception as e:
            return LLMResponse(text=str(e), latency_ms=(time.perf_counter() - start) * 1000)


# =============================================================================
# OpenAI-Compatible Predictor (Generic)
# =============================================================================

class OpenAICompatiblePredictor(LLMPredictor):
    """Generic OpenAI-compatible API predictor."""
    
    def __init__(self, config: LLMConfig | None = None, api_key: str = ""):
        self.config = config or LLMConfig(
            model="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1"
        )
        self.api_key = api_key
        self._available: bool | None = None
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(text="", effects=frozenset())
        
        import requests
        
        prompt = EFFECT_PROMPT_MINIMAL.format(
            action=action,
            inv=",".join(sorted(inventory)[:5])
        )
        
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                },
                timeout=self.config.timeout
            )
            latency = (time.perf_counter() - start) * 1000
            
            if not resp.ok:
                return LLMResponse(text="", latency_ms=latency)
            
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            effects = parse_json_list(text)
            
            return LLMResponse(
                text=text,
                effects=frozenset(effects),
                confidence=0.5 if effects else 0.0,
                latency_ms=latency,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                raw=data
            )
            
        except Exception as e:
            return LLMResponse(text=str(e), latency_ms=(time.perf_counter() - start) * 1000)


# =============================================================================
# Auto-detect Predictor Factory
# =============================================================================

def create_predictor(backend: str = "auto", **kwargs) -> LLMPredictor:
    """
    Create LLM predictor with auto-detection.
    
    Args:
        backend: "ollama", "vllm", "hf", "openai", or "auto"
        **kwargs: Passed to predictor constructor
    
    Returns:
        Available LLM predictor
    """
    if backend == "ollama":
        return OllamaPredictor(**kwargs)
    elif backend == "vllm":
        return VLLMPredictor(**kwargs)
    elif backend == "hf":
        return HFPredictor(**kwargs)
    elif backend == "openai":
        return OpenAICompatiblePredictor(**kwargs)
    elif backend == "auto":
        # Try backends in order of preference
        for cls in [OllamaPredictor, VLLMPredictor, HFPredictor]:
            try:
                pred = cls(**kwargs)
                if pred.is_available():
                    return pred
            except Exception:
                continue
        
        # Return Ollama as default (will fail gracefully)
        return OllamaPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Caching Wrapper
# =============================================================================

class CachedPredictor(LLMPredictor):
    """Caching wrapper for any predictor."""
    
    def __init__(self, predictor: LLMPredictor, maxsize: int = 1000):
        self._predictor = predictor
        self._cache: dict[tuple, LLMResponse] = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
    
    def is_available(self) -> bool:
        return self._predictor.is_available()
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        key = (action, tuple(sorted(inventory)))
        
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        response = self._predictor.predict_effects(action, inventory, goal)
        
        # Simple LRU-ish eviction
        if len(self._cache) >= self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = response
        return response
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(1, total)
    
    def clear_cache(self):
        self._cache.clear()
        self._hits = self._misses = 0


# =============================================================================
# World Model Integration
# =============================================================================

def create_llm_world_model_fn(predictor: LLMPredictor) -> Callable:
    """
    Create Layer 3 prediction function from LLM predictor.
    
    Returns:
        Function (action, inventory) -> (effects, confidence)
    """
    def predict(action: str, inventory: frozenset[str]) -> tuple[frozenset[str], float]:
        response = predictor.predict_effects(action, inventory)
        return response.effects, response.confidence
    
    return predict


# =============================================================================
# Batch Prediction
# =============================================================================

def batch_predict(
    predictor: LLMPredictor,
    actions: list[tuple[str, frozenset[str]]],
    max_concurrent: int = 4
) -> list[LLMResponse]:
    """
    Batch prediction for multiple actions.
    
    Note: For true concurrency, use async version or threading.
    This is a simple sequential implementation.
    """
    return [predictor.predict_effects(action, inv) for action, inv in actions]


# =============================================================================
# Testing Utilities
# =============================================================================

class MockPredictor(LLMPredictor):
    """Mock predictor for testing."""
    
    def __init__(self, default_effects: frozenset[str] | None = None):
        self._effects = default_effects or frozenset(["mock_result"])
        self._calls: list[tuple[str, frozenset]] = []
    
    def is_available(self) -> bool:
        return True
    
    def predict_effects(self, action: str, inventory: frozenset[str], goal: frozenset[str] | None = None) -> LLMResponse:
        self._calls.append((action, inventory))
        
        # Heuristic effects based on action name
        if action.startswith("get_"):
            effects = frozenset([f"{action[4:]}_data"])
        elif action.startswith("search_"):
            effects = frozenset([f"{action[7:]}_results"])
        elif action.startswith("create_"):
            effects = frozenset([f"{action[7:]}_id"])
        else:
            effects = self._effects
        
        return LLMResponse(
            text=str(list(effects)),
            effects=effects,
            confidence=0.6,
            latency_ms=1.0
        )
    
    @property
    def call_count(self) -> int:
        return len(self._calls)