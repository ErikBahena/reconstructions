"""
Reconstructions: A Process-First Memory System

This package implements a human-like memory system where reconstruction
is performed by our algorithms, not delegated to an LLM.
"""

__version__ = "0.1.0"

# Core components
from .core import Fragment, Strand, Query
from .encoding import Experience, Context
from .store import FragmentStore
from .engine import ReconstructionEngine
from .consolidation import ConsolidationScheduler, ConsolidationConfig
from .health import MemoryHealthMonitor, MemoryHealthReport, format_health_report
from .metrics import RetrievalQualityTracker, QueryMetric, RetrievalQualitySnapshot
from .llm_client import LLMConfig, LLMResult, MemoryLLMClient, get_llm_client

# Claude Code integration
from . import claude_code

__all__ = [
    # Version
    "__version__",
    # Core
    "Fragment",
    "Strand",
    "Query",
    "Experience",
    "Context",
    "FragmentStore",
    "ReconstructionEngine",
    "ConsolidationScheduler",
    "ConsolidationConfig",
    # Health Monitoring
    "MemoryHealthMonitor",
    "MemoryHealthReport",
    "format_health_report",
    # Metrics
    "RetrievalQualityTracker",
    "QueryMetric",
    "RetrievalQualitySnapshot",
    # LLM Integration
    "LLMConfig",
    "LLMResult",
    "MemoryLLMClient",
    "get_llm_client",
    # Claude Code
    "claude_code",
]
