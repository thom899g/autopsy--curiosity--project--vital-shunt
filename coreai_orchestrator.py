"""
AI Orchestrator with Fault Tolerance
Manages AI model requests with retries, fallbacks, and state persistence
"""
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import backoff
import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientResponseError

from config.settings import settings, ModelProvider
from core.firebase_client import FirebaseClient
from core.models import AIRequest, AIResponse, RequestState


class AIError(Exception):
    """Base exception for AI-related errors"""
    pass


class RateLimitError(AIError):
    """Rate limit exceeded"""
    pass


class TimeoutError(AIError):
    """Request timeout"""
    pass


class InvalidRequestError(AIError):
    """Invalid request parameters"""
    pass


class AIOrchestrator:
    """
    Orchestrates AI requests with:
    - Retry logic with exponential backoff
    - Multi-provider fallback chain
    - Circuit breaker pattern
    - State persistence
    - Comprehensive telemetry
    """
    
    def __init__(self, firebase_client: Optional[FirebaseClient] = None):
        self.logger = structlog.get_logger(__name__)
        self.firebase_client = firebase_client or FirebaseClient()
        self._circuit_breakers: Dict[ModelProvider, dict] = {}
        self._session: Optional[aiohttp.ClientSession] = None