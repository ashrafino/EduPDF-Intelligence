"""
Robust error handling system with exponential backoff, circuit breaker pattern,
and comprehensive error recovery mechanisms.
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from functools import wraps
import json


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    NETWORK = "network"
    CONTENT = "content"
    PROCESSING = "processing"
    STORAGE = "storage"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Structured error information for logging and analysis."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    traceback_info: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'exception_type': self.exception_type,
            'traceback_info': self.traceback_info,
            'context': self.context,
            'retry_count': self.retry_count,
            'source_url': self.source_url,
            'source_name': self.source_name
        }


class ExponentialBackoff:
    """
    Implements exponential backoff strategy for retry operations.
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize exponential backoff configuration.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for each retry
            jitter: Whether to add random jitter to prevent thundering herd
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate delay for a given retry attempt.
        
        Args:
            retry_count: Number of previous retry attempts
            
        Returns:
            Delay in seconds
        """
        delay = self.initial_delay * (self.backoff_factor ** retry_count)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            # Add up to 25% jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay
    
    async def wait(self, retry_count: int) -> None:
        """
        Wait for the calculated delay period.
        
        Args:
            retry_count: Number of previous retry attempts
        """
        delay = self.calculate_delay(retry_count)
        await asyncio.sleep(delay)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3  # For half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling problematic sources.
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            name: Name for identification and logging
        """
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorClassifier:
    """
    Classifies errors into categories and severity levels for appropriate handling.
    """
    
    def __init__(self):
        """Initialize error classifier with predefined rules."""
        self.classification_rules = {
            # Network errors
            'ConnectionError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            'TimeoutError': (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            'ConnectTimeout': (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            'ReadTimeout': (ErrorCategory.TIMEOUT, ErrorSeverity.LOW),
            'DNSError': (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            'SSLError': (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            
            # HTTP errors
            'ClientResponseError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            'ServerDisconnectedError': (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            
            # Content errors
            'UnicodeDecodeError': (ErrorCategory.CONTENT, ErrorSeverity.LOW),
            'JSONDecodeError': (ErrorCategory.CONTENT, ErrorSeverity.LOW),
            'ParseError': (ErrorCategory.CONTENT, ErrorSeverity.LOW),
            
            # Processing errors
            'MemoryError': (ErrorCategory.PROCESSING, ErrorSeverity.CRITICAL),
            'ProcessingError': (ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM),
            
            # Storage errors
            'PermissionError': (ErrorCategory.STORAGE, ErrorSeverity.HIGH),
            'FileNotFoundError': (ErrorCategory.STORAGE, ErrorSeverity.MEDIUM),
            'OSError': (ErrorCategory.STORAGE, ErrorSeverity.MEDIUM),
            
            # Rate limiting
            'TooManyRequests': (ErrorCategory.RATE_LIMIT, ErrorSeverity.LOW),
            'RateLimitError': (ErrorCategory.RATE_LIMIT, ErrorSeverity.LOW),
        }
    
    def classify_error(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an exception into category and severity.
        
        Args:
            exception: Exception to classify
            
        Returns:
            Tuple of (category, severity)
        """
        exception_name = type(exception).__name__
        
        # Check direct mapping
        if exception_name in self.classification_rules:
            return self.classification_rules[exception_name]
        
        # Check HTTP status codes for aiohttp errors
        if hasattr(exception, 'status'):
            status = exception.status
            if 400 <= status < 500:
                if status == 429:  # Too Many Requests
                    return ErrorCategory.RATE_LIMIT, ErrorSeverity.LOW
                elif status in [401, 403]:  # Authentication/Authorization
                    return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
                else:
                    return ErrorCategory.CONTENT, ErrorSeverity.MEDIUM
            elif 500 <= status < 600:
                return ErrorCategory.NETWORK, ErrorSeverity.HIGH
        
        # Check error message for patterns
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        elif any(keyword in error_message for keyword in ['connection', 'network', 'dns']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif any(keyword in error_message for keyword in ['permission', 'access denied']):
            return ErrorCategory.STORAGE, ErrorSeverity.HIGH
        elif any(keyword in error_message for keyword in ['memory', 'out of memory']):
            return ErrorCategory.PROCESSING, ErrorSeverity.CRITICAL
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class StructuredErrorLogger:
    """
    Comprehensive logging system with structured error information.
    """
    
    def __init__(self, log_file: str = "logs/errors.jsonl"):
        """
        Initialize structured error logger.
        
        Args:
            log_file: Path to error log file
        """
        self.log_file = log_file
        self.classifier = ErrorClassifier()
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        retry_count: int = 0
    ) -> ErrorInfo:
        """
        Log an error with structured information.
        
        Args:
            exception: Exception that occurred
            context: Additional context information
            source_url: URL where error occurred
            source_name: Name of source where error occurred
            retry_count: Number of retry attempts
            
        Returns:
            ErrorInfo object with structured error data
        """
        # Generate unique error ID
        error_id = f"err_{int(time.time() * 1000)}_{id(exception)}"
        
        # Classify error
        category, severity = self.classifier.classify_error(exception)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_info=traceback.format_exc(),
            context=context or {},
            retry_count=retry_count,
            source_url=source_url,
            source_name=source_name
        )
        
        # Log to standard logger
        log_level = self._get_log_level(severity)
        self.logger.log(
            log_level,
            f"[{error_id}] {category.value.upper()}: {exception}",
            extra={
                'error_id': error_id,
                'category': category.value,
                'severity': severity.value,
                'source_url': source_url,
                'source_name': source_name
            }
        )
        
        # Write structured log entry
        self._write_structured_log(error_info)
        
        return error_info
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        severity_mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_mapping.get(severity, logging.WARNING)
    
    def _write_structured_log(self, error_info: ErrorInfo) -> None:
        """Write structured error information to JSON log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(error_info.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write structured error log: {e}")


class RetryManager:
    """
    Manages retry operations with exponential backoff and error classification.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff: Optional[ExponentialBackoff] = None,
        error_logger: Optional[StructuredErrorLogger] = None
    ):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff: Exponential backoff configuration
            error_logger: Error logger for structured logging
        """
        self.max_retries = max_retries
        self.backoff = backoff or ExponentialBackoff()
        self.error_logger = error_logger or StructuredErrorLogger()
        self.logger = logging.getLogger(__name__)
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            context: Additional context for error logging
            source_url: URL for error context
            source_name: Source name for error context
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Log the error
                error_info = self.error_logger.log_error(
                    e,
                    context=context,
                    source_url=source_url,
                    source_name=source_name,
                    retry_count=attempt
                )
                
                # Check if we should retry
                if attempt >= self.max_retries:
                    self.logger.error(
                        f"Max retries ({self.max_retries}) exceeded for {func.__name__}. "
                        f"Final error: {e}"
                    )
                    break
                
                # Check if error is retryable
                if not self._is_retryable_error(error_info):
                    self.logger.info(f"Non-retryable error, stopping retries: {e}")
                    break
                
                # Wait before retry
                await self.backoff.wait(attempt)
                self.logger.info(f"Retrying {func.__name__} (attempt {attempt + 2}/{self.max_retries + 1})")
        
        # All retries failed
        raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute synchronous function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Additional context for error logging
            source_url: URL for error context
            source_name: Source name for error context
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Log the error
                error_info = self.error_logger.log_error(
                    e,
                    context=context,
                    source_url=source_url,
                    source_name=source_name,
                    retry_count=attempt
                )
                
                # Check if we should retry
                if attempt >= self.max_retries:
                    self.logger.error(
                        f"Max retries ({self.max_retries}) exceeded for {func.__name__}. "
                        f"Final error: {e}"
                    )
                    break
                
                # Check if error is retryable
                if not self._is_retryable_error(error_info):
                    self.logger.info(f"Non-retryable error, stopping retries: {e}")
                    break
                
                # Wait before retry (synchronous)
                delay = self.backoff.calculate_delay(attempt)
                time.sleep(delay)
                self.logger.info(f"Retrying {func.__name__} (attempt {attempt + 2}/{self.max_retries + 1})")
        
        # All retries failed
        raise last_exception
    
    def _is_retryable_error(self, error_info: ErrorInfo) -> bool:
        """
        Determine if an error should be retried based on its classification.
        
        Args:
            error_info: Structured error information
            
        Returns:
            True if error should be retried, False otherwise
        """
        # Don't retry critical errors
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry authentication errors
        if error_info.category == ErrorCategory.AUTHENTICATION:
            return False
        
        # Don't retry certain content errors
        if error_info.category == ErrorCategory.CONTENT and 'decode' in error_info.message.lower():
            return False
        
        # Retry network, timeout, and rate limit errors
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT
        }
        
        return error_info.category in retryable_categories


def with_error_handling(
    max_retries: int = 3,
    circuit_breaker: Optional[CircuitBreaker] = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator that adds comprehensive error handling to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        circuit_breaker: Optional circuit breaker for the function
        context: Additional context for error logging
    """
    def decorator(func: Callable) -> Callable:
        retry_manager = RetryManager(max_retries=max_retries)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract source information from kwargs if available
            source_url = kwargs.pop('_source_url', None)
            source_name = kwargs.pop('_source_name', None)
            
            # Apply circuit breaker if provided
            if circuit_breaker:
                return await circuit_breaker.call(
                    retry_manager.retry_async,
                    func, *args,
                    context=context,
                    source_url=source_url,
                    source_name=source_name,
                    **kwargs
                )
            else:
                return await retry_manager.retry_async(
                    func, *args,
                    context=context,
                    source_url=source_url,
                    source_name=source_name,
                    **kwargs
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract source information from kwargs if available
            source_url = kwargs.pop('_source_url', None)
            source_name = kwargs.pop('_source_name', None)
            
            return retry_manager.retry_sync(
                func, *args,
                context=context,
                source_url=source_url,
                source_name=source_name,
                **kwargs
            )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator