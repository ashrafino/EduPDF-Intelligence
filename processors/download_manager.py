"""
Async download manager for concurrent PDF downloads.
Replaces requests with aiohttp for improved performance and scalability.
Enhanced with robust error handling and recovery mechanisms.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from urllib.parse import urlparse
import aiohttp
import aiofiles
from dataclasses import dataclass, field

from data.models import PDFMetadata, SourceConfig
from utils.error_handling import (
    with_error_handling, CircuitBreaker, CircuitBreakerConfig,
    StructuredErrorLogger, RetryManager
)
from utils.error_recovery import ErrorRecoveryManager


@dataclass
class DownloadStats:
    """Statistics for download operations."""
    total_downloads: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    bytes_downloaded: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_downloads == 0:
            return 0.0
        return (self.successful_downloads / self.total_downloads) * 100
    
    @property
    def download_speed_mbps(self) -> float:
        """Calculate average download speed in MB/s."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return (self.bytes_downloaded / (1024 * 1024)) / elapsed


@dataclass
class DownloadTask:
    """Individual download task with metadata."""
    url: str
    destination: Path
    source_config: SourceConfig
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    start_time: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate download progress percentage."""
        if not self.total_bytes or self.total_bytes == 0:
            return 0.0
        return (self.bytes_downloaded / self.total_bytes) * 100


class BandwidthThrottler:
    """Manages bandwidth throttling and rate limiting."""
    
    def __init__(self, max_bandwidth_mbps: float = 10.0):
        """
        Initialize bandwidth throttler.
        
        Args:
            max_bandwidth_mbps: Maximum bandwidth in MB/s
        """
        self.max_bandwidth_mbps = max_bandwidth_mbps
        self.max_bytes_per_second = max_bandwidth_mbps * 1024 * 1024
        self.download_history: List[Tuple[datetime, int]] = []
        self._lock = asyncio.Lock()
    
    async def throttle_if_needed(self, bytes_to_download: int) -> None:
        """
        Apply throttling if current bandwidth usage exceeds limits.
        
        Args:
            bytes_to_download: Number of bytes about to be downloaded
        """
        async with self._lock:
            now = datetime.now()
            
            # Clean old entries (older than 1 second)
            cutoff_time = now - timedelta(seconds=1)
            self.download_history = [
                (timestamp, size) for timestamp, size in self.download_history
                if timestamp > cutoff_time
            ]
            
            # Calculate current bandwidth usage
            current_bytes = sum(size for _, size in self.download_history)
            
            if current_bytes + bytes_to_download > self.max_bytes_per_second:
                # Calculate required delay
                excess_bytes = (current_bytes + bytes_to_download) - self.max_bytes_per_second
                delay_seconds = excess_bytes / self.max_bytes_per_second
                
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
            
            # Record this download
            self.download_history.append((now, bytes_to_download))


class ConnectionPool:
    """Manages HTTP connection pooling and session management."""
    
    def __init__(self, max_connections: int = 100, max_connections_per_host: int = 10):
        """
        Initialize connection pool.
        
        Args:
            max_connections: Maximum total connections
            max_connections_per_host: Maximum connections per host
        """
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self._lock = asyncio.Lock()
    
    async def get_session(self, source_config: SourceConfig) -> aiohttp.ClientSession:
        """
        Get or create a session for the given source configuration.
        
        Args:
            source_config: Source configuration
            
        Returns:
            HTTP client session
        """
        async with self._lock:
            session_key = f"{source_config.name}_{id(source_config)}"
            
            if session_key not in self.sessions:
                # Create new session with appropriate configuration
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=source_config.timeout,
                    connect=10,
                    sock_read=30
                )
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=source_config.headers,
                    raise_for_status=False
                )
                
                self.sessions[session_key] = session
            
            return self.sessions[session_key]
    
    async def close_all(self) -> None:
        """Close all sessions and clean up connections."""
        async with self._lock:
            for session in self.sessions.values():
                await session.close()
            self.sessions.clear()


class AsyncDownloadManager:
    """
    Async download manager for concurrent PDF downloads with advanced features.
    Provides connection pooling, bandwidth throttling, progress tracking, statistics,
    and comprehensive error handling with recovery mechanisms.
    """
    
    def __init__(
        self,
        max_concurrent_downloads: int = 10,
        max_bandwidth_mbps: float = 10.0,
        download_directory: str = "downloads",
        progress_callback: Optional[Callable[[DownloadTask], None]] = None,
        enable_error_recovery: bool = True
    ):
        """
        Initialize the async download manager.
        
        Args:
            max_concurrent_downloads: Maximum number of concurrent downloads
            max_bandwidth_mbps: Maximum bandwidth usage in MB/s
            download_directory: Directory to save downloaded files
            progress_callback: Optional callback for progress updates
            enable_error_recovery: Whether to enable error recovery mechanisms
        """
        self.max_concurrent_downloads = max_concurrent_downloads
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.connection_pool = ConnectionPool()
        self.bandwidth_throttler = BandwidthThrottler(max_bandwidth_mbps)
        self.progress_callback = progress_callback
        
        # Error handling and recovery
        self.enable_error_recovery = enable_error_recovery
        if enable_error_recovery:
            self.error_logger = StructuredErrorLogger("logs/download_errors.jsonl")
            self.recovery_manager = ErrorRecoveryManager("checkpoints/downloads")
            self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # State management
        self.download_queue: asyncio.Queue[DownloadTask] = asyncio.Queue()
        self.active_downloads: Dict[str, DownloadTask] = {}
        self.completed_downloads: List[DownloadTask] = []
        self.stats = DownloadStats()
        
        # Control
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(max_concurrent_downloads)
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the download manager and worker tasks."""
        if self._running:
            return
        
        self._running = True
        self.stats = DownloadStats()
        
        # Start worker tasks
        for i in range(self.max_concurrent_downloads):
            worker = asyncio.create_task(self._download_worker(f"worker-{i}"))
            self._workers.append(worker)
        
        self.logger.info(f"Started download manager with {self.max_concurrent_downloads} workers")
    
    async def stop(self) -> None:
        """Stop the download manager and clean up resources."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Close connection pool
        await self.connection_pool.close_all()
        
        self._workers.clear()
        self.logger.info("Download manager stopped")
    
    async def add_download(
        self,
        url: str,
        source_config: SourceConfig,
        filename: Optional[str] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DownloadTask:
        """
        Add a download task to the queue.
        
        Args:
            url: URL to download
            source_config: Source configuration
            filename: Optional custom filename
            priority: Download priority (higher = more important)
            metadata: Optional metadata dictionary
            
        Returns:
            DownloadTask object for tracking
        """
        if not filename:
            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or not filename.endswith('.pdf'):
                filename = f"document_{int(time.time())}.pdf"
        
        destination = self.download_directory / filename
        
        task = DownloadTask(
            url=url,
            destination=destination,
            source_config=source_config,
            priority=priority,
            metadata=metadata or {}
        )
        
        await self.download_queue.put(task)
        self.logger.debug(f"Added download task: {url}")
        
        return task
    
    async def add_bulk_downloads(
        self,
        urls: List[str],
        source_config: SourceConfig,
        priority: int = 1
    ) -> List[DownloadTask]:
        """
        Add multiple downloads to the queue efficiently.
        
        Args:
            urls: List of URLs to download
            source_config: Source configuration
            priority: Download priority for all URLs
            
        Returns:
            List of DownloadTask objects
        """
        tasks = []
        
        for url in urls:
            task = await self.add_download(url, source_config, priority=priority)
            tasks.append(task)
        
        self.logger.info(f"Added {len(tasks)} downloads to queue")
        return tasks
    
    async def _download_worker(self, worker_name: str) -> None:
        """
        Worker task that processes downloads from the queue.
        
        Args:
            worker_name: Name of the worker for logging
        """
        self.logger.debug(f"Download worker {worker_name} started")
        
        while self._running:
            try:
                # Get next task from queue with timeout
                task = await asyncio.wait_for(
                    self.download_queue.get(),
                    timeout=1.0
                )
                
                async with self._semaphore:
                    await self._process_download_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
        
        self.logger.debug(f"Download worker {worker_name} stopped")
    
    async def _process_download_task(self, task: DownloadTask, worker_name: str) -> None:
        """
        Process a single download task with enhanced error handling.
        
        Args:
            task: Download task to process
            worker_name: Name of the worker processing the task
        """
        task.start_time = datetime.now()
        self.active_downloads[task.url] = task
        self.stats.total_downloads += 1
        
        try:
            self.logger.info(f"[{worker_name}] Starting download: {task.url}")
            
            # Apply rate limiting
            await asyncio.sleep(task.source_config.rate_limit)
            
            # Get or create circuit breaker for this source
            circuit_breaker = self._get_circuit_breaker(task.source_config.name)
            
            # Perform the download with error handling
            if self.enable_error_recovery:
                success = await self._download_file_with_recovery(task, circuit_breaker)
            else:
                success = await self._download_file(task)
            
            if success:
                self.stats.successful_downloads += 1
                self.logger.info(f"[{worker_name}] Completed download: {task.destination}")
            else:
                self.stats.failed_downloads += 1
                
                # Enhanced retry logic with error recovery
                if self.enable_error_recovery:
                    await self._handle_download_failure(task, worker_name)
                else:
                    # Basic retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await self.download_queue.put(task)
                        self.logger.warning(f"[{worker_name}] Retrying download ({task.retry_count}/{task.max_retries}): {task.url}")
                    else:
                        self.logger.error(f"[{worker_name}] Failed download after {task.max_retries} retries: {task.url}")
        
        except Exception as e:
            self.logger.error(f"[{worker_name}] Error processing download {task.url}: {e}")
            self.stats.failed_downloads += 1
            
            # Log structured error if recovery is enabled
            if self.enable_error_recovery:
                self.error_logger.log_error(
                    e,
                    context={'worker': worker_name, 'task_metadata': task.metadata},
                    source_url=task.url,
                    source_name=task.source_config.name,
                    retry_count=task.retry_count
                )
        
        finally:
            # Clean up
            if task.url in self.active_downloads:
                del self.active_downloads[task.url]
            self.completed_downloads.append(task)
            
            # Call progress callback if provided
            if self.progress_callback:
                try:
                    self.progress_callback(task)
                except Exception as e:
                    self.logger.error(f"Progress callback error: {e}")
    
    async def _download_file(self, task: DownloadTask) -> bool:
        """
        Download a single file with progress tracking and bandwidth throttling.
        
        Args:
            task: Download task to execute
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            session = await self.connection_pool.get_session(task.source_config)
            
            async with session.get(task.url) as response:
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {task.url}")
                    return False
                
                # Get content length for progress tracking
                content_length = response.headers.get('content-length')
                if content_length:
                    task.total_bytes = int(content_length)
                
                # Check file size limits
                if task.total_bytes:
                    if task.total_bytes < task.source_config.min_file_size:
                        self.logger.warning(f"File too small ({task.total_bytes} bytes): {task.url}")
                        return False
                    
                    if task.total_bytes > task.source_config.max_file_size:
                        self.logger.warning(f"File too large ({task.total_bytes} bytes): {task.url}")
                        return False
                
                # Download file with progress tracking
                async with aiofiles.open(task.destination, 'wb') as f:
                    chunk_size = 8192  # 8KB chunks
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        # Apply bandwidth throttling
                        await self.bandwidth_throttler.throttle_if_needed(len(chunk))
                        
                        # Write chunk
                        await f.write(chunk)
                        
                        # Update progress
                        task.bytes_downloaded += len(chunk)
                        self.stats.bytes_downloaded += len(chunk)
                
                # Verify download
                if task.destination.exists() and task.destination.stat().st_size > 0:
                    self.logger.debug(f"Successfully downloaded {task.destination} ({task.bytes_downloaded} bytes)")
                    return True
                else:
                    self.logger.error(f"Download verification failed: {task.destination}")
                    return False
        
        except asyncio.TimeoutError:
            self.logger.error(f"Download timeout: {task.url}")
            return False
        except Exception as e:
            self.logger.error(f"Download error for {task.url}: {e}")
            return False
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive download statistics.
        
        Returns:
            Dictionary containing download statistics
        """
        return {
            'total_downloads': self.stats.total_downloads,
            'successful_downloads': self.stats.successful_downloads,
            'failed_downloads': self.stats.failed_downloads,
            'success_rate': self.stats.success_rate,
            'bytes_downloaded': self.stats.bytes_downloaded,
            'download_speed_mbps': self.stats.download_speed_mbps,
            'active_downloads': len(self.active_downloads),
            'queue_size': self.download_queue.qsize(),
            'completed_downloads': len(self.completed_downloads),
            'start_time': self.stats.start_time,
            'elapsed_time': (datetime.now() - self.stats.start_time).total_seconds()
        }
    
    def get_active_downloads(self) -> List[Dict[str, Any]]:
        """
        Get information about currently active downloads.
        
        Returns:
            List of active download information
        """
        active = []
        
        for task in self.active_downloads.values():
            active.append({
                'url': task.url,
                'destination': str(task.destination),
                'progress_percentage': task.progress_percentage,
                'bytes_downloaded': task.bytes_downloaded,
                'total_bytes': task.total_bytes,
                'start_time': task.start_time,
                'source': task.source_config.name
            })
        
        return active
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all downloads to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if all downloads completed, False if timeout
        """
        start_time = time.time()
        
        while self._running and (self.download_queue.qsize() > 0 or len(self.active_downloads) > 0):
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.1)
        
        return True
    
    def _get_circuit_breaker(self, source_name: str) -> Optional[CircuitBreaker]:
        """
        Get or create a circuit breaker for a source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            CircuitBreaker instance if error recovery is enabled, None otherwise
        """
        if not self.enable_error_recovery:
            return None
        
        if source_name not in self.circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=300.0,  # 5 minutes
                expected_exception=Exception
            )
            self.circuit_breakers[source_name] = CircuitBreaker(config, source_name)
        
        return self.circuit_breakers[source_name]
    
    async def _download_file_with_recovery(
        self,
        task: DownloadTask,
        circuit_breaker: Optional[CircuitBreaker]
    ) -> bool:
        """
        Download a file with error recovery mechanisms.
        
        Args:
            task: Download task to execute
            circuit_breaker: Circuit breaker for the source
            
        Returns:
            True if download successful, False otherwise
        """
        retry_manager = RetryManager(
            max_retries=task.max_retries,
            error_logger=self.error_logger
        )
        
        try:
            if circuit_breaker:
                return await circuit_breaker.call(
                    retry_manager.retry_async,
                    self._download_file,
                    task,
                    context={'download_task': task.metadata},
                    source_url=task.url,
                    source_name=task.source_config.name
                )
            else:
                return await retry_manager.retry_async(
                    self._download_file,
                    task,
                    context={'download_task': task.metadata},
                    source_url=task.url,
                    source_name=task.source_config.name
                )
        except Exception as e:
            self.logger.error(f"Download failed with recovery for {task.url}: {e}")
            return False
    
    async def _handle_download_failure(self, task: DownloadTask, worker_name: str) -> None:
        """
        Handle download failure with recovery strategies.
        
        Args:
            task: Failed download task
            worker_name: Name of the worker
        """
        if task.retry_count < task.max_retries:
            # Calculate exponential backoff delay
            delay = min(2 ** task.retry_count, 60)  # Max 60 seconds
            
            task.retry_count += 1
            
            # Add delay before retrying
            await asyncio.sleep(delay)
            await self.download_queue.put(task)
            
            self.logger.warning(
                f"[{worker_name}] Retrying download with {delay}s delay "
                f"({task.retry_count}/{task.max_retries}): {task.url}"
            )
        else:
            self.logger.error(
                f"[{worker_name}] Failed download after {task.max_retries} retries: {task.url}"
            )
            
            # Mark source as potentially problematic if too many failures
            circuit_breaker = self._get_circuit_breaker(task.source_config.name)
            if circuit_breaker:
                circuit_breaker._on_failure()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error and recovery statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        if not self.enable_error_recovery:
            return {'error_recovery_disabled': True}
        
        stats = {
            'circuit_breakers': {},
            'recovery_stats': self.recovery_manager.get_recovery_statistics()
        }
        
        # Add circuit breaker states
        for source_name, breaker in self.circuit_breakers.items():
            stats['circuit_breakers'][source_name] = breaker.get_state()
        
        return stats
    
    def reset_circuit_breaker(self, source_name: str) -> bool:
        """
        Manually reset a circuit breaker for a source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            True if reset successful, False if breaker not found
        """
        if source_name in self.circuit_breakers:
            breaker = self.circuit_breakers[source_name]
            breaker.state = breaker.CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            self.logger.info(f"Manually reset circuit breaker for {source_name}")
            return True
        
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()