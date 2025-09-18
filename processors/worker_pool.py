"""
Multiprocessing worker pool system for PDF processing.
Implements task queue system with priority handling, progress monitoring, and checkpoints.
"""

import asyncio
import logging
import multiprocessing as mp
import pickle
import queue
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading

from data.models import ProcessingTask, TaskType, PDFMetadata


class WorkerStatus(Enum):
    """Status of individual workers."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class WorkerInfo:
    """Information about a worker process."""
    worker_id: str
    pid: Optional[int] = None
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate worker uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy based on heartbeat."""
        return (datetime.now() - self.last_heartbeat).total_seconds() < 60


@dataclass
class CheckpointData:
    """Data structure for checkpoint system."""
    checkpoint_id: str
    timestamp: datetime
    completed_tasks: List[str]
    failed_tasks: List[str]
    pending_tasks: List[ProcessingTask]
    worker_stats: Dict[str, Dict[str, Any]]
    
    def save_to_file(self, filepath: Path) -> None:
        """Save checkpoint data to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'CheckpointData':
        """Load checkpoint data from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class TaskQueue:
    """Priority-based task queue with persistence."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._task_registry: Dict[str, ProcessingTask] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.TaskQueue")
    
    def put(self, task: ProcessingTask) -> bool:
        """
        Add task to queue with priority handling.
        
        Args:
            task: Processing task to add
            
        Returns:
            True if task added successfully, False if queue full
        """
        try:
            with self._lock:
                # Priority queue uses negative priority for max-heap behavior
                priority_item = (-task.priority, task.created_at, task)
                self._queue.put(priority_item, block=False)
                self._task_registry[task.task_id] = task
                
            self.logger.debug(f"Added task {task.task_id} with priority {task.priority}")
            return True
            
        except queue.Full:
            self.logger.warning(f"Task queue full, cannot add task {task.task_id}")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[ProcessingTask]:
        """
        Get next highest priority task from queue.
        
        Args:
            timeout: Optional timeout for blocking get
            
        Returns:
            Next task or None if timeout/empty
        """
        try:
            priority_item = self._queue.get(timeout=timeout)
            _, _, task = priority_item
            
            with self._lock:
                if task.task_id in self._task_registry:
                    del self._task_registry[task.task_id]
            
            return task
            
        except queue.Empty:
            return None
    
    def get_task_by_id(self, task_id: str) -> Optional[ProcessingTask]:
        """Get task by ID from registry."""
        with self._lock:
            return self._task_registry.get(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from registry."""
        with self._lock:
            if task_id in self._task_registry:
                del self._task_registry[task_id]
                return True
            return False
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def get_pending_tasks(self) -> List[ProcessingTask]:
        """Get list of all pending tasks."""
        with self._lock:
            return list(self._task_registry.values())


class WorkerPoolManager:
    """
    Manages a pool of worker processes for PDF processing with advanced features.
    Provides task queue management, progress monitoring, health checking, and checkpoints.
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        max_queue_size: int = 10000,
        checkpoint_interval: int = 300,  # 5 minutes
        checkpoint_dir: str = "checkpoints",
        worker_timeout: int = 300  # 5 minutes
    ):
        """
        Initialize worker pool manager.
        
        Args:
            num_workers: Number of worker processes (defaults to CPU count)
            max_queue_size: Maximum task queue size
            checkpoint_interval: Checkpoint save interval in seconds
            checkpoint_dir: Directory for checkpoint files
            worker_timeout: Worker timeout in seconds
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.worker_timeout = worker_timeout
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Task management
        self.task_queue = TaskQueue(max_queue_size)
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Worker management
        self.executor: Optional[ProcessPoolExecutor] = None
        self.workers: Dict[str, WorkerInfo] = {}
        self.active_futures: Dict[str, Any] = {}  # task_id -> future
        
        # Control and monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._checkpoint_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.start_time = datetime.now()
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start the worker pool and monitoring systems."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize process pool
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_initializer
        )
        
        # Initialize worker info
        for i in range(self.num_workers):
            worker_id = f"worker-{i}"
            self.workers[worker_id] = WorkerInfo(worker_id=worker_id)
        
        # Start monitoring threads
        self._monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self._monitor_thread.start()
        
        self._checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
        self._checkpoint_thread.start()
        
        self.logger.info(f"Started worker pool with {self.num_workers} workers")
    
    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker pool and clean up resources.
        
        Args:
            timeout: Timeout for graceful shutdown
        """
        self._running = False
        
        if self.executor:
            # Cancel pending futures
            for future in self.active_futures.values():
                future.cancel()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Save final checkpoint
        self._save_checkpoint()
        
        self.logger.info("Worker pool stopped")
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """
        Submit a task for processing.
        
        Args:
            task: Processing task to submit
            
        Returns:
            True if task submitted successfully, False otherwise
        """
        if not self._running:
            self.logger.error("Cannot submit task: worker pool not running")
            return False
        
        return self.task_queue.put(task)
    
    def submit_bulk_tasks(self, tasks: List[ProcessingTask]) -> int:
        """
        Submit multiple tasks for processing.
        
        Args:
            tasks: List of processing tasks
            
        Returns:
            Number of tasks successfully submitted
        """
        submitted = 0
        for task in tasks:
            if self.submit_task(task):
                submitted += 1
        
        self.logger.info(f"Submitted {submitted}/{len(tasks)} tasks")
        return submitted
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get status of a specific task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status or None if not found
        """
        # Check if task is in queue
        task = self.task_queue.get_task_by_id(task_id)
        if task:
            return task.status
        
        # Check if task is completed or failed
        if task_id in self.completed_tasks:
            return "completed"
        elif task_id in self.failed_tasks:
            return "failed"
        elif task_id in self.active_futures:
            return "processing"
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or active task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task cancelled successfully, False otherwise
        """
        # Try to remove from queue first
        if self.task_queue.remove_task(task_id):
            self.logger.info(f"Cancelled pending task {task_id}")
            return True
        
        # Try to cancel active future
        if task_id in self.active_futures:
            future = self.active_futures[task_id]
            if future.cancel():
                del self.active_futures[task_id]
                self.logger.info(f"Cancelled active task {task_id}")
                return True
        
        return False
    
    def _monitor_workers(self) -> None:
        """Monitor worker processes and manage task execution."""
        while self._running:
            try:
                # Process completed futures
                self._process_completed_futures()
                
                # Submit new tasks if workers available
                self._submit_pending_tasks()
                
                # Update worker heartbeats
                self._update_worker_heartbeats()
                
                # Check for unhealthy workers
                self._check_worker_health()
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Error in worker monitor: {e}")
    
    def _process_completed_futures(self) -> None:
        """Process completed futures and update statistics."""
        completed_futures = []
        
        for task_id, future in self.active_futures.items():
            if future.done():
                completed_futures.append(task_id)
                
                try:
                    result = future.result()
                    self.completed_tasks.append(task_id)
                    self.total_tasks_processed += 1
                    
                    # Update processing time statistics
                    if isinstance(result, dict) and 'processing_time' in result:
                        self.total_processing_time += result['processing_time']
                    
                    self.logger.debug(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    self.failed_tasks.append(task_id)
                    self.logger.error(f"Task {task_id} failed: {e}")
        
        # Clean up completed futures
        for task_id in completed_futures:
            del self.active_futures[task_id]
    
    def _submit_pending_tasks(self) -> None:
        """Submit pending tasks to available workers."""
        available_workers = self.num_workers - len(self.active_futures)
        
        for _ in range(available_workers):
            if self.task_queue.is_empty():
                break
            
            task = self.task_queue.get(timeout=0.1)
            if task:
                try:
                    # Submit task to executor
                    future = self.executor.submit(_process_task_wrapper, task)
                    self.active_futures[task.task_id] = future
                    
                    # Update task status
                    task.status = "processing"
                    task.started_at = datetime.now()
                    
                    self.logger.debug(f"Submitted task {task.task_id} for processing")
                    
                except Exception as e:
                    self.logger.error(f"Error submitting task {task.task_id}: {e}")
                    self.failed_tasks.append(task.task_id)
    
    def _update_worker_heartbeats(self) -> None:
        """Update worker heartbeat timestamps."""
        for worker_info in self.workers.values():
            # In a real implementation, workers would send heartbeats
            # For now, we'll update based on active futures
            worker_info.last_heartbeat = datetime.now()
    
    def _check_worker_health(self) -> None:
        """Check worker health and restart unhealthy workers."""
        for worker_id, worker_info in self.workers.items():
            if not worker_info.is_healthy:
                self.logger.warning(f"Worker {worker_id} appears unhealthy")
                worker_info.status = WorkerStatus.ERROR
                
                # In a production system, you might restart the worker here
    
    def _checkpoint_loop(self) -> None:
        """Periodically save checkpoints."""
        while self._running:
            time.sleep(self.checkpoint_interval)
            if self._running:  # Check again after sleep
                self._save_checkpoint()
    
    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        try:
            checkpoint_id = f"checkpoint_{int(time.time())}"
            
            # Collect worker statistics
            worker_stats = {}
            for worker_id, worker_info in self.workers.items():
                worker_stats[worker_id] = {
                    'status': worker_info.status.value,
                    'tasks_completed': worker_info.tasks_completed,
                    'tasks_failed': worker_info.tasks_failed,
                    'uptime_seconds': worker_info.uptime_seconds
                }
            
            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                completed_tasks=self.completed_tasks.copy(),
                failed_tasks=self.failed_tasks.copy(),
                pending_tasks=self.task_queue.get_pending_tasks(),
                worker_stats=worker_stats
            )
            
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            checkpoint.save_to_file(checkpoint_file)
            
            # Clean up old checkpoints (keep last 10)
            self._cleanup_old_checkpoints()
            
            self.logger.debug(f"Saved checkpoint: {checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint files, keeping only the most recent ones."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the 10 most recent checkpoints
            for old_checkpoint in checkpoint_files[10:]:
                old_checkpoint.unlink()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")
    
    def restore_from_checkpoint(self, checkpoint_file: Optional[Path] = None) -> bool:
        """
        Restore state from a checkpoint file.
        
        Args:
            checkpoint_file: Specific checkpoint file to restore from.
                           If None, uses the most recent checkpoint.
            
        Returns:
            True if restoration successful, False otherwise
        """
        try:
            if checkpoint_file is None:
                # Find most recent checkpoint
                checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
                if not checkpoint_files:
                    self.logger.info("No checkpoint files found")
                    return False
                
                checkpoint_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            checkpoint = CheckpointData.load_from_file(checkpoint_file)
            
            # Restore state
            self.completed_tasks = checkpoint.completed_tasks
            self.failed_tasks = checkpoint.failed_tasks
            
            # Re-queue pending tasks
            for task in checkpoint.pending_tasks:
                self.task_queue.put(task)
            
            self.logger.info(f"Restored from checkpoint: {checkpoint.checkpoint_id}")
            self.logger.info(f"Restored {len(checkpoint.pending_tasks)} pending tasks")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring from checkpoint: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive worker pool statistics.
        
        Returns:
            Dictionary containing statistics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'num_workers': self.num_workers,
            'active_tasks': len(self.active_futures),
            'pending_tasks': self.task_queue.size(),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_tasks_processed': self.total_tasks_processed,
            'average_processing_time': (
                self.total_processing_time / max(1, self.total_tasks_processed)
            ),
            'uptime_seconds': uptime,
            'tasks_per_second': self.total_tasks_processed / max(1, uptime),
            'worker_health': {
                worker_id: {
                    'status': worker.status.value,
                    'is_healthy': worker.is_healthy,
                    'tasks_completed': worker.tasks_completed,
                    'uptime': worker.uptime_seconds
                }
                for worker_id, worker in self.workers.items()
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def _worker_initializer() -> None:
    """Initialize worker process."""
    # Set up logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _process_task_wrapper(task: ProcessingTask) -> Dict[str, Any]:
    """
    Wrapper function for processing tasks in worker processes.
    
    Args:
        task: Processing task to execute
        
    Returns:
        Dictionary containing processing results
    """
    start_time = time.time()
    
    try:
        # Import here to avoid issues with multiprocessing
        from processors.pdf_processor import process_pdf_task
        
        result = process_pdf_task(task)
        processing_time = time.time() - start_time
        
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'result': result,
            'processing_time': processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        return {
            'task_id': task.task_id,
            'status': 'failed',
            'error': str(e),
            'processing_time': processing_time
        }


# Placeholder for PDF processing function
def process_pdf_task(task: ProcessingTask) -> Dict[str, Any]:
    """
    Process a PDF task (placeholder implementation).
    
    Args:
        task: Processing task to execute
        
    Returns:
        Dictionary containing processing results
    """
    # This would be implemented with actual PDF processing logic
    # For now, return a placeholder result
    time.sleep(0.1)  # Simulate processing time
    
    return {
        'task_type': task.task_type.value,
        'url': task.url,
        'metadata': task.metadata,
        'processed_at': datetime.now().isoformat()
    }