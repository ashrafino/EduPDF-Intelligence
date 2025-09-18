"""
Error recovery and resilience mechanisms for the educational PDF scraper.
Provides graceful degradation, checkpoint/resume functionality, and recovery strategies.
"""

import asyncio
import logging
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from utils.error_handling import ErrorInfo, ErrorCategory, ErrorSeverity


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of failures."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    PAUSE = "pause"
    ABORT = "abort"


@dataclass
class CheckpointData:
    """Data structure for checkpoint information."""
    checkpoint_id: str
    timestamp: datetime
    operation_type: str
    progress_data: Dict[str, Any]
    completed_items: List[str] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    pending_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint data to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp.isoformat(),
            'operation_type': self.operation_type,
            'progress_data': self.progress_data,
            'completed_items': self.completed_items,
            'failed_items': self.failed_items,
            'pending_items': self.pending_items,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create checkpoint data from dictionary."""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            operation_type=data['operation_type'],
            progress_data=data['progress_data'],
            completed_items=data.get('completed_items', []),
            failed_items=data.get('failed_items', []),
            pending_items=data.get('pending_items', []),
            metadata=data.get('metadata', {})
        )


class CheckpointManager:
    """
    Manages checkpoint/resume functionality for long-running operations.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Active checkpoints
        self.active_checkpoints: Dict[str, CheckpointData] = {}
    
    def create_checkpoint(
        self,
        operation_type: str,
        progress_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new checkpoint for an operation.
        
        Args:
            operation_type: Type of operation being checkpointed
            progress_data: Current progress information
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        # Generate unique checkpoint ID
        timestamp = datetime.now()
        checkpoint_id = f"{operation_type}_{int(timestamp.timestamp())}"
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            operation_type=operation_type,
            progress_data=progress_data,
            metadata=metadata or {}
        )
        
        # Save checkpoint
        self._save_checkpoint(checkpoint_data)
        self.active_checkpoints[checkpoint_id] = checkpoint_data
        
        self.logger.info(f"Created checkpoint {checkpoint_id} for {operation_type}")
        return checkpoint_id
    
    def update_checkpoint(
        self,
        checkpoint_id: str,
        progress_data: Optional[Dict[str, Any]] = None,
        completed_items: Optional[List[str]] = None,
        failed_items: Optional[List[str]] = None,
        pending_items: Optional[List[str]] = None
    ) -> None:
        """
        Update an existing checkpoint with new progress information.
        
        Args:
            checkpoint_id: ID of checkpoint to update
            progress_data: Updated progress information
            completed_items: Items that have been completed
            failed_items: Items that have failed
            pending_items: Items still pending
        """
        if checkpoint_id not in self.active_checkpoints:
            self.logger.warning(f"Checkpoint {checkpoint_id} not found in active checkpoints")
            return
        
        checkpoint = self.active_checkpoints[checkpoint_id]
        
        # Update fields if provided
        if progress_data is not None:
            checkpoint.progress_data.update(progress_data)
        
        if completed_items is not None:
            checkpoint.completed_items.extend(completed_items)
        
        if failed_items is not None:
            checkpoint.failed_items.extend(failed_items)
        
        if pending_items is not None:
            checkpoint.pending_items = pending_items
        
        checkpoint.timestamp = datetime.now()
        
        # Save updated checkpoint
        self._save_checkpoint(checkpoint)
        
        self.logger.debug(f"Updated checkpoint {checkpoint_id}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Load a checkpoint from storage.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            CheckpointData if found, None otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            self.logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint_data = CheckpointData.from_dict(data)
            self.active_checkpoints[checkpoint_id] = checkpoint_data
            
            self.logger.info(f"Loaded checkpoint {checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, operation_type: Optional[str] = None) -> List[CheckpointData]:
        """
        List available checkpoints, optionally filtered by operation type.
        
        Args:
            operation_type: Optional filter by operation type
            
        Returns:
            List of available checkpoints
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                checkpoint_data = CheckpointData.from_dict(data)
                
                if operation_type is None or checkpoint_data.operation_type == operation_type:
                    checkpoints.append(checkpoint_data)
                    
            except Exception as e:
                self.logger.error(f"Error reading checkpoint file {checkpoint_file}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint from storage.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            if checkpoint_id in self.active_checkpoints:
                del self.active_checkpoints[checkpoint_id]
            
            self.logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old checkpoint files.
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    checkpoint_file.unlink()
                    deleted_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing checkpoint file {checkpoint_file}: {e}")
        
        self.logger.info(f"Cleaned up {deleted_count} old checkpoint files")
        return deleted_count
    
    def _save_checkpoint(self, checkpoint_data: CheckpointData) -> None:
        """Save checkpoint data to file."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_data.checkpoint_id}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving checkpoint {checkpoint_data.checkpoint_id}: {e}")


class GracefulDegradation:
    """
    Implements graceful degradation strategies when components fail.
    """
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.fallback_strategies: Dict[str, Callable] = {}
        self.degraded_services: Set[str] = set()
        self.logger = logging.getLogger(__name__)
    
    def register_fallback(self, service_name: str, fallback_func: Callable) -> None:
        """
        Register a fallback function for a service.
        
        Args:
            service_name: Name of the service
            fallback_func: Fallback function to use when service fails
        """
        self.fallback_strategies[service_name] = fallback_func
        self.logger.info(f"Registered fallback for service: {service_name}")
    
    def mark_service_degraded(self, service_name: str) -> None:
        """
        Mark a service as degraded.
        
        Args:
            service_name: Name of the service to mark as degraded
        """
        self.degraded_services.add(service_name)
        self.logger.warning(f"Service marked as degraded: {service_name}")
    
    def restore_service(self, service_name: str) -> None:
        """
        Restore a service from degraded state.
        
        Args:
            service_name: Name of the service to restore
        """
        self.degraded_services.discard(service_name)
        self.logger.info(f"Service restored: {service_name}")
    
    def is_service_degraded(self, service_name: str) -> bool:
        """
        Check if a service is in degraded state.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if service is degraded, False otherwise
        """
        return service_name in self.degraded_services
    
    async def call_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a function with fallback if the service is degraded.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from primary or fallback function
        """
        if not self.is_service_degraded(service_name):
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    return await primary_func(*args, **kwargs)
                else:
                    return primary_func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Primary function failed for {service_name}: {e}")
                self.mark_service_degraded(service_name)
        
        # Use fallback if service is degraded or primary function failed
        if service_name in self.fallback_strategies:
            fallback_func = self.fallback_strategies[service_name]
            self.logger.info(f"Using fallback for degraded service: {service_name}")
            
            try:
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback function also failed for {service_name}: {e}")
                raise e
        else:
            raise RuntimeError(f"No fallback available for degraded service: {service_name}")


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager that coordinates various recovery strategies.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize error recovery manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage
        """
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.graceful_degradation = GracefulDegradation()
        self.recovery_strategies: Dict[ErrorCategory, RecoveryStrategy] = {}
        self.logger = logging.getLogger(__name__)
        
        # Set default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self) -> None:
        """Set up default recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorCategory.RATE_LIMIT: RecoveryStrategy.PAUSE,
            ErrorCategory.CONTENT: RecoveryStrategy.SKIP,
            ErrorCategory.PROCESSING: RecoveryStrategy.FALLBACK,
            ErrorCategory.STORAGE: RecoveryStrategy.RETRY,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ABORT,
            ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY
        }
    
    def set_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy) -> None:
        """
        Set recovery strategy for an error category.
        
        Args:
            category: Error category
            strategy: Recovery strategy to use
        """
        self.recovery_strategies[category] = strategy
        self.logger.info(f"Set recovery strategy for {category.value}: {strategy.value}")
    
    def get_recovery_strategy(self, error_info: ErrorInfo) -> RecoveryStrategy:
        """
        Get the appropriate recovery strategy for an error.
        
        Args:
            error_info: Error information
            
        Returns:
            Recovery strategy to use
        """
        # Check for critical errors first
        if error_info.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
        
        # Get strategy based on category
        return self.recovery_strategies.get(error_info.category, RecoveryStrategy.RETRY)
    
    async def handle_error_with_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any],
        recovery_callback: Optional[Callable] = None
    ) -> bool:
        """
        Handle an error using appropriate recovery strategy.
        
        Args:
            error_info: Structured error information
            operation_context: Context information about the operation
            recovery_callback: Optional callback for custom recovery logic
            
        Returns:
            True if recovery was successful, False otherwise
        """
        strategy = self.get_recovery_strategy(error_info)
        
        self.logger.info(
            f"Handling error {error_info.error_id} with strategy: {strategy.value}"
        )
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._handle_retry_recovery(error_info, operation_context)
            
            elif strategy == RecoveryStrategy.SKIP:
                return await self._handle_skip_recovery(error_info, operation_context)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._handle_fallback_recovery(error_info, operation_context)
            
            elif strategy == RecoveryStrategy.PAUSE:
                return await self._handle_pause_recovery(error_info, operation_context)
            
            elif strategy == RecoveryStrategy.ABORT:
                return await self._handle_abort_recovery(error_info, operation_context)
            
            else:
                self.logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during recovery handling: {e}")
            return False
    
    async def _handle_retry_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any]
    ) -> bool:
        """Handle retry recovery strategy."""
        max_retries = operation_context.get('max_retries', 3)
        
        if error_info.retry_count >= max_retries:
            self.logger.warning(
                f"Max retries exceeded for error {error_info.error_id}, switching to skip"
            )
            return await self._handle_skip_recovery(error_info, operation_context)
        
        # Calculate delay based on retry count
        delay = min(2 ** error_info.retry_count, 60)  # Exponential backoff, max 60s
        
        self.logger.info(f"Retrying after {delay}s delay (attempt {error_info.retry_count + 1})")
        await asyncio.sleep(delay)
        
        return True
    
    async def _handle_skip_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any]
    ) -> bool:
        """Handle skip recovery strategy."""
        self.logger.info(f"Skipping failed item due to error {error_info.error_id}")
        
        # Update checkpoint if available
        checkpoint_id = operation_context.get('checkpoint_id')
        if checkpoint_id and error_info.source_url:
            self.checkpoint_manager.update_checkpoint(
                checkpoint_id,
                failed_items=[error_info.source_url]
            )
        
        return True
    
    async def _handle_fallback_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any]
    ) -> bool:
        """Handle fallback recovery strategy."""
        service_name = operation_context.get('service_name', 'unknown')
        
        self.graceful_degradation.mark_service_degraded(service_name)
        
        self.logger.info(
            f"Marked service {service_name} as degraded due to error {error_info.error_id}"
        )
        
        return True
    
    async def _handle_pause_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any]
    ) -> bool:
        """Handle pause recovery strategy (typically for rate limiting)."""
        # Calculate pause duration based on error type
        if error_info.category == ErrorCategory.RATE_LIMIT:
            pause_duration = operation_context.get('rate_limit_pause', 300)  # 5 minutes default
        else:
            pause_duration = 60  # 1 minute default
        
        self.logger.info(
            f"Pausing operation for {pause_duration}s due to error {error_info.error_id}"
        )
        
        await asyncio.sleep(pause_duration)
        return True
    
    async def _handle_abort_recovery(
        self,
        error_info: ErrorInfo,
        operation_context: Dict[str, Any]
    ) -> bool:
        """Handle abort recovery strategy."""
        self.logger.error(
            f"Aborting operation due to critical error {error_info.error_id}: {error_info.message}"
        )
        
        # Save checkpoint before aborting
        checkpoint_id = operation_context.get('checkpoint_id')
        if checkpoint_id:
            self.checkpoint_manager.update_checkpoint(
                checkpoint_id,
                progress_data={'aborted': True, 'abort_reason': error_info.message}
            )
        
        return False
    
    def create_operation_checkpoint(
        self,
        operation_type: str,
        items_to_process: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a checkpoint for a batch operation.
        
        Args:
            operation_type: Type of operation
            items_to_process: List of items to process
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        progress_data = {
            'total_items': len(items_to_process),
            'completed_count': 0,
            'failed_count': 0,
            'start_time': datetime.now().isoformat()
        }
        
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            operation_type=operation_type,
            progress_data=progress_data,
            metadata=metadata
        )
        
        # Set pending items
        self.checkpoint_manager.update_checkpoint(
            checkpoint_id,
            pending_items=items_to_process
        )
        
        return checkpoint_id
    
    def resume_from_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        Resume an operation from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to resume from
            
        Returns:
            CheckpointData if successful, None otherwise
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        
        if checkpoint_data:
            self.logger.info(
                f"Resuming operation {checkpoint_data.operation_type} from checkpoint {checkpoint_id}"
            )
            
            # Log resume statistics
            completed = len(checkpoint_data.completed_items)
            failed = len(checkpoint_data.failed_items)
            pending = len(checkpoint_data.pending_items)
            
            self.logger.info(
                f"Resume stats - Completed: {completed}, Failed: {failed}, Pending: {pending}"
            )
        
        return checkpoint_data
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about error recovery operations.
        
        Returns:
            Dictionary containing recovery statistics
        """
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'active_checkpoints': len(self.checkpoint_manager.active_checkpoints),
            'degraded_services': list(self.graceful_degradation.degraded_services),
            'recovery_strategies': {
                category.value: strategy.value
                for category, strategy in self.recovery_strategies.items()
            }
        }
        
        # Add checkpoint statistics by operation type
        operation_stats = {}
        for checkpoint in checkpoints:
            op_type = checkpoint.operation_type
            if op_type not in operation_stats:
                operation_stats[op_type] = {'count': 0, 'latest': None}
            
            operation_stats[op_type]['count'] += 1
            if (operation_stats[op_type]['latest'] is None or 
                checkpoint.timestamp > operation_stats[op_type]['latest']):
                operation_stats[op_type]['latest'] = checkpoint.timestamp.isoformat()
        
        stats['operation_statistics'] = operation_stats
        
        return stats