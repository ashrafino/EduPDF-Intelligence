"""
System health monitoring and performance metrics collection for the educational PDF scraper.
Provides comprehensive monitoring, alerting, and reporting capabilities.
"""

import asyncio
import logging
import psutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import gc


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert information structure."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'component': self.component,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'metadata': self.metadata
        }


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels
        }


class SystemHealthMonitor:
    """
    Monitors system health including CPU, memory, disk usage, and application metrics.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize system health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Health thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'load_average': 4.0,
            'open_files': 1000
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._health_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        
        # Callbacks for health events
        self._health_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """
        Set threshold for a health metric.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold
        self.logger.info(f"Set threshold for {metric_name}: {threshold}")
    
    def add_health_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add callback for health check events.
        
        Args:
            callback: Function to call with health data
        """
        self._health_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Started system health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped system health monitoring")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                health_data = await self.check_system_health()
                
                # Store in history
                self._health_history.append(health_data)
                if len(self._health_history) > self._max_history_size:
                    self._health_history.pop(0)
                
                # Call health callbacks
                for callback in self._health_callbacks:
                    try:
                        callback(health_data)
                    except Exception as e:
                        self.logger.error(f"Error in health callback: {e}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary containing health metrics and status
        """
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'metrics': {},
            'alerts': []
        }
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            health_data['metrics']['cpu_percent'] = cpu_percent
            health_data['metrics']['load_average'] = load_avg
            
            # Memory metrics
            memory = psutil.virtual_memory()
            health_data['metrics']['memory_percent'] = memory.percent
            health_data['metrics']['memory_available_gb'] = memory.available / (1024**3)
            health_data['metrics']['memory_used_gb'] = memory.used / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            health_data['metrics']['disk_percent'] = (disk.used / disk.total) * 100
            health_data['metrics']['disk_free_gb'] = disk.free / (1024**3)
            
            # Process metrics
            process = psutil.Process()
            health_data['metrics']['process_memory_mb'] = process.memory_info().rss / (1024**2)
            health_data['metrics']['process_cpu_percent'] = process.cpu_percent()
            health_data['metrics']['open_files'] = len(process.open_files())
            health_data['metrics']['threads'] = process.num_threads()
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                health_data['metrics']['network_bytes_sent'] = net_io.bytes_sent
                health_data['metrics']['network_bytes_recv'] = net_io.bytes_recv
            except Exception:
                pass  # Network metrics not available on all systems
            
            # Check thresholds and generate alerts
            alerts = self._check_thresholds(health_data['metrics'])
            health_data['alerts'] = [alert.to_dict() for alert in alerts]
            
            # Determine overall health status
            if any(alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] for alert in alerts):
                health_data['status'] = 'unhealthy'
            elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
                health_data['status'] = 'degraded'
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            health_data['status'] = 'error'
            health_data['error'] = str(e)
        
        return health_data
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Check metrics against thresholds and generate alerts.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            List of alerts for threshold violations
        """
        alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if value > threshold:
                    # Determine severity based on how much threshold is exceeded
                    excess_percent = ((value - threshold) / threshold) * 100
                    
                    if excess_percent > 50:
                        severity = AlertSeverity.CRITICAL
                    elif excess_percent > 25:
                        severity = AlertSeverity.ERROR
                    else:
                        severity = AlertSeverity.WARNING
                    
                    alert = Alert(
                        alert_id=f"threshold_{metric_name}_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity=severity,
                        component="system_health",
                        message=f"{metric_name} ({value:.2f}) exceeds threshold ({threshold})",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold
                    )
                    alerts.append(alert)
        
        return alerts
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get health history for the specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of health check results
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            health_data for health_data in self._health_history
            if datetime.fromisoformat(health_data['timestamp']) > cutoff_time
        ]
    
    def get_current_health_summary(self) -> Dict[str, Any]:
        """
        Get current health summary.
        
        Returns:
            Dictionary containing current health status
        """
        if not self._health_history:
            return {'status': 'no_data', 'message': 'No health data available'}
        
        latest_health = self._health_history[-1]
        
        return {
            'status': latest_health['status'],
            'timestamp': latest_health['timestamp'],
            'key_metrics': {
                'cpu_percent': latest_health['metrics'].get('cpu_percent', 0),
                'memory_percent': latest_health['metrics'].get('memory_percent', 0),
                'disk_percent': latest_health['metrics'].get('disk_percent', 0)
            },
            'active_alerts': len(latest_health.get('alerts', [])),
            'monitoring_active': self._monitoring
        }


class PerformanceMetricsCollector:
    """
    Collects and aggregates performance metrics for the scraper application.
    """
    
    def __init__(self, metrics_file: str = "logs/metrics.jsonl"):
        """
        Initialize performance metrics collector.
        
        Args:
            metrics_file: Path to metrics log file
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory metrics storage
        self._metrics: Dict[str, List[Metric]] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = {}
        
        # Metrics retention
        self._max_metrics_per_type = 10000
        self._metrics_lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels for the metric
        """
        with self._metrics_lock:
            key = self._get_metric_key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + value
            
            metric = Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                value=self._counters[key],
                timestamp=datetime.now(),
                labels=labels or {}
            )
            
            self._store_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        with self._metrics_lock:
            key = self._get_metric_key(name, labels)
            self._gauges[key] = value
            
            metric = Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )
            
            self._store_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timer metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        with self._metrics_lock:
            key = self._get_metric_key(name, labels)
            if key not in self._timers:
                self._timers[key] = []
            
            self._timers[key].append(duration)
            
            # Keep only recent timer values
            if len(self._timers[key]) > 1000:
                self._timers[key] = self._timers[key][-1000:]
            
            metric = Metric(
                name=name,
                metric_type=MetricType.TIMER,
                value=duration,
                timestamp=datetime.now(),
                labels=labels or {}
            )
            
            self._store_metric(metric)
    
    def timer_context(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            name: Metric name
            labels: Optional labels for the metric
            
        Returns:
            Context manager that records timing
        """
        return TimerContext(self, name, labels)
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _store_metric(self, metric: Metric) -> None:
        """Store metric in memory and write to file."""
        # Store in memory
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []
        
        self._metrics[metric.name].append(metric)
        
        # Limit memory usage
        if len(self._metrics[metric.name]) > self._max_metrics_per_type:
            self._metrics[metric.name] = self._metrics[metric.name][-self._max_metrics_per_type:]
        
        # Write to file
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                json.dump(metric.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Error writing metric to file: {e}")
    
    def get_metric_summary(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            hours: Number of hours to include in summary
            
        Returns:
            Dictionary containing metric summary
        """
        if name not in self._metrics:
            return {'error': f'Metric {name} not found'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self._metrics[name]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': f'No recent data for metric {name}'}
        
        values = [m.value for m in recent_metrics]
        
        summary = {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1],
            'timestamp_range': {
                'start': min(m.timestamp for m in recent_metrics).isoformat(),
                'end': max(m.timestamp for m in recent_metrics).isoformat()
            }
        }
        
        # Add percentiles for timer metrics
        if recent_metrics[0].metric_type == MetricType.TIMER:
            sorted_values = sorted(values)
            summary['percentiles'] = {
                'p50': sorted_values[len(sorted_values) // 2],
                'p90': sorted_values[int(len(sorted_values) * 0.9)],
                'p95': sorted_values[int(len(sorted_values) * 0.95)],
                'p99': sorted_values[int(len(sorted_values) * 0.99)]
            }
        
        return summary
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected metrics.
        
        Returns:
            Dictionary containing all metrics summaries
        """
        summary = {
            'counters': {},
            'gauges': {},
            'timers': {},
            'collection_stats': {
                'total_metric_types': len(self._metrics),
                'total_counter_keys': len(self._counters),
                'total_gauge_keys': len(self._gauges),
                'total_timer_keys': len(self._timers)
            }
        }
        
        # Get summaries for each metric type
        for metric_name in self._metrics:
            metric_summary = self.get_metric_summary(metric_name)
            
            if 'error' not in metric_summary:
                metric_type = self._metrics[metric_name][0].metric_type
                
                if metric_type == MetricType.COUNTER:
                    summary['counters'][metric_name] = metric_summary
                elif metric_type == MetricType.GAUGE:
                    summary['gauges'][metric_name] = metric_summary
                elif metric_type == MetricType.TIMER:
                    summary['timers'][metric_name] = metric_summary
        
        return summary


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: PerformanceMetricsCollector, name: str, labels: Optional[Dict[str, str]]):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)


class SourceAvailabilityChecker:
    """
    Monitors source availability and reports status.
    """
    
    def __init__(self, check_interval: float = 300.0):  # 5 minutes
        """
        Initialize source availability checker.
        
        Args:
            check_interval: Interval between availability checks in seconds
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Source status tracking
        self._source_status: Dict[str, Dict[str, Any]] = {}
        self._checking = False
        self._check_task: Optional[asyncio.Task] = None
        
        # Callbacks for status changes
        self._status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def add_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add callback for source status changes.
        
        Args:
            callback: Function to call with source name and status
        """
        self._status_callbacks.append(callback)
    
    async def start_checking(self, sources: List[Any]) -> None:
        """
        Start continuous source availability checking.
        
        Args:
            sources: List of source configurations to monitor
        """
        if self._checking:
            return
        
        self._sources = sources
        self._checking = True
        self._check_task = asyncio.create_task(self._check_loop())
        self.logger.info(f"Started source availability checking for {len(sources)} sources")
    
    async def stop_checking(self) -> None:
        """Stop source availability checking."""
        self._checking = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped source availability checking")
    
    async def _check_loop(self) -> None:
        """Main checking loop."""
        while self._checking:
            try:
                await self._check_all_sources()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in source availability check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_sources(self) -> None:
        """Check availability of all sources."""
        if not hasattr(self, '_sources'):
            return
        
        tasks = []
        for source in self._sources:
            task = asyncio.create_task(
                self._check_source_availability(source),
                name=f"check_{source.name}"
            )
            tasks.append(task)
        
        # Wait for all checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_source_availability(self, source: Any) -> None:
        """
        Check availability of a single source.
        
        Args:
            source: Source configuration to check
        """
        start_time = time.time()
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    source.base_url,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_time = time.time() - start_time
                    
                    status_info = {
                        'available': response.status == 200,
                        'status_code': response.status,
                        'response_time': response_time,
                        'last_checked': datetime.now(),
                        'error': None
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            status_info = {
                'available': False,
                'status_code': None,
                'response_time': response_time,
                'last_checked': datetime.now(),
                'error': str(e)
            }
        
        # Check if status changed
        previous_status = self._source_status.get(source.name, {})
        status_changed = (
            previous_status.get('available') != status_info['available']
        )
        
        # Update status
        self._source_status[source.name] = status_info
        
        # Call callbacks if status changed
        if status_changed:
            for callback in self._status_callbacks:
                try:
                    callback(source.name, status_info)
                except Exception as e:
                    self.logger.error(f"Error in status callback: {e}")
    
    def get_source_status(self, source_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status for a specific source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            Status dictionary or None if not found
        """
        return self._source_status.get(source_name)
    
    def get_all_source_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all monitored sources.
        
        Returns:
            Dictionary mapping source names to status information
        """
        return self._source_status.copy()
    
    def get_availability_summary(self) -> Dict[str, Any]:
        """
        Get summary of source availability.
        
        Returns:
            Dictionary containing availability statistics
        """
        if not self._source_status:
            return {'total_sources': 0, 'available_sources': 0, 'availability_rate': 0}
        
        total_sources = len(self._source_status)
        available_sources = sum(
            1 for status in self._source_status.values()
            if status.get('available', False)
        )
        
        return {
            'total_sources': total_sources,
            'available_sources': available_sources,
            'unavailable_sources': total_sources - available_sources,
            'availability_rate': (available_sources / total_sources * 100) if total_sources > 0 else 0,
            'last_check_time': max(
                (status.get('last_checked', datetime.min) for status in self._source_status.values()),
                default=None
            )
        }


class MemoryMonitor:
    """
    Monitors memory usage and triggers garbage collection when needed.
    """
    
    def __init__(self, gc_threshold_mb: float = 500.0, check_interval: float = 60.0):
        """
        Initialize memory monitor.
        
        Args:
            gc_threshold_mb: Memory threshold in MB to trigger garbage collection
            check_interval: Interval between memory checks in seconds
        """
        self.gc_threshold_mb = gc_threshold_mb
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._gc_stats = {
            'total_collections': 0,
            'last_collection': None,
            'memory_freed_mb': 0
        }
    
    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Started memory monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped memory monitoring")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._check_memory_usage()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_memory_usage(self) -> None:
        """Check current memory usage and trigger GC if needed."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        if memory_mb > self.gc_threshold_mb:
            self.logger.info(f"Memory usage ({memory_mb:.1f} MB) exceeds threshold, triggering GC")
            
            # Record memory before GC
            memory_before = memory_mb
            
            # Trigger garbage collection
            collected = gc.collect()
            
            # Record memory after GC
            memory_after = process.memory_info().rss / (1024 * 1024)
            memory_freed = memory_before - memory_after
            
            # Update statistics
            self._gc_stats['total_collections'] += 1
            self._gc_stats['last_collection'] = datetime.now()
            self._gc_stats['memory_freed_mb'] += memory_freed
            
            self.logger.info(
                f"GC completed: collected {collected} objects, "
                f"freed {memory_freed:.1f} MB memory"
            )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory monitoring statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'current_memory_mb': memory_info.rss / (1024 * 1024),
            'gc_threshold_mb': self.gc_threshold_mb,
            'gc_stats': self._gc_stats.copy(),
            'monitoring_active': self._monitoring
        }
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Dictionary containing GC results
        """
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        collected = gc.collect()
        
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_freed = memory_before - memory_after
        
        # Update statistics
        self._gc_stats['total_collections'] += 1
        self._gc_stats['last_collection'] = datetime.now()
        self._gc_stats['memory_freed_mb'] += memory_freed
        
        return {
            'objects_collected': collected,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed
        }