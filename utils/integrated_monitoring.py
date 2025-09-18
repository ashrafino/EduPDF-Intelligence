"""
Integrated monitoring system that combines all monitoring components
for the educational PDF scraper application.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from utils.monitoring_dashboard import MonitoringDashboard, MonitoringConfig, AutomatedReporter
from utils.monitoring import PerformanceMetricsCollector
from utils.error_handling import StructuredErrorLogger
from utils.error_recovery import ErrorRecoveryManager


@dataclass
class IntegratedMonitoringConfig:
    """Configuration for integrated monitoring system."""
    # Health monitoring
    health_check_interval: float = 30.0
    
    # Source monitoring
    source_check_interval: float = 300.0
    
    # Memory monitoring
    memory_check_interval: float = 60.0
    gc_threshold_mb: float = 500.0
    
    # Reporting
    report_interval_hours: int = 24
    
    # Alerting
    alert_email_enabled: bool = False
    alert_email_recipients: List[str] = None
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    # File paths
    logs_directory: str = "logs"
    reports_directory: str = "reports"
    checkpoints_directory: str = "checkpoints"


class IntegratedMonitoringSystem:
    """
    Comprehensive monitoring system that integrates all monitoring components
    and provides a unified interface for the PDF scraper application.
    """
    
    def __init__(self, config: IntegratedMonitoringConfig):
        """
        Initialize integrated monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self._create_directories()
        
        # Initialize monitoring components
        self._initialize_components()
        
        # State tracking
        self._monitoring_active = False
        self._sources: List[Any] = []
    
    def _create_directories(self) -> None:
        """Create necessary directories for monitoring."""
        directories = [
            self.config.logs_directory,
            self.config.reports_directory,
            self.config.checkpoints_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self) -> None:
        """Initialize all monitoring components."""
        # Create monitoring configuration
        monitoring_config = MonitoringConfig(
            health_check_interval=self.config.health_check_interval,
            source_check_interval=self.config.source_check_interval,
            memory_check_interval=self.config.memory_check_interval,
            gc_threshold_mb=self.config.gc_threshold_mb,
            report_interval_hours=self.config.report_interval_hours,
            alert_email_enabled=self.config.alert_email_enabled,
            alert_email_recipients=self.config.alert_email_recipients or [],
            smtp_server=self.config.smtp_server,
            smtp_port=self.config.smtp_port,
            smtp_username=self.config.smtp_username,
            smtp_password=self.config.smtp_password
        )
        
        # Initialize main components
        self.dashboard = MonitoringDashboard(monitoring_config)
        self.automated_reporter = AutomatedReporter(self.dashboard, monitoring_config)
        
        # Additional components
        self.error_logger = StructuredErrorLogger(
            f"{self.config.logs_directory}/integrated_errors.jsonl"
        )
        self.recovery_manager = ErrorRecoveryManager(
            f"{self.config.checkpoints_directory}/integrated"
        )
        
        # Setup integration callbacks
        self._setup_integration_callbacks()
    
    def _setup_integration_callbacks(self) -> None:
        """Setup callbacks to integrate monitoring components."""
        # Add alert callback to log errors
        self.dashboard.add_alert_callback(self._handle_monitoring_alert)
    
    def _handle_monitoring_alert(self, alert) -> None:
        """
        Handle alerts from the monitoring system.
        
        Args:
            alert: Alert object
        """
        # Log alert as structured error for analysis
        try:
            # Create a mock exception for the error logger
            class MonitoringAlert(Exception):
                pass
            
            mock_exception = MonitoringAlert(alert.message)
            
            self.error_logger.log_error(
                mock_exception,
                context={
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'metric_name': alert.metric_name,
                    'metric_value': alert.metric_value,
                    'threshold': alert.threshold,
                    'metadata': alert.metadata
                },
                source_name=alert.component
            )
        except Exception as e:
            self.logger.error(f"Error logging monitoring alert: {e}")
    
    async def start_monitoring(self, sources: Optional[List[Any]] = None) -> None:
        """
        Start the integrated monitoring system.
        
        Args:
            sources: Optional list of sources to monitor
        """
        if self._monitoring_active:
            self.logger.warning("Monitoring system is already active")
            return
        
        try:
            self.logger.info("Starting integrated monitoring system...")
            
            # Store sources for monitoring
            if sources:
                self._sources = sources
            
            # Start main dashboard
            await self.dashboard.start_monitoring(self._sources)
            
            # Start automated reporting
            await self.automated_reporter.start_automated_reporting()
            
            self._monitoring_active = True
            
            self.logger.info("Integrated monitoring system started successfully")
            
            # Log initial system status
            await self._log_startup_status()
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop the integrated monitoring system."""
        if not self._monitoring_active:
            return
        
        try:
            self.logger.info("Stopping integrated monitoring system...")
            
            # Stop automated reporting
            await self.automated_reporter.stop_automated_reporting()
            
            # Stop main dashboard
            await self.dashboard.stop_monitoring()
            
            self._monitoring_active = False
            
            self.logger.info("Integrated monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
    
    async def _log_startup_status(self) -> None:
        """Log system status at startup."""
        try:
            health_status = await self.dashboard.health_check()
            self.logger.info(f"Monitoring system health check: {health_status['monitoring_dashboard']}")
            
            # Log component status
            for component, status in health_status['components'].items():
                self.logger.info(f"Component {component}: {status['status']}")
                
        except Exception as e:
            self.logger.error(f"Error logging startup status: {e}")
    
    def get_metrics_collector(self) -> PerformanceMetricsCollector:
        """
        Get the performance metrics collector for application use.
        
        Returns:
            PerformanceMetricsCollector instance
        """
        return self.dashboard.metrics_collector
    
    def get_error_logger(self) -> StructuredErrorLogger:
        """
        Get the structured error logger for application use.
        
        Returns:
            StructuredErrorLogger instance
        """
        return self.error_logger
    
    def get_recovery_manager(self) -> ErrorRecoveryManager:
        """
        Get the error recovery manager for application use.
        
        Returns:
            ErrorRecoveryManager instance
        """
        return self.recovery_manager
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary containing system status information
        """
        try:
            # Get dashboard summary
            dashboard_summary = self.dashboard.get_dashboard_summary()
            
            # Get monitoring system health
            monitoring_health = await self.dashboard.health_check()
            
            # Get recovery statistics
            recovery_stats = self.recovery_manager.get_recovery_statistics()
            
            return {
                'monitoring_active': self._monitoring_active,
                'dashboard_summary': dashboard_summary,
                'monitoring_health': monitoring_health,
                'recovery_statistics': recovery_stats,
                'monitored_sources': len(self._sources),
                'config': {
                    'health_check_interval': self.config.health_check_interval,
                    'source_check_interval': self.config.source_check_interval,
                    'memory_check_interval': self.config.memory_check_interval,
                    'report_interval_hours': self.config.report_interval_hours
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'monitoring_active': self._monitoring_active
            }
    
    async def generate_comprehensive_report(self, hours: int = 24) -> str:
        """
        Generate comprehensive monitoring report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Path to generated report file
        """
        try:
            # Generate main monitoring report
            report_path = await self.dashboard.generate_and_save_report(hours)
            
            # Add recovery statistics to the report
            recovery_stats = self.recovery_manager.get_recovery_statistics()
            
            # Log report generation
            self.logger.info(f"Comprehensive monitoring report generated: {report_path}")
            
            # Record metrics
            self.dashboard.metrics_collector.increment_counter("reports_generated")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def update_sources(self, sources: List[Any]) -> None:
        """
        Update the list of sources being monitored.
        
        Args:
            sources: New list of sources to monitor
        """
        self._sources = sources
        self.logger.info(f"Updated monitored sources count: {len(sources)}")
        
        # If monitoring is active, restart source checking with new sources
        if self._monitoring_active:
            asyncio.create_task(self._restart_source_monitoring())
    
    async def _restart_source_monitoring(self) -> None:
        """Restart source monitoring with updated sources."""
        try:
            await self.dashboard.source_checker.stop_checking()
            await self.dashboard.source_checker.start_checking(self._sources)
            self.logger.info("Source monitoring restarted with updated sources")
        except Exception as e:
            self.logger.error(f"Error restarting source monitoring: {e}")
    
    def force_health_check(self) -> Dict[str, Any]:
        """
        Force an immediate health check.
        
        Returns:
            Health check results
        """
        try:
            # Trigger immediate health check
            health_task = asyncio.create_task(self.dashboard.health_monitor.check_system_health())
            
            # Wait for result (with timeout)
            loop = asyncio.get_event_loop()
            health_data = loop.run_until_complete(
                asyncio.wait_for(health_task, timeout=30.0)
            )
            
            self.logger.info("Forced health check completed")
            return health_data
            
        except Exception as e:
            self.logger.error(f"Error in forced health check: {e}")
            return {'error': str(e)}
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Garbage collection results
        """
        try:
            gc_results = self.dashboard.memory_monitor.force_garbage_collection()
            
            # Record metrics
            self.dashboard.metrics_collector.increment_counter("forced_gc_operations")
            self.dashboard.metrics_collector.set_gauge(
                "memory_freed_mb", 
                gc_results['memory_freed_mb']
            )
            
            self.logger.info(f"Forced garbage collection: freed {gc_results['memory_freed_mb']:.1f} MB")
            return gc_results
            
        except Exception as e:
            self.logger.error(f"Error in forced garbage collection: {e}")
            return {'error': str(e)}
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary for the specified period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Alert summary statistics
        """
        try:
            return self.dashboard.get_alert_statistics(hours)
        except Exception as e:
            self.logger.error(f"Error getting alert summary: {e}")
            return {'error': str(e)}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()


# Convenience function for easy setup
def create_monitoring_system(
    health_check_interval: float = 30.0,
    source_check_interval: float = 300.0,
    memory_threshold_mb: float = 500.0,
    enable_email_alerts: bool = False,
    email_recipients: Optional[List[str]] = None
) -> IntegratedMonitoringSystem:
    """
    Create and configure an integrated monitoring system with common settings.
    
    Args:
        health_check_interval: Interval between health checks in seconds
        source_check_interval: Interval between source availability checks in seconds
        memory_threshold_mb: Memory threshold for garbage collection in MB
        enable_email_alerts: Whether to enable email alerts
        email_recipients: List of email addresses for alerts
        
    Returns:
        Configured IntegratedMonitoringSystem instance
    """
    config = IntegratedMonitoringConfig(
        health_check_interval=health_check_interval,
        source_check_interval=source_check_interval,
        gc_threshold_mb=memory_threshold_mb,
        alert_email_enabled=enable_email_alerts,
        alert_email_recipients=email_recipients or []
    )
    
    return IntegratedMonitoringSystem(config)