"""
Monitoring dashboard and reporting system for the educational PDF scraper.
Provides comprehensive monitoring interface and automated reporting capabilities.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from utils.monitoring import (
    SystemHealthMonitor, PerformanceMetricsCollector,
    SourceAvailabilityChecker, MemoryMonitor, Alert, AlertSeverity
)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    health_check_interval: float = 30.0
    source_check_interval: float = 300.0
    memory_check_interval: float = 60.0
    gc_threshold_mb: float = 500.0
    report_interval_hours: int = 24
    alert_email_enabled: bool = False
    alert_email_recipients: List[str] = None
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard that coordinates all monitoring components.
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize monitoring dashboard.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring components
        self.health_monitor = SystemHealthMonitor(config.health_check_interval)
        self.metrics_collector = PerformanceMetricsCollector()
        self.source_checker = SourceAvailabilityChecker(config.source_check_interval)
        self.memory_monitor = MemoryMonitor(config.gc_threshold_mb, config.memory_check_interval)
        
        # Alert management
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Reporting
        self.last_report_time: Optional[datetime] = None
        self._monitoring_active = False
        
        # Setup callbacks
        self._setup_monitoring_callbacks()
    
    def _setup_monitoring_callbacks(self) -> None:
        """Setup callbacks for monitoring components."""
        # Health monitor callback
        self.health_monitor.add_health_callback(self._handle_health_update)
        
        # Source availability callback
        self.source_checker.add_status_callback(self._handle_source_status_change)
    
    async def start_monitoring(self, sources: Optional[List[Any]] = None) -> None:
        """
        Start all monitoring components.
        
        Args:
            sources: Optional list of sources to monitor
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start all monitoring components
        await self.health_monitor.start_monitoring()
        await self.memory_monitor.start_monitoring()
        
        if sources:
            await self.source_checker.start_checking(sources)
        
        self.logger.info("Monitoring dashboard started")
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self._monitoring_active = False
        
        await self.health_monitor.stop_monitoring()
        await self.memory_monitor.stop_monitoring()
        await self.source_checker.stop_checking()
        
        self.logger.info("Monitoring dashboard stopped")
    
    def _handle_health_update(self, health_data: Dict[str, Any]) -> None:
        """
        Handle health update from system health monitor.
        
        Args:
            health_data: Health check results
        """
        # Process alerts from health data
        if 'alerts' in health_data:
            for alert_data in health_data['alerts']:
                alert = Alert(
                    alert_id=alert_data['alert_id'],
                    timestamp=datetime.fromisoformat(alert_data['timestamp']),
                    severity=AlertSeverity(alert_data['severity']),
                    component=alert_data['component'],
                    message=alert_data['message'],
                    metric_name=alert_data.get('metric_name'),
                    metric_value=alert_data.get('metric_value'),
                    threshold=alert_data.get('threshold'),
                    metadata=alert_data.get('metadata', {})
                )
                
                self._process_alert(alert)
        
        # Update metrics
        if 'metrics' in health_data:
            for metric_name, value in health_data['metrics'].items():
                self.metrics_collector.set_gauge(f"system_{metric_name}", value)
    
    def _handle_source_status_change(self, source_name: str, status_info: Dict[str, Any]) -> None:
        """
        Handle source status change.
        
        Args:
            source_name: Name of the source
            status_info: Status information
        """
        # Create alert for source availability changes
        if not status_info.get('available', False):
            alert = Alert(
                alert_id=f"source_unavailable_{source_name}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                component="source_availability",
                message=f"Source {source_name} is unavailable: {status_info.get('error', 'Unknown error')}",
                metadata={'source_name': source_name, 'status_info': status_info}
            )
            
            self._process_alert(alert)
        
        # Update metrics
        self.metrics_collector.set_gauge(
            "source_available",
            1.0 if status_info.get('available', False) else 0.0,
            labels={'source': source_name}
        )
        
        if 'response_time' in status_info:
            self.metrics_collector.record_timer(
                "source_response_time",
                status_info['response_time'],
                labels={'source': source_name}
            )
    
    def _process_alert(self, alert: Alert) -> None:
        """
        Process a new alert.
        
        Args:
            alert: Alert to process
        """
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Limit active alerts size
        if len(self.active_alerts) > 1000:
            self.active_alerts = self.active_alerts[-1000:]
        
        # Limit alert history size
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-10000:]
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.log(log_level, f"Alert: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Send email alert if configured and severity is high enough
        if (self.config.alert_email_enabled and 
            alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]):
            asyncio.create_task(self._send_alert_email(alert))
    
    async def _send_alert_email(self, alert: Alert) -> None:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
        """
        if not self.config.alert_email_recipients:
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.alert_email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] PDF Scraper Alert: {alert.component}"
            
            # Email body
            body = f"""
Alert Details:
- ID: {alert.alert_id}
- Timestamp: {alert.timestamp}
- Severity: {alert.severity.value.upper()}
- Component: {alert.component}
- Message: {alert.message}

Additional Information:
- Metric: {alert.metric_name or 'N/A'}
- Value: {alert.metric_value or 'N/A'}
- Threshold: {alert.threshold or 'N/A'}

This is an automated alert from the PDF Scraper monitoring system.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            
            if self.config.smtp_username and self.config.smtp_password:
                server.login(self.config.smtp_username, self.config.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert email sent for {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add callback for alert processing.
        
        Args:
            callback: Function to call when alerts are generated
        """
        self.alert_callbacks.append(callback)
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard summary.
        
        Returns:
            Dictionary containing dashboard status and metrics
        """
        # System health summary
        health_summary = self.health_monitor.get_current_health_summary()
        
        # Source availability summary
        source_summary = self.source_checker.get_availability_summary()
        
        # Memory statistics
        memory_stats = self.memory_monitor.get_memory_stats()
        
        # Alert summary
        recent_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        alert_summary = {
            'total_active_alerts': len(self.active_alerts),
            'recent_alerts_24h': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            'error_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.ERROR]),
            'warning_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING])
        }
        
        # Performance metrics summary
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        return {
            'monitoring_active': self._monitoring_active,
            'timestamp': datetime.now().isoformat(),
            'system_health': health_summary,
            'source_availability': source_summary,
            'memory_stats': memory_stats,
            'alerts': alert_summary,
            'performance_metrics': metrics_summary,
            'last_report_time': self.last_report_time.isoformat() if self.last_report_time else None
        }
    
    def get_detailed_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate detailed monitoring report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Dictionary containing detailed report
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Health history
        health_history = self.health_monitor.get_health_history(hours)
        
        # Alert history
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Source status
        source_status = self.source_checker.get_all_source_status()
        
        # Performance metrics
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'report_period_hours': hours,
            'summary': self.get_dashboard_summary(),
            'health_history': health_history,
            'alert_history': [alert.to_dict() for alert in recent_alerts],
            'source_status': source_status,
            'performance_metrics': metrics_summary
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save report to file.
        
        Args:
            report: Report data to save
            filename: Optional custom filename
            
        Returns:
            Path to saved report file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Monitoring report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            raise
    
    async def generate_and_save_report(self, hours: int = 24) -> str:
        """
        Generate and save a monitoring report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Path to saved report file
        """
        report = self.get_detailed_report(hours)
        report_path = self.save_report(report)
        
        self.last_report_time = datetime.now()
        return report_path
    
    def clear_old_alerts(self, hours: int = 168) -> int:  # Default 7 days
        """
        Clear old alerts from memory.
        
        Args:
            hours: Age threshold in hours
            
        Returns:
            Number of alerts cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        initial_count = len(self.alert_history)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        cleared_count = initial_count - len(self.alert_history)
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} old alerts")
        
        return cleared_count
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics for the specified period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing alert statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Group by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        # Group by component
        component_counts = {}
        for alert in recent_alerts:
            component_counts[alert.component] = component_counts.get(alert.component, 0) + 1
        
        # Calculate alert rate (alerts per hour)
        alert_rate = len(recent_alerts) / hours if hours > 0 else 0
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alert_rate_per_hour': alert_rate,
            'severity_breakdown': severity_counts,
            'component_breakdown': component_counts,
            'most_active_component': max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the monitoring system itself.
        
        Returns:
            Dictionary containing monitoring system health status
        """
        health_status = {
            'monitoring_dashboard': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check each monitoring component
        try:
            # Health monitor
            health_summary = self.health_monitor.get_current_health_summary()
            health_status['components']['health_monitor'] = {
                'status': 'healthy' if health_summary.get('monitoring_active') else 'inactive',
                'details': health_summary
            }
        except Exception as e:
            health_status['components']['health_monitor'] = {
                'status': 'error',
                'error': str(e)
            }
        
        try:
            # Memory monitor
            memory_stats = self.memory_monitor.get_memory_stats()
            health_status['components']['memory_monitor'] = {
                'status': 'healthy' if memory_stats.get('monitoring_active') else 'inactive',
                'details': memory_stats
            }
        except Exception as e:
            health_status['components']['memory_monitor'] = {
                'status': 'error',
                'error': str(e)
            }
        
        try:
            # Source checker
            source_summary = self.source_checker.get_availability_summary()
            health_status['components']['source_checker'] = {
                'status': 'healthy',
                'details': source_summary
            }
        except Exception as e:
            health_status['components']['source_checker'] = {
                'status': 'error',
                'error': str(e)
            }
        
        try:
            # Metrics collector
            metrics_summary = self.metrics_collector.get_all_metrics_summary()
            health_status['components']['metrics_collector'] = {
                'status': 'healthy',
                'details': {
                    'collection_stats': metrics_summary.get('collection_stats', {})
                }
            }
        except Exception as e:
            health_status['components']['metrics_collector'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Overall health determination
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if any(status == 'error' for status in component_statuses):
            health_status['monitoring_dashboard'] = 'degraded'
        elif any(status == 'inactive' for status in component_statuses):
            health_status['monitoring_dashboard'] = 'partially_active'
        
        return health_status


class AutomatedReporter:
    """
    Automated reporting system for monitoring data.
    """
    
    def __init__(self, dashboard: MonitoringDashboard, config: MonitoringConfig):
        """
        Initialize automated reporter.
        
        Args:
            dashboard: Monitoring dashboard instance
            config: Monitoring configuration
        """
        self.dashboard = dashboard
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self._reporting_active = False
        self._report_task: Optional[asyncio.Task] = None
    
    async def start_automated_reporting(self) -> None:
        """Start automated report generation."""
        if self._reporting_active:
            return
        
        self._reporting_active = True
        self._report_task = asyncio.create_task(self._reporting_loop())
        self.logger.info("Started automated reporting")
    
    async def stop_automated_reporting(self) -> None:
        """Stop automated report generation."""
        self._reporting_active = False
        
        if self._report_task:
            self._report_task.cancel()
            try:
                await self._report_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped automated reporting")
    
    async def _reporting_loop(self) -> None:
        """Main reporting loop."""
        while self._reporting_active:
            try:
                # Wait for report interval
                await asyncio.sleep(self.config.report_interval_hours * 3600)
                
                # Generate and save report
                report_path = await self.dashboard.generate_and_save_report(
                    hours=self.config.report_interval_hours
                )
                
                self.logger.info(f"Automated report generated: {report_path}")
                
                # Clean up old alerts
                self.dashboard.clear_old_alerts()
                
            except Exception as e:
                self.logger.error(f"Error in automated reporting loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying