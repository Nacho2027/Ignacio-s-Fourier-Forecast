import os
import sys
import logging
import json
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Callable, Deque, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import pytz
import uuid


@dataclass
class ErrorContext:
    """Context for an error occurrence"""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    service: str
    operation: str
    severity: str
    user_impact: Optional[str] = None
    recovery_action: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ServiceMetric:
    """Metric for a service operation"""
    service: str
    operation: str
    timestamp: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ServiceType(Enum):
    """Service classifications"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"


class CriticalError(Exception):
    """Exception for critical errors that stop pipeline"""
    pass


class NonCriticalError(Exception):
    """Exception for non-critical errors that allow continuation"""
    pass


class ErrorHandler:
    """
    Intelligent error handling with NO FALLBACKS philosophy.
    """

    def __init__(self) -> None:
        self.service_criticality: Dict[str, ServiceType] = {
            'anthropic': ServiceType.CRITICAL,
            'voyage': ServiceType.CRITICAL,
            'ai_service': ServiceType.CRITICAL,
            'email': ServiceType.CRITICAL,
            'cache': ServiceType.IMPORTANT,
            'deduplication': ServiceType.IMPORTANT,
            'arxiv': ServiceType.OPTIONAL,
            'rss': ServiceType.OPTIONAL,
            'historical': ServiceType.OPTIONAL,
        }

        self.error_history: Deque[ErrorContext] = deque(maxlen=100)
        self.error_counts: Dict[str, int] = defaultdict(int)

        self.recovery_strategies: Dict[str, Callable[[Exception], str]] = {
            'ConnectionError': self._suggest_connection_recovery,
            'RateLimitError': self._suggest_rate_limit_recovery,
            'AuthenticationError': self._suggest_auth_recovery,
            'TimeoutError': self._suggest_timeout_recovery,
        }

        self.logger = logging.getLogger(__name__)

    def handle_error(
        self,
        error: Exception,
        service: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        timestamp = datetime.now()

        severity = self.classify_severity(error, service)

        recovery_action = self.get_recovery_suggestion(error)
        error_context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=timestamp,
            service=service,
            operation=operation,
            severity=severity.value,
            recovery_action=recovery_action,
            metadata=context or {},
        )

        self.error_history.append(error_context)
        self.error_counts[error_type] += 1

        # Structured log for error
        self.logger.error(json.dumps({
            'event': 'error',
            'service': service,
            'operation': operation,
            'severity': severity.value,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': timestamp.isoformat(),
        }))

        if self.should_stop_pipeline(error_context):
            raise CriticalError(f"Critical error in {service}:{operation} - {error_message}")

        return error_context

    def classify_severity(self, error: Exception, service: str) -> ErrorSeverity:
        error_name = type(error).__name__
        message_lower = str(error).lower()
        service_type = self.service_criticality.get(service, ServiceType.OPTIONAL)

        # Authentication problems are serious
        if 'auth' in message_lower or 'unauthorized' in message_lower or 'invalid api key' in message_lower:
            return ErrorSeverity.CRITICAL if service_type == ServiceType.CRITICAL else ErrorSeverity.HIGH

        if error_name in ('ConnectionError', 'HTTPError'):
            if service_type == ServiceType.CRITICAL:
                return ErrorSeverity.CRITICAL
            if service_type == ServiceType.IMPORTANT:
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.LOW

        if error_name in ('TimeoutError',):
            if service_type == ServiceType.CRITICAL:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM if service_type == ServiceType.IMPORTANT else ErrorSeverity.LOW

        if 'rate limit' in message_lower or 'too many requests' in message_lower:
            return ErrorSeverity.HIGH if service_type != ServiceType.OPTIONAL else ErrorSeverity.MEDIUM

        # Default based on service type
        if service_type == ServiceType.CRITICAL:
            return ErrorSeverity.HIGH
        if service_type == ServiceType.IMPORTANT:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW

    def should_stop_pipeline(self, error_context: ErrorContext) -> bool:
        return error_context.severity == ErrorSeverity.CRITICAL.value

    def get_recovery_suggestion(self, error: Exception) -> Optional[str]:
        name = type(error).__name__
        # Keyword-based mapping for generic Exceptions
        msg = str(error).lower()
        if 'rate' in msg and 'limit' in msg:
            return self._suggest_rate_limit_recovery(error)
        if 'timeout' in msg:
            return self._suggest_timeout_recovery(error)
        if 'auth' in msg or 'api key' in msg or 'unauthorized' in msg:
            return self._suggest_auth_recovery(error)
        if name in self.recovery_strategies:
            return self.recovery_strategies[name](error)
        return None

    def _suggest_connection_recovery(self, error: Exception) -> str:
        return (
            "Check network connectivity, DNS resolution, and provider status page. "
            "Verify base URLs and retry after a short delay."
        )

    def _suggest_rate_limit_recovery(self, error: Exception) -> str:
        return (
            "Rate limit encountered. Reduce request frequency, implement backoff, "
            "and ensure API plan limits are sufficient."
        )

    def _suggest_auth_recovery(self, error: Exception) -> str:
        return (
            "Authentication failure. Verify API keys/tokens, scopes, and environment variables. "
            "Rotate credentials if necessary."
        )

    def _suggest_timeout_recovery(self, error: Exception) -> str:
        return (
            "Operation timed out. Increase timeouts where safe, optimize request payloads, "
            "and check provider latency status."
        )

    def detect_error_patterns(self) -> List[str]:
        patterns: List[str] = []
        if not self.error_history:
            return patterns

        # Count repeated (error_type, service)
        tuple_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for ctx in self.error_history:
            tuple_counts[(ctx.error_type, ctx.service)] += 1

        for (etype, service), count in tuple_counts.items():
            if count >= 3:
                patterns.append(
                    f"Repeated pattern: {etype} in {service} occurred {count} times recently"
                )

        return patterns

    def get_error_statistics(self) -> Dict[str, Any]:
        total = sum(self.error_counts.values())
        return {
            'total_errors': total,
            'error_types': dict(self.error_counts),
        }


class MonitoringService:
    """Comprehensive monitoring and observability service."""

    def __init__(self, log_dir: str = "logs", metrics_dir: str = "metrics"):
        self.log_dir = log_dir
        self.metrics_dir = metrics_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        self._setup_logging()

        self.metrics: Deque[ServiceMetric] = deque(maxlen=10000)
        self.operation_timings: Dict[str, List[float]] = defaultdict(list)
        self._operations: Dict[str, Dict[str, Any]] = {}

        self.alert_email: Optional[str] = os.getenv('ALERT_EMAIL')
        self.alert_thresholds: Dict[str, float] = {
            'pipeline_duration_seconds': 300,
            'error_rate_percent': 10,
            'api_response_ms': 5000,
        }

        self.et_timezone = pytz.timezone('America/New_York')
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self) -> None:
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s %(name)s %(message)s'
            )

        # File handler for structured logs
        file_path = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))

        logger = logging.getLogger(__name__)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

    def log_operation_start(self, service: str, operation: str, context: Optional[Dict[str, Any]] = None) -> str:
        operation_id = str(uuid.uuid4())
        self._operations[operation_id] = {
            'service': service,
            'operation': operation,
            'start': datetime.now(),
            'context': context or {},
        }
        self.log_structured('INFO', 'operation_start', {
            'operation_id': operation_id,
            'service': service,
            'operation': operation,
            'context': context or {},
        })
        return operation_id

    def log_operation_end(
        self,
        operation_id: str,
        success: bool,
        duration_ms: float,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        op_info = self._operations.pop(operation_id, None)
        service = op_info['service'] if op_info else 'unknown'
        operation = op_info['operation'] if op_info else 'unknown'

        self.operation_timings[operation].append(duration_ms)
        metric = ServiceMetric(
            service=service,
            operation=operation,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self.metrics.append(metric)

        self.log_structured('INFO' if success else 'ERROR', 'operation_end', {
            'operation_id': operation_id,
            'service': service,
            'operation': operation,
            'success': success,
            'duration_ms': duration_ms,
            'error': error,
            'result': result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result),
        })

    async def track_async_operation(self, service: str, operation: str, func: Callable, *args, **kwargs) -> Any:
        start = datetime.now()
        op_id = self.log_operation_start(service, operation, {'args': str(args), 'kwargs': str(kwargs)})
        try:
            result = await func(*args, **kwargs)
            duration_ms = (datetime.now() - start).total_seconds() * 1000.0
            self.log_operation_end(op_id, True, duration_ms, result=result)
            return result
        except Exception as e:
            duration_ms = (datetime.now() - start).total_seconds() * 1000.0
            self.log_operation_end(op_id, False, duration_ms, error=str(e))
            raise

    def record_metric(self, metric: ServiceMetric) -> None:
        self.metrics.append(metric)
        self.operation_timings[metric.operation].append(metric.duration_ms)

    def log_structured(self, level: str, message: str, context: Dict[str, Any]) -> None:
        payload = {'message': message, **context}
        if level.upper() == 'ERROR':
            self.logger.error(json.dumps(payload))
        elif level.upper() == 'WARNING':
            self.logger.warning(json.dumps(payload))
        else:
            self.logger.info(json.dumps(payload))

    async def check_thresholds(self) -> List[str]:
        violations: List[str] = []
        # API response latency
        threshold_ms = self.alert_thresholds.get('api_response_ms', 5000)
        for op, timings in self.operation_timings.items():
            if not timings:
                continue
            max_ms = max(timings)
            if max_ms > threshold_ms:
                violations.append(f"Operation {op} exceeded latency threshold: {max_ms:.2f}ms > {threshold_ms}ms")
        return violations

    async def send_alert(self, subject: str, message: str, severity: ErrorSeverity) -> bool:
        if not self.alert_email:
            return False
        try:
            email_from = os.getenv('ALERT_FROM_EMAIL', 'alerts@example.com')
            email_to = self.alert_email
            mime = MIMEMultipart()
            mime['From'] = email_from
            mime['To'] = email_to
            mime['Subject'] = f"[{severity.value.upper()}] {subject}"
            mime.attach(MIMEText(message, 'plain'))

            smtp_server = os.getenv('SMTP_SERVER', 'localhost')
            smtp_port = int(os.getenv('SMTP_PORT', '25'))
            smtp_timeout = float(os.getenv('SMTP_TIMEOUT', '10'))
            with smtplib.SMTP(smtp_server, smtp_port, timeout=smtp_timeout) as server:
                server.send_message(mime)
            return True
        except Exception:
            # Log but do not raise in alerting to avoid masking root cause
            self.logger.error(json.dumps({'event': 'alert_send_failed', 'subject': subject}))
            return False

    def get_performance_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        cutoff = datetime.now() - time_window
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff]
        total = len(recent_metrics)
        successes = sum(1 for m in recent_metrics if m.success)
        success_rate = (successes / total * 100.0) if total else 0.0
        services: Dict[str, Dict[str, int]] = defaultdict(lambda: {'count': 0, 'errors': 0})
        for m in recent_metrics:
            services[m.service]['count'] += 1
            if not m.success:
                services[m.service]['errors'] += 1
        return {
            'total_operations': total,
            'success_rate': round(success_rate, 2),
            'services': {k: dict(v) for k, v in services.items()},
        }

    def get_error_rate(self, service: Optional[str] = None, time_window: timedelta = timedelta(hours=1)) -> float:
        cutoff = datetime.now() - time_window
        filtered = [m for m in self.metrics if m.timestamp >= cutoff and (service is None or m.service == service)]
        if not filtered:
            return 0.0
        errors = sum(1 for m in filtered if not m.success)
        return round(errors / len(filtered) * 100.0, 2)

    def get_operation_percentiles(self, operation: str, percentiles: List[int] = [50, 90, 95, 99]) -> Dict[int, float]:
        values = self.operation_timings.get(operation, [])
        if not values:
            return {p: 0.0 for p in percentiles}
        data = sorted(values)
        n = len(data)

        def percentile(pct: float) -> float:
            if n == 1:
                return float(data[0])
            pos = (n - 1) * (pct / 100.0)
            lower = int(pos)
            upper = min(lower + 1, n - 1)
            weight = pos - lower
            return data[lower] * (1 - weight) + data[upper] * weight

        return {p: float(percentile(p)) for p in percentiles}

    async def generate_daily_report(self) -> str:
        summary = self.get_performance_summary(time_window=timedelta(hours=24))
        lines = [
            "Daily Monitoring Report",
            f"Total operations: {summary['total_operations']}",
            f"Success rate: {summary['success_rate']}%",
            "Services:",
        ]
        for service, stats in summary['services'].items():
            lines.append(f" - {service}: {stats['count']} ops, {stats['errors']} errors")
        return "\n".join(lines)

    def export_metrics(self, format: str = "json") -> str:
        if format.lower() == 'csv':
            # CSV header
            lines = ["service,operation,timestamp,duration_ms,success,error"]
            for m in self.metrics:
                ts = m.timestamp.isoformat()
                err = (m.error or '').replace('\n', ' ').replace(',', ';')
                lines.append(f"{m.service},{m.operation},{ts},{m.duration_ms:.2f},{int(m.success)},{err}")
            return "\n".join(lines)
        # Default to JSON
        return json.dumps([{
            'service': m.service,
            'operation': m.operation,
            'timestamp': m.timestamp.isoformat(),
            'duration_ms': m.duration_ms,
            'success': m.success,
            'error': m.error,
            'metadata': m.metadata,
        } for m in self.metrics])


class CircuitBreaker:
    """Circuit breaker for service protection."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: timedelta = timedelta(minutes=5)):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.circuit_states: Dict[str, Tuple[str, datetime]] = {}

    def record_success(self, service: str) -> None:
        self.failure_counts[service] = 0
        self.circuit_states[service] = ('closed', datetime.now())

    def record_failure(self, service: str) -> None:
        self.failure_counts[service] += 1
        if self.failure_counts[service] >= self.failure_threshold:
            self.circuit_states[service] = ('open', datetime.now())

    def is_open(self, service: str) -> bool:
        state = self.circuit_states.get(service)
        if not state:
            return False
        status, ts = state
        if status != 'open':
            return False
        if datetime.now() - ts >= self.recovery_timeout:
            # Auto-close after timeout
            self.circuit_states[service] = ('closed', datetime.now())
            self.failure_counts[service] = 0
            return False
        return True

    def should_attempt(self, service: str) -> bool:
        return not self.is_open(service)


