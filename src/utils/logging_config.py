"""
Centralized logging configuration for Fourier Forecast pipeline.

This module provides consistent, structured logging across all services with:
- Color-coded console output for development
- File logging for production debugging
- Structured JSON logging for analysis
- Performance tracking and error reporting
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs for analysis."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
            
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better development experience."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format: [TIMESTAMP] LEVEL [MODULE] MESSAGE
        formatted = f"{color}[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:8} [{record.name:20}] {record.getMessage()}{reset}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_structured_logging: bool = False
) -> None:
    """
    Configure centralized logging for the Fourier Forecast pipeline.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to 'logs')
        enable_file_logging: Whether to write logs to files
        enable_structured_logging: Whether to use JSON structured logging
    """
    # Create logs directory
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if enable_structured_logging:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColoredConsoleFormatter())
    
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Daily rotating file handler for all logs
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "fourier_forecast.log",
            when='midnight',
            backupCount=7,
            encoding='utf-8'
        )
        daily_handler.setLevel(logging.DEBUG)
        
        if enable_structured_logging:
            daily_handler.setFormatter(StructuredFormatter())
        else:
            daily_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
            ))
        
        root_logger.addHandler(daily_handler)
        
        # Separate error log for critical issues
        error_handler = logging.FileHandler(
            log_dir / "errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)s:%(lineno)d | %(message)s\n%(exc_text)s'
        ))
        
        root_logger.addHandler(error_handler)
    
    # Configure specific loggers for pipeline components
    configure_pipeline_loggers(log_level)


def configure_pipeline_loggers(log_level: str) -> None:
    """Configure specific loggers for different pipeline components."""
    
    # AI Service - detailed GPT-5 interactions
    ai_logger = logging.getLogger('src.services.ai_service')
    ai_logger.setLevel(logging.DEBUG)
    
    # Synthesis Service - golden thread and surprise generation
    synthesis_logger = logging.getLogger('src.services.synthesis_service')
    synthesis_logger.setLevel(logging.DEBUG)
    
    # Content Aggregator - LLMLayer and RSS fetching
    content_logger = logging.getLogger('src.pipeline.content_aggregator')
    content_logger.setLevel(logging.INFO)
    
    # Email Compiler - final newsletter assembly
    email_logger = logging.getLogger('src.pipeline.email_compiler')
    email_logger.setLevel(logging.INFO)
    
    # Summarization Service - content processing
    summary_logger = logging.getLogger('src.services.summarization_service')
    summary_logger.setLevel(logging.INFO)
    
    # Pipeline Runner - orchestration
    pipeline_logger = logging.getLogger('src.pipeline.pipeline_runner')
    pipeline_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# Performance tracking utilities
class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"⏱️ Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.now() - self.start_time
            duration_ms = duration.total_seconds() * 1000
            
            if exc_type:
                self.logger.error(f"💥 Failed: {self.operation_name} ({duration_ms:.1f}ms) - {exc_val}")
            else:
                self.logger.info(f"✅ Completed: {self.operation_name} ({duration_ms:.1f}ms)")


# Structured logging helpers
def log_pipeline_metrics(
    logger: logging.Logger,
    stage: str,
    input_count: int,
    output_count: int,
    duration_ms: float,
    **extra_data
):
    """Log structured pipeline metrics for analysis."""
    metrics = {
        'stage': stage,
        'input_count': input_count,
        'output_count': output_count,
        'duration_ms': duration_ms,
        'reduction_rate': (input_count - output_count) / input_count if input_count > 0 else 0,
        **extra_data
    }
    
    # Create log record with extra data
    logger.info(f"📊 {stage}: {input_count} → {output_count} ({duration_ms:.1f}ms)", extra={'extra_data': metrics})


def log_ai_interaction(
    logger: logging.Logger,
    prompt_key: str,
    model: str,
    tokens_used: int,
    response_time_ms: float,
    success: bool,
    **extra_data
):
    """Log AI service interactions for monitoring and analysis."""
    interaction = {
        'prompt_key': prompt_key,
        'model': model,
        'tokens_used': tokens_used,
        'response_time_ms': response_time_ms,
        'success': success,
        **extra_data
    }
    
    status = "✅" if success else "❌"
    logger.info(
        f"{status} AI: {prompt_key} | {model} | {tokens_used} tokens | {response_time_ms:.1f}ms",
        extra={'extra_data': interaction}
    )


# Initialize logging on module import
if __name__ != '__main__':
    # Default logging setup for development
    setup_logging(
        log_level="DEBUG",
        enable_file_logging=True,
        enable_structured_logging=False
    )