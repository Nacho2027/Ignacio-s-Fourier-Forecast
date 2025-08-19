#!/bin/bash
set -e

# Entrypoint script for Ignacio's Fourier Forecast Newsletter Service
# Handles container initialization, health checks, and graceful startup

echo "🚀 Starting Ignacio's Fourier Forecast Newsletter Service"
echo "=================================================="

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if required environment variables are set
check_environment() {
    log "🔍 Checking environment configuration..."
    
    required_vars=(
        "ANTHROPIC_API_KEY"
        "VOYAGE_API_KEY" 
        "LLMLAYER_API_KEY"
        "SMTP_HOST"
        "SMTP_PORT"
        "SMTP_USER"
        "SMTP_PASSWORD"
        "RECIPIENT_EMAIL"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log "❌ Missing required environment variables:"
        printf '   - %s\n' "${missing_vars[@]}"
        log "Please check your .env file and docker-compose configuration"
        exit 1
    fi
    
    log "✅ All required environment variables are set"
}

# Function to initialize application directories
initialize_directories() {
    log "📁 Initializing application directories..."
    
    # Ensure directories exist with proper permissions
    mkdir -p /app/data /app/logs /app/deployment_reports
    
    # Check if directories are writable
    if [ ! -w /app/data ]; then
        log "❌ Data directory is not writable"
        exit 1
    fi
    
    if [ ! -w /app/logs ]; then
        log "❌ Logs directory is not writable"  
        exit 1
    fi
    
    log "✅ Application directories initialized"
}

# Function to run health check
health_check() {
    log "🔧 Running initial health check..."
    
    # Test database connectivity and service initialization
    if python src/main.py --health > /dev/null 2>&1; then
        log "✅ Health check passed"
        return 0
    else
        log "❌ Health check failed"
        return 1
    fi
}

# Function to display startup information
show_startup_info() {
    log "📊 Container startup information:"
    echo "   User: $(whoami)"
    echo "   Working Directory: $(pwd)"
    echo "   Timezone: ${TZ:-UTC}"
    echo "   Python Version: $(python --version)"
    echo "   Current Time: $(date)"
    
    # Show next scheduled run time
    log "📅 Checking next scheduled execution..."
    if python -c "
import sys
sys.path.insert(0, '/app')
from src.main import MainPipeline
from datetime import datetime
from zoneinfo import ZoneInfo

pipeline = MainPipeline()
next_run = pipeline._calculate_next_run_time()
et_tz = ZoneInfo('America/New_York')
now = datetime.now(et_tz)

print(f'   Current time (ET): {now.strftime(\"%Y-%m-%d %H:%M:%S %Z\")}')
print(f'   Next execution: {next_run.strftime(\"%Y-%m-%d %H:%M:%S %Z\")}')
print(f'   Time until next run: {next_run - now}')
"; then
        log "✅ Scheduling information displayed"
    else
        log "⚠️  Could not determine next run time"
    fi
}

# Function to handle graceful shutdown
cleanup() {
    log "📤 Received shutdown signal, cleaning up..."
    if [ ! -z "$MAIN_PID" ]; then
        log "Stopping main process (PID: $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    log "✅ Cleanup completed"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main initialization sequence
main() {
    log "🔄 Starting initialization sequence..."
    
    # Step 1: Check environment
    check_environment
    
    # Step 2: Initialize directories  
    initialize_directories
    
    # Step 3: Run health check
    if ! health_check; then
        log "❌ Initial health check failed, exiting"
        exit 1
    fi
    
    # Step 4: Show startup information
    show_startup_info
    
    log "✅ Initialization completed successfully"
    log "🎯 Starting newsletter service..."
    
    # Execute the main command
    exec "$@" &
    MAIN_PID=$!
    
    # Wait for the main process
    wait $MAIN_PID
}

# Handle special cases for testing and one-off commands
if [ "$1" = "python" ] && [[ "$*" == *"--test"* || "$*" == *"--once"* || "$*" == *"--health"* ]]; then
    log "🧪 Running in test/one-off mode, skipping full initialization"
    check_environment
    initialize_directories
    exec "$@"
fi

# Run main initialization and start service
main "$@"