#!/bin/bash
# Docker management script for Ignacio's Fourier Forecast Newsletter Service
# Provides easy commands for common Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}🚀 Ignacio's Fourier Forecast - Docker Manager${NC}"
    echo "=================================================="
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_error ".env file not found!"
        print_warning "Copy .env.docker to .env and configure your settings:"
        echo "  cp .env.docker .env"
        echo "  # Edit .env with your API keys and SMTP settings"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  start         Start the newsletter service"
    echo "  stop          Stop the newsletter service"
    echo "  restart       Restart the newsletter service" 
    echo "  status        Show service status"
    echo "  logs          Show service logs (follow mode)"
    echo "  logs-tail     Show last 50 lines of logs"
    echo "  shell         Open shell in running container"
    echo "  test          Run test pipeline"
    echo "  health        Run health check"
    echo "  run-once      Execute newsletter pipeline once"
    echo "  clean         Stop and remove containers/images"
    echo "  rebuild       Clean build and restart"
    echo ""
    echo "Examples:"
    echo "  $0 build                # Build the image"
    echo "  $0 start                # Start service in background"
    echo "  $0 logs                 # Follow logs"
    echo "  $0 test                 # Test configuration"
    echo "  $0 run-once             # Generate newsletter now"
}

# Main command processing
case "$1" in
    "build")
        print_header
        check_env_file
        print_status "Building Docker image..."
        docker-compose build --no-cache
        print_status "✅ Build completed"
        ;;
        
    "start")
        print_header
        check_env_file
        print_status "Starting newsletter service..."
        docker-compose up -d
        sleep 3
        docker-compose ps newsletter
        print_status "✅ Service started. Use '$0 logs' to monitor."
        ;;
        
    "stop")
        print_header
        print_status "Stopping newsletter service..."
        docker-compose down
        print_status "✅ Service stopped"
        ;;
        
    "restart")
        print_header
        check_env_file
        print_status "Restarting newsletter service..."
        docker-compose restart newsletter
        sleep 3
        docker-compose ps newsletter
        print_status "✅ Service restarted"
        ;;
        
    "status")
        print_header
        echo "Service Status:"
        docker-compose ps newsletter
        echo ""
        echo "Container Health:"
        docker-compose exec newsletter python src/main.py --health 2>/dev/null || print_warning "Service not running or health check failed"
        ;;
        
    "logs")
        print_header
        print_status "Following logs (Ctrl+C to exit)..."
        docker-compose logs -f newsletter
        ;;
        
    "logs-tail")
        print_header
        print_status "Last 50 lines of logs:"
        docker-compose logs --tail=50 newsletter
        ;;
        
    "shell")
        print_header
        print_status "Opening shell in container..."
        docker-compose exec newsletter /bin/bash
        ;;
        
    "test")
        print_header
        check_env_file
        print_status "Running test pipeline..."
        docker-compose run --rm newsletter python src/main.py --test
        ;;
        
    "health")
        print_header
        check_env_file
        print_status "Running health check..."
        docker-compose run --rm newsletter python src/main.py --health
        ;;
        
    "run-once")
        print_header
        check_env_file
        print_status "Running newsletter pipeline once..."
        docker-compose run --rm newsletter python src/main.py --once
        ;;
        
    "clean")
        print_header
        print_warning "This will stop and remove all containers and images"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Cleaning up..."
            docker-compose down --rmi all --volumes --remove-orphans
            print_status "✅ Cleanup completed"
        else
            print_status "Cleanup cancelled"
        fi
        ;;
        
    "rebuild")
        print_header
        check_env_file
        print_status "Rebuilding and restarting service..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        sleep 3
        docker-compose ps newsletter
        print_status "✅ Rebuild and restart completed"
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac