#!/bin/bash
# Quick setup verification script for Docker deployment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "🚀 Docker Setup Verification for Ignacio's Fourier Forecast"
echo "=========================================================="

# Check Docker installation
if command -v docker &> /dev/null; then
    print_step "Docker is installed"
    docker --version
else
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_step "Docker Compose is installed"
    docker-compose --version
else
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ -f ".env" ]; then
    print_step ".env file exists"
else
    print_warning ".env file not found"
    echo "Creating .env from template..."
    cp .env.docker .env
    print_warning "Please edit .env with your API keys and SMTP settings before proceeding"
    echo "Required variables to configure:"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - VOYAGE_API_KEY" 
    echo "  - LLMLAYER_API_KEY"
    echo "  - SMTP_USER, SMTP_PASSWORD"
    echo "  - RECIPIENT_EMAIL"
fi

# Check required directories
for dir in data logs deployment_reports; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_step "Created $dir directory"
    else
        print_step "$dir directory exists"
    fi
done

# Make scripts executable
chmod +x docker-run.sh
chmod +x docker/entrypoint.sh
print_step "Made scripts executable"

echo ""
echo "🎯 Next Steps:"
echo "1. Edit .env with your API keys and email settings"
echo "2. Build the image: ./docker-run.sh build"
echo "3. Test the setup: ./docker-run.sh test"
echo "4. Start the service: ./docker-run.sh start"
echo "5. Monitor logs: ./docker-run.sh logs"
echo ""
echo "📖 See DOCKER.md for complete documentation"