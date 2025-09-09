#!/bin/bash
set -e

echo "🤖 Setting up AURA development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install dependencies
echo "Installing Python dependencies..."
uv sync

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p logs data cache

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "⚠️  Docker is not running. Please start Docker to use containerized features."
else
    echo "✅ Docker is running"
fi

# Check if Redis is available
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is not running. Starting with Docker Compose is recommended."
    fi
else
    echo "⚠️  Redis CLI not found. Starting with Docker Compose is recommended."
fi

# Check if Ollama is available
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    echo "Pulling recommended models..."
    ollama pull llama3.1:8b
    ollama pull mixtral:8x7b
else
    echo "⚠️  Ollama not found. Please install Ollama or use Docker Compose."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Start services: docker-compose up -d"
echo "3. Run AURA: uv run python -m aura.main"
echo ""
echo "For development:"
echo "- Run tests: uv run pytest"
echo "- Format code: uv run ruff format ."
echo "- Type check: uv run mypy aura/"