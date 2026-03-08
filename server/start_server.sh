#!/bin/bash
# Quick start script for Flask server

echo "🎤 Starting Voice Health Analysis Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Running setup..."
    ./setup.sh
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed. Please run ./setup.sh manually."
        exit 1
    fi
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "✏️  Please edit .env and add your API keys:"
    echo "   - GOOGLE_API_KEY (for Gemini LLM)"
    echo "   - or ANTHROPIC_API_KEY (for Claude LLM)"
    echo ""
fi

# Start server
echo "🚀 Starting Flask server..."
echo "   Server will be available at: http://localhost:5000"
echo "   API base URL: http://localhost:5000/api/v1"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

python app.py
