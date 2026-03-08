#!/bin/bash
# Flask Server Setup Script

echo "🎤 Voice Health Analysis API - Setup"
echo "===================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
fi

# Create models directory
echo ""
echo "Creating models directory..."
mkdir -p models

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the server: python app.py"
echo ""
echo "Or use the production server:"
echo "gunicorn -w 4 -b 0.0.0.0:5000 'app:create_app()'"
