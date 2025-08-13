#!/bin/bash

# Azure VM Deployment Script for Wildfire Risk Prediction System
# Run this script on your Azure VM (172.214.136.108)

echo "🔥 Wildfire Risk Prediction System - Azure Deployment"
echo "=================================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt update
    sudo apt install -y docker-ce
    sudo usermod -aG docker $USER
    echo "✅ Docker installed successfully"
else
    echo "✅ Docker is already installed"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed successfully"
else
    echo "✅ Docker Compose is already installed"
fi

# Create project directory
PROJECT_DIR="/opt/wildfire-prediction"
echo "📁 Creating project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR
cd $PROJECT_DIR

# Clone or update repository (if you have the code in a git repo)
# echo "📥 Cloning repository..."
# git clone https://github.com/your-username/wildfire-prediction.git .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/{raw,processed} models logs ssl static templates src notebooks

# Set up firewall rules
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 5050/tcp  # Direct app access (optional)
sudo ufw --force enable

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# Production environment for Azure VM
FLASK_ENV=production
FLASK_DEBUG=False
HOST=0.0.0.0
PORT=5050

# Database
DATABASE_PATH=/app/data/wildfire_predictions.db

# API Configuration
HIGH_RISK_THRESHOLD=0.8
MODERATE_RISK_THRESHOLD=0.6

# Logging
LOG_LEVEL=INFO

# Add your API keys here
NOAA_API_KEY=your_noaa_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
EOF

echo "🔧 Environment file created. Please edit .env to add your API keys:"
echo "   nano .env"

# Build and start the application
echo "🚀 Building and starting the application..."
# Uncomment the following lines after you've copied your code to the VM
# docker-compose build
# docker-compose up -d

echo ""
echo "✅ Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your project files to: $PROJECT_DIR"
echo "2. Edit .env file with your API keys: nano $PROJECT_DIR/.env"
echo "3. Build and start: cd $PROJECT_DIR && docker-compose up -d"
echo "4. Check status: docker-compose ps"
echo "5. View logs: docker-compose logs -f"
echo ""
echo "🌐 Your application will be accessible at:"
echo "   - HTTP: http://172.214.136.108"
echo "   - Direct API: http://172.214.136.108:5050"
echo "   - Health check: http://172.214.136.108/api/health"
echo ""
echo "🔒 Security notes:"
echo "   - Firewall is configured for ports 22, 80, 443, 5050"
echo "   - Consider setting up SSL certificates for HTTPS"
echo "   - Change default passwords and API keys"
echo ""



