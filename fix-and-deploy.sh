#!/bin/bash

echo "🔧 Fixing and redeploying Wildfire Prediction System"
echo "=================================================="

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
docker compose down

# Remove old images to force rebuild
echo "🗑️ Removing old images..."
docker rmi wildfire-prediction-wildfire-api || true

# Rebuild containers
echo "🔨 Rebuilding containers..."
docker compose build --no-cache

# Start containers
echo "🚀 Starting containers..."
docker compose up -d

# Wait a moment for containers to start
echo "⏳ Waiting for containers to start..."
sleep 10

# Check container status
echo "📊 Container status:"
docker compose ps

# Check logs
echo "📋 Application logs:"
docker compose logs wildfire-api --tail=20

# Test health endpoint
echo "🩺 Testing health endpoint..."
sleep 5
curl -f http://localhost:5050/api/health || echo "❌ Health check failed"

echo ""
echo "✅ Deployment complete!"
echo "🌐 Access the application at:"
echo "   - Main app: http://localhost:5050"
echo "   - Via nginx: http://localhost:80" 
echo "   - Health check: http://localhost:5050/api/health"
