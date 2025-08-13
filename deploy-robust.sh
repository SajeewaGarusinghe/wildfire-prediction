#!/bin/bash

echo "🔥 Robust Wildfire Prediction System Deployment"
echo "=============================================="

# Function to test if a container is healthy
test_health() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "🩺 Health check attempt $attempt/$max_attempts..."
        
        if curl -f http://localhost:5050/api/health >/dev/null 2>&1; then
            echo "✅ Health check passed!"
            return 0
        fi
        
        sleep 2
        ((attempt++))
    done
    
    echo "❌ Health check failed after $max_attempts attempts"
    return 1
}

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null || true

# Try simple build first
echo "🚀 Attempting simple build (Dockerfile.simple)..."
if docker compose build --no-cache; then
    echo "✅ Simple build successful!"
    
    echo "🚀 Starting containers..."
    docker compose up -d
    
    echo "⏳ Waiting for application to start..."
    sleep 15
    
    if test_health; then
        echo "🎉 Deployment successful with simple build!"
        docker compose ps
        echo ""
        echo "🌐 Application is available at:"
        echo "   - Main app: http://localhost:5050"
        echo "   - Health: http://localhost:5050/api/health"
        echo "   - Dashboard: http://localhost:5050/"
        exit 0
    else
        echo "❌ Simple build containers not healthy"
        echo "📋 Container logs:"
        docker compose logs wildfire-api --tail=50
    fi
else
    echo "❌ Simple build failed"
fi

# If simple build fails, try minimal build
echo ""
echo "🔄 Trying minimal build (Dockerfile.minimal)..."

# Update docker-compose to use minimal dockerfile
sed -i 's/dockerfile: Dockerfile.simple/dockerfile: Dockerfile.minimal/' docker-compose.yml

docker compose down 2>/dev/null || true

if docker compose build --no-cache; then
    echo "✅ Minimal build successful!"
    
    echo "🚀 Starting containers..."
    docker compose up -d
    
    echo "⏳ Waiting for application to start..."
    sleep 15
    
    if test_health; then
        echo "🎉 Deployment successful with minimal build!"
        docker compose ps
        echo ""
        echo "🌐 Application is available at:"
        echo "   - Main app: http://localhost:5050"
        echo "   - Health: http://localhost:5050/api/health"
        echo "   - Dashboard: http://localhost:5050/"
        exit 0
    else
        echo "❌ Minimal build containers not healthy"
        echo "📋 Container logs:"
        docker compose logs wildfire-api --tail=50
    fi
else
    echo "❌ Minimal build also failed"
fi

# If both fail, try without nginx
echo ""
echo "🔄 Trying direct deployment without nginx..."

cat > docker-compose-simple.yml << EOF
services:
  wildfire-api:
    build:
      context: .
      dockerfile: Dockerfile.simple
    ports:
      - "5050:5050"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PYTHONPATH=/app
      - HOST=0.0.0.0
      - PORT=5050
    restart: unless-stopped

volumes:
  wildfire_data:
EOF

docker compose -f docker-compose-simple.yml down 2>/dev/null || true

if docker compose -f docker-compose-simple.yml build --no-cache; then
    echo "✅ Direct build successful!"
    
    echo "🚀 Starting container..."
    docker compose -f docker-compose-simple.yml up -d
    
    echo "⏳ Waiting for application to start..."
    sleep 15
    
    if test_health; then
        echo "🎉 Deployment successful with direct deployment!"
        docker compose -f docker-compose-simple.yml ps
        echo ""
        echo "🌐 Application is available at:"
        echo "   - Main app: http://localhost:5050"
        echo "   - Health: http://localhost:5050/api/health"
        echo "   - Dashboard: http://localhost:5050/"
        exit 0
    else
        echo "❌ Direct deployment containers not healthy"
        echo "📋 Container logs:"
        docker compose -f docker-compose-simple.yml logs wildfire-api --tail=50
    fi
else
    echo "❌ All build strategies failed"
fi

echo ""
echo "💡 Alternative: Try running locally without Docker:"
echo "   pip install -r requirements-local.txt"
echo "   python app.py"
echo ""
echo "🔍 For debugging, check the logs above or run:"
echo "   docker compose logs wildfire-api"

exit 1



