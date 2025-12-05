#!/bin/bash

# Quick deployment helper script
# Usage: ./deploy.sh [platform]

set -e

PLATFORM=${1:-help}

echo "🚀 EPFL Smart Kitchen - Deployment Helper"
echo "=========================================="
echo ""

case $PLATFORM in
  heroku)
    echo "📦 Deploying to Heroku..."
    echo ""
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "❌ Heroku CLI not found. Install from: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Check if app exists
    if ! heroku apps:info &> /dev/null; then
        echo "Creating new Heroku app..."
        heroku create
    fi
    
    echo "Deploying to Heroku..."
    git push heroku main
    
    echo ""
    echo "✅ Deployed to Heroku!"
    echo "🌐 Opening app..."
    heroku open
    ;;
    
  vercel)
    echo "⚡ Deploying to Vercel..."
    echo ""
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        echo "Installing Vercel CLI..."
        npm i -g vercel
    fi
    
    echo "Deploying to Vercel..."
    vercel --prod
    
    echo ""
    echo "✅ Deployed to Vercel!"
    ;;
    
  # docker and docker-compose support removed
    
  render)
    echo "🎨 Deploying to Render..."
    echo ""
    echo "Render deployment is done through their web interface:"
    echo ""
    echo "1. Go to https://render.com"
    echo "2. Connect your GitHub repository"
    echo "3. Select 'website_node' directory"
    echo "4. Click 'Deploy'"
    echo ""
    echo "Or use the render.yaml file for automatic configuration!"
    echo ""
    echo "📚 See DEPLOYMENT.md for detailed instructions"
    ;;
    
  railway)
    echo "🚂 Deploying to Railway..."
    echo ""
    echo "Railway deployment is done through their web interface:"
    echo ""
    echo "1. Go to https://railway.app"
    echo "2. Connect your GitHub repository"
    echo "3. Select 'website_node' directory"
    echo "4. Deploy automatically"
    echo ""
    echo "📚 See DEPLOYMENT.md for detailed instructions"
    ;;
    
  test)
    echo "🧪 Testing locally..."
    echo ""
    
    # Make sure dependencies are installed
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install
    fi
    
    # Make sure assets are copied
    echo "Copying assets..."
    npm run build
    
    echo "Starting development server..."
    npm run dev
    ;;
    
  help|*)
    echo "Usage: ./deploy.sh [platform]"
    echo ""
  echo "Available platforms:"
  echo "  heroku          - Deploy to Heroku"
  echo "  vercel          - Deploy to Vercel"
  echo "  render          - Show Render deployment instructions"
  echo "  railway         - Show Railway deployment instructions"
  echo "  test            - Test locally"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh test              # Test locally"
    echo "  ./deploy.sh vercel            # Deploy to Vercel"
  echo "  ./deploy.sh vercel            # Deploy to Vercel"
    echo ""
    echo "📚 For detailed deployment guides, see DEPLOYMENT.md"
    ;;
esac
