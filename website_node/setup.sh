#!/bin/bash

# EPFL Smart Kitchen - Node.js Website Setup Script
# This script sets up the Node.js version of the website

set -e

echo "🚀 EPFL Smart Kitchen - Node.js Setup"
echo "======================================"
echo ""

# Check Node.js version
echo "📋 Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js >= 18.0.0"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or higher is required. Current: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"
echo ""

# Navigate to website_node directory
cd "$(dirname "$0")"
echo "📁 Working directory: $(pwd)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "package-lock.json" ]; then
    npm ci
else
    npm install
fi
echo "✅ Dependencies installed"
echo ""

# Copy .env.example to .env if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Copy assets from original website
echo "📋 Copying assets from original website..."
npm run build
echo "✅ Assets copied"
echo ""

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs
echo "✅ Logs directory created"
echo ""

echo "======================================"
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "  1. Review .env file for configuration"
echo "  2. Run 'npm run dev' to start development server"
echo "  3. Visit http://localhost:3000"
echo ""
echo "📚 Documentation:"
echo "  - QUICKSTART.md - Quick start guide"
echo "  - DEPLOYMENT.md - Deployment instructions"
echo "  - README.md - Project overview"
echo ""
echo "🚀 Ready to deploy? See DEPLOYMENT.md for options"
echo "======================================"
