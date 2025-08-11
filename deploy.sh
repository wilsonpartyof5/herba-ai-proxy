#!/bin/bash

# Herba2 Backend Deployment Script
# This script validates configuration, deploys, and verifies the deployment

set -e  # Exit on any error

echo "🚀 Herba2 Backend Deployment Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
log "Checking prerequisites..."

if ! command_exists vercel; then
    echo -e "${RED}❌ Vercel CLI not found. Install with: npm install -g vercel${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 1: Pre-deployment validation
log "Running pre-deployment validation..."
if ! python3 validate_config.py; then
    echo -e "${RED}❌ Configuration validation failed. Fix issues before deployment.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-deployment validation passed${NC}"

# Step 2: Deploy to production
log "Deploying to production..."
echo -e "${YELLOW}⚠️  This will deploy to production. Continue? (y/N)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

vercel --prod

# Step 3: Get new deployment URL
log "Getting new deployment URL..."
NEW_URL=$(vercel ls | head -2 | tail -1 | awk '{print $4}')
echo -e "${GREEN}✅ New deployment URL: $NEW_URL${NC}"

# Step 4: Wait for deployment to be ready
log "Waiting for deployment to be ready..."
sleep 30

# Step 5: Update monitoring script with new URL
log "Updating monitoring script with new URL..."
sed -i.bak "s|DEPLOYMENT_URL=.*|DEPLOYMENT_URL=\"$NEW_URL\"|" deployment_monitor.sh
echo -e "${GREEN}✅ Monitoring script updated${NC}"

# Step 6: Run health checks
log "Running health checks..."
if ./deployment_monitor.sh; then
    echo -e "${GREEN}✅ Health checks passed${NC}"
else
    echo -e "${RED}❌ Health checks failed${NC}"
    echo -e "${YELLOW}⚠️  Deployment may have issues. Check logs with: vercel logs $NEW_URL${NC}"
    exit 1
fi

# Step 7: Update deployment registry
log "Updating deployment registry..."
cat > deployment_registry.json << EOF
{
  "current_production": "$NEW_URL",
  "deployment_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "main_fixed.py",
  "features": [
    "firebase_auth",
    "rate_limiting", 
    "ai_analysis",
    "input_validation"
  ],
  "ios_app_urls": [
    "Herba2/ViewModels/AIHerbalistChatViewModel.swift",
    "Herba2/Services/AIService.swift",
    "Herba2/Services/AIHerbalistService.swift"
  ],
  "deployment_success": true
}
EOF

echo -e "${GREEN}✅ Deployment registry updated${NC}"

# Step 8: Summary
echo ""
echo -e "${GREEN}🎉 Deployment Successful!${NC}"
echo "=================================="
echo -e "${BLUE}Production URL:${NC} $NEW_URL"
echo -e "${BLUE}Deployment Date:${NC} $(date)"
echo -e "${BLUE}Version:${NC} main_fixed.py"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Update iOS app URLs to: $NEW_URL"
echo "2. Test iOS app connectivity"
echo "3. Verify herbal growth animation"
echo "4. Run daily monitoring: ./deployment_monitor.sh"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "- Deployment Guide: Documentation/DEPLOYMENT_MONITORING_CHECKLIST.md"
echo "- Backend Fix: Documentation/BACKEND_DEPLOYMENT_FIX.md"
echo "- Health Check: ./deployment_monitor.sh"
echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
