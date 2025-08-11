#!/bin/bash

# Deployment Monitor Script
# Run this script daily to verify backend deployment is working correctly

echo "🔍 Starting deployment health check..."

# Current production URL (update this when deployment changes)
DEPLOYMENT_URL="https://herba-ai-proxy-31dzwinpz-dustins-projects-2a4636fb.vercel.app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local expected_content=$2
    local description=$3
    
    log "Testing $description..."
    
    response=$(curl -s -X GET "$DEPLOYMENT_URL$endpoint" 2>/dev/null)
    
    if [[ $response == *"$expected_content"* ]]; then
        echo -e "${GREEN}✅ $description: PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $description: FAILED${NC}"
        echo "   Expected: $expected_content"
        echo "   Got: $response"
        return 1
    fi
}

# Function to test AI analysis endpoint
test_ai_analysis() {
    log "Testing AI analysis endpoint..."
    
    response=$(curl -s -X POST "$DEPLOYMENT_URL/analyzeResponseForRemedy" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test-token" \
        -d '{"response": "**Remedy: Peppermint Tea**\n\n**How it helps:** Peppermint has been used in traditional medicine for centuries."}' \
        2>/dev/null)
    
    if [[ $response == *"contains_remedy"* ]]; then
        echo -e "${GREEN}✅ AI Analysis Endpoint: PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ AI Analysis Endpoint: FAILED${NC}"
        echo "   Got: $response"
        return 1
    fi
}

# Initialize counters
passed=0
failed=0

# Test 1: Health endpoint
if check_endpoint "/health" "healthy" "Health Endpoint"; then
    ((passed++))
else
    ((failed++))
fi

# Test 2: Rate limit info
if check_endpoint "/rate-limit-info" "ai_analysis" "Rate Limit Info"; then
    ((passed++))
else
    ((failed++))
fi

# Test 3: AI analysis endpoint
if test_ai_analysis; then
    ((passed++))
else
    ((failed++))
fi

# Test 4: OpenAPI spec
if check_endpoint "/openapi.json" "analyzeResponseForRemedy" "OpenAPI Spec"; then
    ((passed++))
else
    ((failed++))
fi

# Summary
echo ""
echo "📊 Health Check Summary:"
echo -e "${GREEN}✅ Passed: $passed${NC}"
echo -e "${RED}❌ Failed: $failed${NC}"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Deployment is healthy.${NC}"
    exit 0
else
    echo -e "${RED}🚨 Some checks failed! Deployment may have issues.${NC}"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "1. Check Vercel deployment logs: vercel logs $DEPLOYMENT_URL"
    echo "2. Verify vercel.json points to main_fixed.py"
    echo "3. Check requirements.txt includes firebase-admin"
    echo "4. Force redeploy: vercel --prod"
    exit 1
fi
