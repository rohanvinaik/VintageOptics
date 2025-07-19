#!/bin/bash
cd /Users/rohanvinaik/VintageOptics

echo "📋 Checking git status..."
git status

echo ""
echo "📦 Adding all changes..."
git add -A

echo ""
echo "💾 Committing changes..."
git commit -m "Add frontend-compatible API with restore/synthesize endpoints

- Created frontend_api_compatible.py with proper endpoints
- Added /api/restore endpoint for vintage image restoration  
- Added /api/synthesize endpoint for adding vintage effects
- Implemented real-time progress tracking with Server-Sent Events
- Support for multiple lens profiles (Helios, Canon, Takumar, Meyer-Optik)
- Added defect removal and optical correction features
- Added vintage effect synthesis with customizable parameters
- Full integration with React frontend
- Working swirly bokeh effect for Helios lens profile"

echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Done! Checking final status..."
git status
echo ""
echo "Latest commit:"
git log --oneline -1
