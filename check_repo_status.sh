#!/bin/bash

# Script to verify and fix GitHub push issues

echo "=== Checking VintageOptics Repository Status ==="
echo

cd /Users/rohanvinaik/VintageOptics/VintageOptics

# Check git status
echo "1. Git Status:"
git status

echo -e "\n2. Checking for uncommitted changes:"
git diff --stat

echo -e "\n3. Current branch:"
git branch --show-current

echo -e "\n4. Remote configuration:"
git remote -v

echo -e "\n5. Last few commits:"
git log --oneline -5

echo -e "\n6. Checking if files exist locally:"
echo "   - Core pipeline: $(test -f src/vintageoptics/core/pipeline.py && echo "EXISTS" || echo "MISSING")"
echo "   - Detection module: $(test -f src/vintageoptics/detection/base_detector.py && echo "EXISTS" || echo "MISSING")"
echo "   - Physics module: $(test -f src/vintageoptics/physics/optics_engine.py && echo "EXISTS" || echo "MISSING")"

echo -e "\n7. Checking file sizes (to detect empty files):"
find src -name "*.py" -type f -exec ls -la {} \; | grep -E "(0 .* \.py$|\.py$)" | head -20

echo -e "\n8. Count of Python files with actual content (>100 bytes):"
find src -name "*.py" -type f -size +100c | wc -l

echo -e "\n=== Suggested Fix Commands ==="
echo "If files are not committed:"
echo "  git add -A"
echo "  git commit -m 'Add all implementation files'"
echo "  git push origin main"
echo
echo "If on wrong branch:"
echo "  git checkout main"
echo "  git merge <your-branch>"
echo "  git push origin main"
