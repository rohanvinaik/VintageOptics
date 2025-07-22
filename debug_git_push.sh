#!/bin/bash
# Debug git status and push

cd /Users/rohanvinaik/VintageOptics

echo "=== Current Git Status ==="
git status

echo -e "\n=== Recent Commits ==="
git log --oneline -5

echo -e "\n=== Remote Configuration ==="
git remote -v

echo -e "\n=== Branch Information ==="
git branch -vv

echo -e "\n=== Attempting to add and commit changes ==="
git add -A
git status

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit"
else
    echo "Changes detected, committing..."
    git commit -m "fix: Apply TailChasingFixer recommendations

- Enhanced synthesis module implementations
- Added error handling and validation
- Improved documentation"
fi

echo -e "\n=== Pushing to remote ==="
git push -v origin main 2>&1

echo -e "\n=== Final Status ==="
git status
