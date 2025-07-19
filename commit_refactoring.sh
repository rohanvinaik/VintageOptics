#!/bin/bash

# VintageOptics - Git Commit and Push Script
# This script commits the major refactoring changes

echo "🚀 VintageOptics - Committing Refactoring Changes"
echo "================================================"

# Change to project directory
cd /Users/rohanvinaik/VintageOptics

# Check current status
echo "📊 Current Git Status:"
git status --short

# Add all changes
echo -e "\n📁 Adding all changes..."
git add -A

# Show what will be committed
echo -e "\n📋 Files to be committed:"
git status --short

# Create detailed commit message
COMMIT_MESSAGE="Major refactor: Reorganize project structure for clarity and maintainability

🏗️ Structure Changes:
- Moved 14 frontend_api_*.py variants to /legacy/
- Organized all shell scripts into /scripts/shell/
- Moved test files to proper /tests/ directory
- Moved development scripts to /dev/
- Consolidated requirements in /requirements/
- Removed duplicate VintageOptics/ directory

✨ New Features:
- Added main.py as primary CLI entry point
- Created run_tests.py for easy test execution
- Added comprehensive README files for new directories
- Implemented proper pytest test structure

🐛 Fixes:
- Fixed merge conflicts in test files
- Resolved module-level execution issues in tests
- Updated imports for new structure
- Added pytest.ini configuration

📚 Documentation:
- Added REFACTORING_SUMMARY.md
- Added REFACTORING_COMPLETE.md
- Added TEST_FIX_SUMMARY.md
- Updated directory READMEs

This refactoring creates a clean, modular structure that embodies the 
constraint-oriented architecture vision, making the codebase more 
maintainable and ready for future development."

# Commit changes
echo -e "\n💾 Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Check if commit was successful
if [ $? -eq 0 ]; then
    echo -e "\n✅ Commit successful!"
    
    # Show the commit
    echo -e "\n📝 Latest commit:"
    git log --oneline -1
    
    # Ask for confirmation before pushing
    echo -e "\n🤔 Ready to push to GitHub?"
    echo "This will push to: $(git remote get-url origin)"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n📤 Pushing to GitHub..."
        git push origin main
        
        if [ $? -eq 0 ]; then
            echo -e "\n🎉 Successfully pushed to GitHub!"
            echo "View your repository at: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')"
        else
            echo -e "\n❌ Push failed. You may need to:"
            echo "1. Set up authentication: git config --global credential.helper osxkeychain"
            echo "2. Pull latest changes first: git pull origin main"
            echo "3. Resolve any conflicts"
        fi
    else
        echo -e "\n⏸️  Push cancelled. You can push later with: git push origin main"
    fi
else
    echo -e "\n❌ Commit failed. Please check for errors above."
fi

echo -e "\n✨ Done!"
