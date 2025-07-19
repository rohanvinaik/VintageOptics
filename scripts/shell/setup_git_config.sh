#!/bin/bash
# Quick setup script to ensure git is configured

echo "=== Git Configuration Check ==="
echo

# Check if git user is configured
if [ -z "$(git config --global user.name)" ]; then
    echo "Git user name not configured. Setting it up..."
    git config --global user.name "Rohan Vinaik"
fi

if [ -z "$(git config --global user.email)" ]; then
    echo "Git user email not configured. Setting it up..."
    git config --global user.email "rohanpvinaik@gmail.com"
fi

echo "Git configuration:"
echo "  User: $(git config --global user.name)"
echo "  Email: $(git config --global user.email)"
echo

cd /Users/rohanvinaik/VintageOptics/VintageOptics

# Check if this is already a git repository
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    echo
fi

# Check if remote is set
if ! git remote | grep -q origin; then
    echo "No remote 'origin' found. Adding GitHub remote..."
    echo "Please enter your GitHub repository URL:"
    echo "Format: https://github.com/rohanvinaik/VintageOptics.git"
    read -p "GitHub URL: " REPO_URL
    
    if [ ! -z "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        echo "Remote added successfully!"
    fi
fi

echo
echo "Current remotes:"
git remote -v

echo
echo "=== Setup Complete ==="
echo "You can now run: ./push_vintageoptics_update.sh"
