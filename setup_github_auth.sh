#!/bin/bash

# GitHub Token Setup Script
echo "========================================="
echo "GitHub Token Authentication Setup"
echo "========================================="
echo ""
echo "This script will help you set up GitHub authentication using a Personal Access Token"
echo ""

# Method 1: Using Git Credential Manager
echo "Method 1: Store credentials in macOS Keychain (Recommended)"
echo "--------------------------------------------------------"
echo "Run this command:"
echo ""
echo "git config --global credential.helper osxkeychain"
echo ""
echo "Then push again. When prompted:"
echo "- Username: rohanvinaik"
echo "- Password: [your-personal-access-token]"
echo ""
echo "The token will be saved in Keychain for future use."
echo ""

# Method 2: Using HTTPS with token in URL
echo "Method 2: Update remote URL with token (Less secure)"
echo "---------------------------------------------------"
echo "Replace <TOKEN> with your actual token:"
echo ""
echo "git remote set-url origin https://<TOKEN>@github.com/rohanvinaik/VintageOptics.git"
echo ""
echo "WARNING: This stores your token in .git/config (less secure)"
echo ""

# Method 3: Using SSH (Most secure)
echo "Method 3: Switch to SSH (Most secure, requires SSH key setup)"
echo "-----------------------------------------------------------"
echo "1. First, check if you have an SSH key:"
echo "   ls -la ~/.ssh/id_*.pub"
echo ""
echo "2. If no key exists, create one:"
echo "   ssh-keygen -t ed25519 -C \"your-email@example.com\""
echo ""
echo "3. Add the public key to GitHub:"
echo "   cat ~/.ssh/id_ed25519.pub"
echo "   Copy this and add to GitHub Settings → SSH and GPG keys"
echo ""
echo "4. Change remote to use SSH:"
echo "   git remote set-url origin git@github.com:rohanvinaik/VintageOptics.git"
echo ""
echo "5. Test SSH connection:"
echo "   ssh -T git@github.com"
echo ""

# Current status
echo "========================================="
echo "Current Setup"
echo "========================================="
echo "Current remote URL:"
git remote -v
echo ""
echo "Current credential helper:"
git config --global credential.helper
echo ""

# Quick fix
echo "========================================="
echo "Quick Fix for Your Current Situation"
echo "========================================="
echo "1. Create a Personal Access Token on GitHub (if you haven't already)"
echo "2. Run: git config --global credential.helper osxkeychain"
echo "3. Try pushing again: git push -u origin main"
echo "4. Use your GitHub username and the token as password"
echo ""
