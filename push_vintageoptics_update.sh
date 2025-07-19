#!/bin/bash
# Script to push VintageOptics updates to GitHub

echo "=== Pushing VintageOptics to GitHub ==="
echo

# Navigate to the project directory
cd /Users/rohanvinaik/VintageOptics/VintageOptics

# Check current status
echo "1. Current Git Status:"
git status
echo

# Add all new and modified files
echo "2. Adding all changes..."
git add .
echo "Files added."
echo

# Show what will be committed
echo "3. Changes to be committed:"
git status --short
echo

# Create commit message
COMMIT_MSG="Add Vintage ML implementation with de-dimensionalization techniques

- Implemented vintage ML algorithms (Perceptron, ADALINE, SOM, k-NN)
- Added hybrid physics-ML pipeline with iterative refinement
- Integrated PAC learning principles for adaptive processing
- Added de-dimensionalization techniques:
  * Random Projection for fast dimensionality reduction
  * Sparse PCA for interpretable features
  * Incremental PCA for large datasets
  * Quantized PCA for memory efficiency
  * Adaptive dimensionality reducer
- Implemented model compression techniques:
  * Low-rank approximation
  * Weight pruning (90% sparsity)
  * Knowledge distillation
- Added comprehensive documentation and demos
- Configured YAML settings for all features"

# Commit changes
echo "4. Committing changes..."
git commit -m "$COMMIT_MSG"
echo

# Check remote
echo "5. Checking remote repository..."
git remote -v
echo

# Push to GitHub
echo "6. Pushing to GitHub..."
git push origin main

echo
echo "=== Push Complete! ==="
echo
echo "Your VintageOptics project with Vintage ML implementation has been pushed to GitHub."
echo "Check your repository at: https://github.com/rohanvinaik/VintageOptics"
