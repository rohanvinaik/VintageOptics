#!/bin/bash
# Install and run TailChasingFixer on VintageOptics

echo "🔧 Setting up TailChasingFixer for VintageOptics"
echo "============================================"

# Change to VintageOptics directory
cd /Users/rohanvinaik/VintageOptics

# Install TailChasingFixer
echo "📦 Installing TailChasingFixer..."
pip install tail-chasing-detector

# Create configuration
echo "📝 Creating .tailchasing.yml configuration..."
cat > .tailchasing.yml << 'EOF'
paths:
  include:
    - src/vintageoptics
  exclude:
    - tests
    - build
    - venv
    - __pycache__
    - "*.egg-info"
    - backup_*

risk_thresholds:
  warn: 15
  fail: 30

placeholders:
  allow:
    - BaseDetector.__init__
    - BasePlugin.initialize

scoring_weights:
  missing_symbol: 3
  phantom_function: 3
  duplicate_function: 3
  wrapper_abstraction: 1
  semantic_duplicate_function: 4
  prototype_fragmentation: 4
  circular_import: 3
  hallucinated_import: 3
  rename_cascade_chain: 4
  tail_chasing_chain: 4

semantic:
  enable: true
  hv_dim: 8192
  min_functions: 30
  z_threshold: 2.5
  channel_weights:
    NAME_TOKENS: 1.0
    CALLS: 1.2
    DOC_TOKENS: 0.8
EOF

# Run analysis
echo "🔍 Running tail-chasing analysis..."
tailchasing src/vintageoptics --semantic --output tail_chasing_report.json

# Generate detailed report
echo "📊 Generating detailed report..."
tailchasing src/vintageoptics --semantic --format detailed > tail_chasing_detailed_report.txt

# Show summary
echo ""
echo "✅ Analysis complete!"
echo ""
echo "Reports generated:"
echo "  - tail_chasing_report.json (machine-readable)"
echo "  - tail_chasing_detailed_report.txt (human-readable)"
echo ""
echo "To view the report:"
echo "  cat tail_chasing_detailed_report.txt | less"
