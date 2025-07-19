#!/bin/bash

PROJECT="VintageOptics"

# Directories to create
DIRS=(
"$PROJECT/src/vintageoptics/core"
"$PROJECT/src/vintageoptics/depth"
"$PROJECT/src/vintageoptics/integrations"
"$PROJECT/src/vintageoptics/synthesis"
"$PROJECT/config/schemas"
"$PROJECT/config/environments"
)

# Files to create
FILES=(
"$PROJECT/src/vintageoptics/depth/depth_analyzer.py"
"$PROJECT/src/vintageoptics/depth/frequency_analyzer.py"
"$PROJECT/src/vintageoptics/depth/layer_processor.py"
"$PROJECT/src/vintageoptics/depth/bokeh_analyzer.py"
"$PROJECT/src/vintageoptics/depth/depth_aware_cleanup.py"
"$PROJECT/src/vintageoptics/core/depth_aware_pipeline.py"
"$PROJECT/src/vintageoptics/core/synthesis_pipeline.py"
"$PROJECT/config/depth_aware.yaml"
"$PROJECT/config/synthesis.yaml"
"$PROJECT/src/vintageoptics/integrations/__init__.py"
"$PROJECT/src/vintageoptics/integrations/midas_integration.py"
"$PROJECT/src/vintageoptics/integrations/exiftool_integration.py"
"$PROJECT/src/vintageoptics/integrations/hugin_integration.py"
"$PROJECT/src/vintageoptics/synthesis/lens_synthesizer.py"
"$PROJECT/src/vintageoptics/synthesis/bokeh_synthesis.py"
"$PROJECT/src/vintageoptics/synthesis/optical_characteristics.py"
"$PROJECT/src/vintageoptics/synthesis/fov_transformer.py"
"$PROJECT/src/vintageoptics/synthesis/lens_library.py"
)

# Create directories
for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
done

# Create empty files
for file in "${FILES[@]}"; do
  touch "$file"
done

echo "✅ VintageOptics updated structure created successfully!"