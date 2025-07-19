#!/usr/bin/env python3
"""
VintageOptics - Main Entry Point
================================

A comprehensive lens correction and synthesis framework that bridges vintage optical
characteristics with modern computational photography.

This module provides the main command-line interface for VintageOptics operations.
"""

import sys
import argparse
import logging
from pathlib import Path

from src.vintageoptics.core.synthesis_pipeline import SynthesisPipeline
from src.vintageoptics.core.config_manager import ConfigManager
from src.vintageoptics.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def main():
    """Main entry point for VintageOptics CLI."""
    parser = argparse.ArgumentParser(
        description="VintageOptics - Lens Correction and Synthesis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply vintage lens correction to an image
  python main.py correct input.jpg output.jpg --lens "Canon FD 50mm f/1.4"
  
  # Synthesize vintage lens characteristics
  python main.py synthesize input.jpg output.jpg --profile "retro_film"
  
  # Run calibration on a set of images
  python main.py calibrate ./calibration_images/ --output ./lens_profile.json
  
  # Start the web interface
  python main.py server --port 8080
        """
    )
    
    # Global arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Correct command
    correct_parser = subparsers.add_parser(
        "correct",
        help="Apply lens correction to images"
    )
    correct_parser.add_argument("input", type=Path, help="Input image path")
    correct_parser.add_argument("output", type=Path, help="Output image path")
    correct_parser.add_argument("--lens", help="Lens model name")
    correct_parser.add_argument("--strength", type=float, default=1.0, help="Correction strength (0-1)")
    
    # Synthesize command
    synth_parser = subparsers.add_parser(
        "synthesize",
        help="Synthesize vintage lens characteristics"
    )
    synth_parser.add_argument("input", type=Path, help="Input image path")
    synth_parser.add_argument("output", type=Path, help="Output image path")
    synth_parser.add_argument("--profile", help="Synthesis profile name")
    synth_parser.add_argument("--intensity", type=float, default=0.8, help="Effect intensity")
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate lens profile from images"
    )
    calibrate_parser.add_argument("input_dir", type=Path, help="Directory containing calibration images")
    calibrate_parser.add_argument("--output", type=Path, help="Output profile path")
    calibrate_parser.add_argument("--method", choices=["opencv", "hugin", "ml"], default="opencv")
    
    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start the web interface"
    )
    server_parser.add_argument("--port", type=int, default=5000, help="Server port")
    server_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    server_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = ConfigManager()
    if args.config:
        config.load_from_file(args.config)
    
    # Execute command
    if args.command == "correct":
        from src.vintageoptics.api.cli import correct_image
        correct_image(args.input, args.output, lens=args.lens, strength=args.strength)
        
    elif args.command == "synthesize":
        from src.vintageoptics.api.cli import synthesize_effect
        synthesize_effect(args.input, args.output, profile=args.profile, intensity=args.intensity)
        
    elif args.command == "calibrate":
        from src.vintageoptics.api.cli import calibrate_lens
        calibrate_lens(args.input_dir, output=args.output, method=args.method)
        
    elif args.command == "server":
        from frontend_api import app
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
