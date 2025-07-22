# src/vintageoptics/core/config_manager.py

"""
Configuration management system
"""

import yaml
from typing import Dict, Any

class ConfigManager:
    """Configuration management system"""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return minimal default config
            return {
                'processing': {'default_mode': 'correct'},
                'physics': {'distortion_correction': 0.9},
                'depth': {'enabled': True},
                'synthesis': {'bokeh_quality': 'high'},
                'cleanup': {'dust_sensitivity': 0.8}
            }
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {
                'processing': {'default_mode': 'correct'},
                'physics': {'distortion_correction': 0.9},
                'depth': {'enabled': True},
                'synthesis': {'bokeh_quality': 'high'},
                'cleanup': {'dust_sensitivity': 0.8}
            }
