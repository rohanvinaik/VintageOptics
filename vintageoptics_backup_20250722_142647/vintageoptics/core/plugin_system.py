# src/vintageoptics/core/plugin_system.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import importlib
import inspect

class VintageOpticsPlugin(ABC):
    """Base class for all plugins"""
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """Return plugin information"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data through plugin"""
        pass


class PluginManager:
    """Manages plugin discovery and lifecycle"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.plugins: Dict[str, VintageOpticsPlugin] = {}
        self.hooks: Dict[str, List[callable]] = {}
        
    def discover_plugins(self, plugin_dir: str = "plugins"):
        """Discover and load plugins"""
        # Implementation for dynamic plugin loading
        pass
    
    def register_hook(self, hook_name: str, callback: callable):
        """Register a hook callback"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, data: Any) -> Any:
        """Execute all callbacks for a hook"""
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                data = callback(data)
        return data