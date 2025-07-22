# src/vintageoptics/core/performance_monitor.py

"""
Performance monitoring system
"""

import time
from typing import Dict, Any

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self):
        self.metrics = {}
    
    def track(self, operation_name: str):
        """Context manager for tracking operation performance"""
        return PerformanceContext(operation_name, self)

class PerformanceContext:
    """Context manager for performance tracking"""
    
    def __init__(self, operation_name: str, monitor: PerformanceMonitor):
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.metrics[self.operation_name] = duration
