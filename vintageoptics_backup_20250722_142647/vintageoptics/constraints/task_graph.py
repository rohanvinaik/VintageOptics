"""
Task graph system for optical processing pipelines.

Enables declarative, constraint-aware pipeline construction that can be
orchestrated by LLMs or configured manually.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime


class TaskStatus(Enum):
    """Status of a task in the execution graph."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    """Represents a single processing task in the graph."""
    
    id: str
    name: str
    task_type: str
    processor: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize task node (excluding processor function)."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.task_type,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'status': self.status.value,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class OpticalTaskGraph:
    """
    Represent lens correction as composable, constraint-aware operations.
    
    This enables both programmatic and LLM-guided pipeline construction.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.processors = self._register_processors()
        self.execution_context = {}
        self.logger = logging.getLogger(__name__)
        
    def _register_processors(self) -> Dict[str, Callable]:
        """Register available processing functions."""
        # Import processors lazily to avoid circular imports
        processors = {}
        
        try:
            from ..analysis.lens_characterizer import LensCharacterizer
            processors['characterize'] = LensCharacterizer()
        except ImportError:
            self.logger.warning("LensCharacterizer not available")
        
        try:
            from ..detection.vintage_detector import VintageDetector
            processors['detect_vintage'] = VintageDetector()
        except ImportError:
            self.logger.warning("VintageDetector not available")
        
        # Add more processors as they become available
        # This is where the modular architecture shines
        
        return processors
    
    def add_task(self, task: TaskNode) -> str:
        """Add a task to the graph."""
        self.graph.add_node(task.id, task=task)
        return task.id
    
    def add_dependency(self, from_task: str, to_task: str, 
                      data_key: Optional[str] = None):
        """Add a dependency between tasks."""
        self.graph.add_edge(from_task, to_task, data_key=data_key)
    
    def compile_from_spec(self, spec: Dict[str, Any]) -> 'OpticalTaskGraph':
        """Compile a graph from a JSON specification."""
        # Clear existing graph
        self.graph.clear()
        
        # Add tasks
        for task_spec in spec.get('tasks', []):
            task = TaskNode(
                id=task_spec['id'],
                name=task_spec['name'],
                task_type=task_spec['type'],
                parameters=task_spec.get('parameters', {}),
                constraints=task_spec.get('constraints', []),
                inputs=task_spec.get('inputs', []),
                outputs=task_spec.get('outputs', [])
            )
            
            # Assign processor if available
            if task.task_type in self.processors:
                task.processor = self.processors[task.task_type]
            
            self.add_task(task)
        
        # Add dependencies
        for dep in spec.get('dependencies', []):
            self.add_dependency(
                dep['from'], 
                dep['to'], 
                dep.get('data_key')
            )
        
        return self
    
    def compile_from_natural_language(self, prompt: str) -> 'OpticalTaskGraph':
        """
        Compile a graph from natural language using LLM interpretation.
        
        This is a skeleton for future LLM integration.
        """
        # Example mappings for common requests
        templates = {
            'helios': {
                'tasks': [
                    {
                        'id': 'detect',
                        'name': 'Detect Lens Type',
                        'type': 'detect_vintage',
                        'parameters': {'confidence_threshold': 0.7}
                    },
                    {
                        'id': 'characterize',
                        'name': 'Extract Lens Characteristics',
                        'type': 'characterize',
                        'parameters': {'extract_bokeh': True}
                    },
                    {
                        'id': 'synthesize',
                        'name': 'Apply Helios Effect',
                        'type': 'synthesize',
                        'parameters': {'lens_model': 'helios_44_2'}
                    }
                ],
                'dependencies': [
                    {'from': 'detect', 'to': 'characterize'},
                    {'from': 'characterize', 'to': 'synthesize'}
                ]
            }
        }
        
        # Simple keyword matching for MVP
        prompt_lower = prompt.lower()
        for keyword, spec in templates.items():
            if keyword in prompt_lower:
                return self.compile_from_spec(spec)
        
        # Default pipeline
        return self.compile_from_spec({
            'tasks': [
                {
                    'id': 'analyze',
                    'name': 'Analyze Image',
                    'type': 'characterize',
                    'parameters': {}
                }
            ],
            'dependencies': []
        })
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate the task graph for correctness."""
        errors = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Graph contains cycles")
        
        # Check for orphaned nodes
        for node in self.graph.nodes():
            if (self.graph.in_degree(node) == 0 and 
                self.graph.out_degree(node) == 0 and
                len(self.graph) > 1):
                errors.append(f"Task {node} is orphaned")
        
        # Check processor availability
        for node in self.graph.nodes():
            task = self.graph.nodes[node]['task']
            if task.processor is None and task.task_type != 'manual':
                errors.append(f"No processor available for task {node} of type {task.task_type}")
        
        # Validate data flow
        for node in self.graph.nodes():
            task = self.graph.nodes[node]['task']
            
            # Check inputs are satisfied
            incoming_outputs = set()
            for pred in self.graph.predecessors(node):
                pred_task = self.graph.nodes[pred]['task']
                incoming_outputs.update(pred_task.outputs)
            
            missing_inputs = set(task.inputs) - incoming_outputs
            if missing_inputs and task.inputs:  # Only error if inputs were specified
                errors.append(f"Task {node} missing inputs: {missing_inputs}")
        
        return len(errors) == 0, errors
    
    def get_execution_order(self) -> List[str]:
        """Get topological order for task execution."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return []
    
    def execute(self, input_data: Dict[str, Any], 
               constraint_checker: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute the task graph with optional constraint checking.
        
        Args:
            input_data: Initial data for the pipeline
            constraint_checker: Optional function to validate constraints
            
        Returns:
            Dictionary of results from all tasks
        """
        # Initialize execution context
        self.execution_context = {
            'input': input_data,
            'results': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'tasks_completed': 0,
                'tasks_failed': 0
            }
        }
        
        # Get execution order
        execution_order = self.get_execution_order()
        
        if not execution_order:
            self.logger.error("Cannot determine execution order - graph may contain cycles")
            return self.execution_context
        
        # Execute tasks in order
        for task_id in execution_order:
            task = self.graph.nodes[task_id]['task']
            
            try:
                # Update status
                task.status = TaskStatus.RUNNING
                self.logger.info(f"Executing task {task_id}: {task.name}")
                
                # Gather inputs
                task_inputs = self._gather_task_inputs(task_id)
                
                # Check constraints if checker provided
                if constraint_checker and task.constraints:
                    constraint_valid = constraint_checker(
                        task.constraints, 
                        task_inputs, 
                        self.execution_context
                    )
                    if not constraint_valid:
                        task.status = TaskStatus.SKIPPED
                        self.logger.warning(f"Task {task_id} skipped due to constraint violation")
                        continue
                
                # Execute processor
                if task.processor:
                    import time
                    start_time = time.time()
                    
                    # Merge parameters with inputs
                    execution_params = {**task.parameters, **task_inputs}
                    
                    # Execute based on processor type
                    if callable(task.processor):
                        result = task.processor(**execution_params)
                    else:
                        # Assume processor has a process/execute method
                        if hasattr(task.processor, 'process'):
                            result = task.processor.process(**execution_params)
                        elif hasattr(task.processor, 'execute'):
                            result = task.processor.execute(**execution_params)
                        else:
                            raise ValueError(f"Processor for {task_id} has no process/execute method")
                    
                    task.execution_time = time.time() - start_time
                    
                    # Store results
                    self.execution_context['results'][task_id] = result
                    
                    # Update status
                    task.status = TaskStatus.COMPLETED
                    self.execution_context['metadata']['tasks_completed'] += 1
                    
                else:
                    # Manual task or placeholder
                    task.status = TaskStatus.SKIPPED
                    self.logger.info(f"Task {task_id} has no processor - skipping")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                self.execution_context['metadata']['tasks_failed'] += 1
                self.logger.error(f"Task {task_id} failed: {e}")
                
                # Decide whether to continue or abort
                if task.metadata.get('critical', False):
                    self.logger.error("Critical task failed - aborting pipeline")
                    break
        
        # Finalize metadata
        self.execution_context['metadata']['end_time'] = datetime.now().isoformat()
        
        return self.execution_context
    
    def _gather_task_inputs(self, task_id: str) -> Dict[str, Any]:
        """Gather inputs for a task from predecessor outputs."""
        inputs = {}
        
        # Get data from predecessors
        for pred_id in self.graph.predecessors(task_id):
            edge_data = self.graph.edges[pred_id, task_id]
            data_key = edge_data.get('data_key')
            
            if pred_id in self.execution_context['results']:
                pred_result = self.execution_context['results'][pred_id]
                
                if data_key and isinstance(pred_result, dict):
                    # Extract specific key
                    if data_key in pred_result:
                        inputs[data_key] = pred_result[data_key]
                else:
                    # Use entire result
                    if isinstance(pred_result, dict):
                        inputs.update(pred_result)
                    else:
                        inputs[f"{pred_id}_output"] = pred_result
        
        # Add original input if this is a root node
        if self.graph.in_degree(task_id) == 0:
            inputs.update(self.execution_context['input'])
        
        return inputs
    
    def visualize(self, output_path: Optional[str] = None) -> str:
        """
        Generate a visual representation of the task graph.
        
        Returns GraphViz DOT format string.
        """
        dot_lines = ["digraph TaskGraph {"]
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [shape=box, style=rounded];')
        
        # Add nodes with status coloring
        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            
            # Color based on status
            color_map = {
                TaskStatus.PENDING: "lightgray",
                TaskStatus.RUNNING: "yellow",
                TaskStatus.COMPLETED: "lightgreen",
                TaskStatus.FAILED: "lightcoral",
                TaskStatus.SKIPPED: "lightblue"
            }
            color = color_map.get(task.status, "white")
            
            label = f"{task.name}\\n({task.task_type})"
            if task.execution_time:
                label += f"\\n{task.execution_time:.2f}s"
            
            dot_lines.append(f'  "{node_id}" [label="{label}", fillcolor={color}, style=filled];')
        
        # Add edges
        for from_id, to_id, data in self.graph.edges(data=True):
            label = data.get('data_key', '')
            if label:
                dot_lines.append(f'  "{from_id}" -> "{to_id}" [label="{label}"];')
            else:
                dot_lines.append(f'  "{from_id}" -> "{to_id}";')
        
        dot_lines.append("}")
        
        dot_content = "\n".join(dot_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(dot_content)
        
        return dot_content
    
    def export_spec(self) -> Dict[str, Any]:
        """Export the graph as a JSON specification."""
        spec = {
            'tasks': [],
            'dependencies': [],
            'metadata': {
                'created': datetime.now().isoformat(),
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges()
            }
        }
        
        # Export tasks
        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            spec['tasks'].append(task.to_dict())
        
        # Export dependencies
        for from_id, to_id, data in self.graph.edges(data=True):
            dep = {
                'from': from_id,
                'to': to_id
            }
            if data.get('data_key'):
                dep['data_key'] = data['data_key']
            
            spec['dependencies'].append(dep)
        
        return spec
    
    def save(self, filename: str):
        """Save the graph specification to a file."""
        spec = self.export_spec()
        with open(filename, 'w') as f:
            json.dump(spec, f, indent=2)
    
    def load(self, filename: str):
        """Load a graph specification from a file."""
        with open(filename, 'r') as f:
            spec = json.load(f)
        
        self.compile_from_spec(spec)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for the graph."""
        stats = {
            'total_tasks': self.graph.number_of_nodes(),
            'total_dependencies': self.graph.number_of_edges(),
            'execution_summary': {
                'completed': 0,
                'failed': 0,
                'skipped': 0,
                'pending': 0
            },
            'total_execution_time': 0
        }
        
        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            status_key = task.status.value
            if status_key in stats['execution_summary']:
                stats['execution_summary'][status_key] += 1
            
            if task.execution_time:
                stats['total_execution_time'] += task.execution_time
        
        return stats
