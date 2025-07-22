"""
Semantic orchestrator for LLM-guided tool composition.

Provides a high-level interface for natural language-driven
optical processing pipelines.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from ..constraints.task_graph import OpticalTaskGraph, TaskNode
from .tool_registry import OpticalToolRegistry, ToolType


class SemanticOpticalOrchestrator:
    """
    LLM-guided orchestration of optical processing tools.
    
    This is a skeleton for future LLM integration that demonstrates
    the pattern for semantic tool composition.
    """
    
    def __init__(self, tool_registry: Optional[OpticalToolRegistry] = None):
        self.registry = tool_registry or OpticalToolRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Template patterns for common requests
        self.request_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize request understanding templates."""
        return {
            'vintage_look': {
                'keywords': ['vintage', 'old', 'retro', 'classic', 'film'],
                'tool_sequence': ['lens_analyzer', 'vintage_detector', 'lens_emulation', 'film_emulation'],
                'default_params': {
                    'vintage_strength': 0.7,
                    'preserve_sharpness': False
                }
            },
            'professional_correction': {
                'keywords': ['correct', 'fix', 'professional', 'clean', 'sharp'],
                'tool_sequence': ['lens_analyzer', 'chromatic_corrector', 'distortion_corrector', 'sharpness_enhancer'],
                'default_params': {
                    'auto_detect': True,
                    'preserve_character': False
                }
            },
            'artistic_bokeh': {
                'keywords': ['bokeh', 'blur', 'background', 'portrait', 'artistic'],
                'tool_sequence': ['depth_estimator', 'bokeh_synthesizer'],
                'default_params': {
                    'bokeh_quality': 'smooth',
                    'aperture_simulation': 1.4
                }
            },
            'specific_lens': {
                'keywords': ['helios', 'zeiss', 'leica', 'canon', 'nikon'],
                'tool_sequence': ['lens_analyzer', '{lens_model}'],
                'default_params': {
                    'emulation_accuracy': 0.9
                }
            }
        }
    
    def process_request(self, natural_language_prompt: str, 
                       context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a natural language request into an executable pipeline.
        
        Args:
            natural_language_prompt: User's request in natural language
            context: Optional context (image metadata, user preferences, etc.)
            
        Returns:
            Execution plan with task graph and parameters
        """
        # Analyze prompt
        intent, entities = self._analyze_prompt(natural_language_prompt)
        
        # Extract constraints from prompt
        constraints = self._extract_constraints(natural_language_prompt, context)
        
        # Build task graph
        task_graph = self._build_task_graph(intent, entities, constraints)
        
        # Optimize pipeline
        optimized_graph = self._optimize_pipeline(task_graph, constraints)
        
        return {
            'intent': intent,
            'entities': entities,
            'constraints': constraints,
            'task_graph': optimized_graph,
            'explanation': self._generate_explanation(optimized_graph)
        }
    
    def _analyze_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze prompt to extract intent and entities.
        
        This is a simplified version - real implementation would use NLP/LLM.
        """
        prompt_lower = prompt.lower()
        
        # Determine intent
        intent = 'unknown'
        for template_name, template in self.request_templates.items():
            if any(keyword in prompt_lower for keyword in template['keywords']):
                intent = template_name
                break
        
        # Extract entities (simplified pattern matching)
        entities = {}
        
        # Lens models
        lens_keywords = {
            'helios': 'helios_44_2',
            'canon 50': 'canon_50mm_f095',
            'zeiss': 'zeiss_planar_50'
        }
        
        for keyword, lens_id in lens_keywords.items():
            if keyword in prompt_lower:
                entities['lens_model'] = lens_id
        
        # Film types
        film_keywords = {
            'portra': 'kodak_portra_400',
            'velvia': 'fuji_velvia_50',
            'tri-x': 'kodak_trix_400'
        }
        
        for keyword, film_id in film_keywords.items():
            if keyword in prompt_lower:
                entities['film_type'] = film_id
        
        # Effect parameters
        if 'strong' in prompt_lower or 'heavy' in prompt_lower:
            entities['strength'] = 0.9
        elif 'subtle' in prompt_lower or 'light' in prompt_lower:
            entities['strength'] = 0.3
        else:
            entities['strength'] = 0.6
        
        return intent, entities
    
    def _extract_constraints(self, prompt: str, 
                           context: Optional[Dict]) -> Dict[str, Any]:
        """Extract physical and quality constraints from prompt."""
        constraints = {
            'physical': {},
            'quality': {},
            'performance': {}
        }
        
        prompt_lower = prompt.lower()
        
        # Physical constraints
        if 'realistic' in prompt_lower:
            constraints['physical']['respect_physics'] = True
            constraints['physical']['max_aberration'] = 3.0
        
        if 'preserve sharpness' in prompt_lower:
            constraints['quality']['min_sharpness'] = 0.8
        
        # Performance constraints
        if 'fast' in prompt_lower or 'quick' in prompt_lower:
            constraints['performance']['max_time'] = 5.0
            constraints['performance']['prefer_gpu'] = True
        
        # Context-based constraints
        if context:
            if context.get('image_size', [0, 0])[0] > 4000:
                constraints['performance']['use_tiling'] = True
            
            if context.get('bit_depth') == 16:
                constraints['quality']['preserve_bit_depth'] = True
        
        return constraints
    
    def _build_task_graph(self, intent: str, entities: Dict[str, Any],
                         constraints: Dict[str, Any]) -> OpticalTaskGraph:
        """Build task graph based on intent and entities."""
        graph = OpticalTaskGraph()
        
        # Get template for intent
        template = self.request_templates.get(intent, {})
        tool_sequence = template.get('tool_sequence', ['lens_analyzer'])
        
        # Build graph from template
        task_id = 0
        previous_task_id = None
        
        for tool_ref in tool_sequence:
            # Resolve tool reference
            if tool_ref.startswith('{') and tool_ref.endswith('}'):
                # Dynamic tool reference
                entity_key = tool_ref[1:-1]
                tool_id = entities.get(entity_key)
                if not tool_id:
                    continue
            else:
                tool_id = tool_ref
            
            # Look up tool in registry
            tool = self.registry.get_tool(tool_id)
            if not tool:
                self.logger.warning(f"Tool {tool_id} not found in registry")
                continue
            
            # Create task node
            task = TaskNode(
                id=f"task_{task_id}",
                name=tool.name,
                task_type=tool_id,
                parameters=self._get_tool_parameters(tool, entities, template),
                constraints=self._get_tool_constraints(tool, constraints)
            )
            
            # Add to graph
            graph.add_task(task)
            
            # Add dependency
            if previous_task_id is not None:
                graph.add_dependency(f"task_{previous_task_id}", f"task_{task_id}")
            
            previous_task_id = task_id
            task_id += 1
        
        return graph
    
    def _get_tool_parameters(self, tool, entities: Dict, 
                           template: Dict) -> Dict[str, Any]:
        """Determine parameters for a tool based on context."""
        params = {}
        
        # Start with template defaults
        if 'default_params' in template:
            params.update(template['default_params'])
        
        # Override with entity-specific values
        if 'strength' in entities:
            # Map strength to tool-specific parameters
            for cap in tool.capabilities:
                for param_name, param_spec in cap.parameters.items():
                    if 'strength' in param_name or 'intensity' in param_name:
                        params[param_name] = entities['strength']
        
        return params
    
    def _get_tool_constraints(self, tool, constraints: Dict) -> List[str]:
        """Extract relevant constraints for a tool."""
        tool_constraints = []
        
        # Map global constraints to tool-specific ones
        if constraints['physical'].get('respect_physics'):
            if tool.type == ToolType.LENS_EMULATION:
                tool_constraints.append('diffraction_limit')
                tool_constraints.append('energy_conservation')
        
        if constraints['quality'].get('min_sharpness'):
            if tool.type in [ToolType.CORRECTION, ToolType.ENHANCEMENT]:
                tool_constraints.append('preserve_detail')
        
        return tool_constraints
    
    def _optimize_pipeline(self, graph: OpticalTaskGraph,
                          constraints: Dict) -> OpticalTaskGraph:
        """Optimize pipeline based on constraints."""
        # This is where advanced optimization would happen
        # For now, just validate and return
        
        is_valid, errors = graph.validate_graph()
        if not is_valid:
            self.logger.warning(f"Graph validation errors: {errors}")
        
        return graph
    
    def _generate_explanation(self, graph: OpticalTaskGraph) -> str:
        """Generate human-readable explanation of the pipeline."""
        explanation_parts = []
        
        explanation_parts.append("I'll process your image through the following steps:")
        
        task_order = graph.get_execution_order()
        for i, task_id in enumerate(task_order, 1):
            task = graph.graph.nodes[task_id]['task']
            tool = self.registry.get_tool(task.task_type)
            
            if tool:
                explanation_parts.append(
                    f"{i}. {task.name}: {tool.description}"
                )
                
                # Add parameter explanations
                if task.parameters:
                    param_strs = []
                    for key, value in task.parameters.items():
                        param_strs.append(f"{key}={value}")
                    explanation_parts.append(f"   Settings: {', '.join(param_strs)}")
        
        return "\n".join(explanation_parts)
    
    def suggest_alternatives(self, request: str, 
                           current_result: Dict) -> List[Dict[str, Any]]:
        """Suggest alternative approaches for a request."""
        alternatives = []
        
        # Get current intent
        current_intent = current_result.get('intent', 'unknown')
        
        # Suggest variations
        if current_intent == 'vintage_look':
            alternatives.extend([
                {
                    'description': 'Try a different vintage lens',
                    'modification': {'lens_model': 'zeiss_biotar_75'},
                    'reason': 'Zeiss Biotar offers smoother bokeh transitions'
                },
                {
                    'description': 'Add film grain for authenticity',
                    'modification': {'add_tool': 'film_grain_synthesizer'},
                    'reason': 'Film grain enhances the vintage aesthetic'
                }
            ])
        
        elif current_intent == 'professional_correction':
            alternatives.extend([
                {
                    'description': 'Preserve some lens character',
                    'modification': {'preserve_character': 0.3},
                    'reason': 'Maintains artistic qualities while correcting flaws'
                }
            ])
        
        return alternatives
    
    def explain_constraints(self, constraint_name: str) -> str:
        """Provide explanation for a specific constraint."""
        explanations = {
            'diffraction_limit': 
                "The diffraction limit prevents details smaller than λ/2NA from being "
                "resolved, where λ is the wavelength of light and NA is the numerical aperture.",
            
            'energy_conservation':
                "Total light energy must be conserved - the image can't become "
                "artificially brighter without a physical light source.",
            
            'chromatic_aberration':
                "Different wavelengths of light focus at different distances, "
                "causing color fringing at high-contrast edges.",
            
            'vignetting_profile':
                "Light falloff toward image edges follows physics - typically "
                "cos^4(θ) for wide-angle lenses."
        }
        
        return explanations.get(
            constraint_name,
            f"Physical constraint ensuring realistic optical behavior: {constraint_name}"
        )
    
    def generate_code(self, execution_plan: Dict) -> str:
        """Generate executable code from an execution plan."""
        code_lines = [
            "# Auto-generated VintageOptics pipeline",
            "from vintageoptics import OpticalPipeline",
            "",
            "# Initialize pipeline",
            "pipeline = OpticalPipeline()",
            ""
        ]
        
        # Add tasks
        task_graph = execution_plan['task_graph']
        for task_id in task_graph.get_execution_order():
            task = task_graph.graph.nodes[task_id]['task']
            
            code_lines.append(f"# {task.name}")
            code_lines.append(
                f"pipeline.add_step('{task.task_type}', "
                f"parameters={task.parameters})"
            )
            code_lines.append("")
        
        # Add execution
        code_lines.extend([
            "# Execute pipeline",
            "result = pipeline.execute(input_image)",
            ""
        ])
        
        return "\n".join(code_lines)
