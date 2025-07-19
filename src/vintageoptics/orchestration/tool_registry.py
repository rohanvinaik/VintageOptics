"""
Semantic tool registry for discoverable, composable optical tools.

Makes tools findable by capability and semantic tags for both
programmatic and LLM-guided composition.
"""

from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging


class ToolType(Enum):
    """Types of optical processing tools."""
    LENS_EMULATION = "lens_emulation"
    FILM_EMULATION = "film_emulation"
    CORRECTION = "correction"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    DETECTION = "detection"
    CALIBRATION = "calibration"
    ENHANCEMENT = "enhancement"


@dataclass
class ToolCapability:
    """Describes what a tool can do."""
    
    name: str
    description: str
    input_types: List[str]  # e.g., ['image', 'metadata']
    output_types: List[str]  # e.g., ['image', 'lens_profile']
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)  # e.g., {'speed': 0.8, 'quality': 0.9}


@dataclass
class OpticalTool:
    """Registry entry for a single tool."""
    
    id: str
    name: str
    type: ToolType
    description: str
    capabilities: List[ToolCapability]
    semantic_tags: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    processor: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_query(self, query: str) -> float:
        """Calculate relevance score for a search query."""
        query_lower = query.lower()
        score = 0.0
        
        # Check name and description
        if query_lower in self.name.lower():
            score += 2.0
        if query_lower in self.description.lower():
            score += 1.0
        
        # Check semantic tags
        for tag in self.semantic_tags:
            if query_lower in tag.lower():
                score += 1.5
        
        # Check capabilities
        for cap in self.capabilities:
            if query_lower in cap.name.lower():
                score += 1.0
            if query_lower in cap.description.lower():
                score += 0.5
        
        return score
    
    def to_dict(self) -> Dict:
        """Serialize tool (excluding processor function)."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'capabilities': [
                {
                    'name': cap.name,
                    'description': cap.description,
                    'input_types': cap.input_types,
                    'output_types': cap.output_types,
                    'parameters': cap.parameters,
                    'constraints': cap.constraints,
                    'performance': cap.performance
                }
                for cap in self.capabilities
            ],
            'semantic_tags': self.semantic_tags,
            'constraints': self.constraints,
            'metadata': self.metadata
        }


class OpticalToolRegistry:
    """
    Registry of available optical processing tools.
    
    Makes tools discoverable by capability, type, and semantic tags.
    """
    
    def __init__(self):
        self.tools: Dict[str, OpticalTool] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_builtin_tools()
    
    def _initialize_builtin_tools(self):
        """Register built-in VintageOptics tools."""
        
        # Lens emulation tools
        self.register_tool(OpticalTool(
            id='helios_44_2',
            name='Helios 44-2',
            type=ToolType.LENS_EMULATION,
            description='Soviet vintage lens with swirly bokeh characteristics',
            capabilities=[
                ToolCapability(
                    name='swirly_bokeh',
                    description='Creates characteristic swirling out-of-focus areas',
                    input_types=['image', 'depth_map'],
                    output_types=['image'],
                    parameters={
                        'intensity': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.7},
                        'swirl_center': {'type': 'point', 'default': 'auto'}
                    }
                ),
                ToolCapability(
                    name='low_contrast',
                    description='Reduces overall image contrast',
                    input_types=['image'],
                    output_types=['image'],
                    parameters={
                        'reduction': {'type': 'float', 'min': 0, 'max': 0.5, 'default': 0.2}
                    }
                )
            ],
            semantic_tags=['vintage', 'soviet', 'biotar_formula', 'portrait', 'artistic'],
            constraints={
                'aperture_range': [2, 8],
                'focal_length': 58
            }
        ))
        
        self.register_tool(OpticalTool(
            id='canon_50mm_f095',
            name='Canon 50mm f/0.95 Dream Lens',
            type=ToolType.LENS_EMULATION,
            description='Ultra-fast lens with extreme shallow depth of field',
            capabilities=[
                ToolCapability(
                    name='extreme_bokeh',
                    description='Creates ultra-shallow depth of field with creamy bokeh',
                    input_types=['image', 'depth_map'],
                    output_types=['image'],
                    parameters={
                        'aperture': {'type': 'float', 'min': 0.95, 'max': 16, 'default': 0.95}
                    }
                ),
                ToolCapability(
                    name='spherical_aberration',
                    description='Adds dream-like spherical aberration',
                    input_types=['image'],
                    output_types=['image'],
                    parameters={
                        'amount': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.3}
                    }
                )
            ],
            semantic_tags=['fast', 'dream', 'portrait', 'low_light', 'japanese'],
            constraints={
                'aperture_range': [0.95, 16],
                'focal_length': 50
            }
        ))
        
        # Film emulation tools
        self.register_tool(OpticalTool(
            id='kodak_portra_400',
            name='Kodak Portra 400',
            type=ToolType.FILM_EMULATION,
            description='Professional color negative film with excellent skin tones',
            capabilities=[
                ToolCapability(
                    name='color_grading',
                    description='Applies Portra-style color grading',
                    input_types=['image'],
                    output_types=['image'],
                    parameters={
                        'strength': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.8}
                    }
                ),
                ToolCapability(
                    name='grain_structure',
                    description='Adds film grain pattern',
                    input_types=['image'],
                    output_types=['image'],
                    parameters={
                        'iso': {'type': 'int', 'min': 160, 'max': 800, 'default': 400},
                        'grain_size': {'type': 'float', 'min': 0.5, 'max': 2, 'default': 1}
                    }
                )
            ],
            semantic_tags=['film', 'portrait', 'warm_tones', 'professional', 'kodak'],
            constraints={
                'iso_range': [160, 800]
            }
        ))
        
        # Correction tools
        self.register_tool(OpticalTool(
            id='chromatic_corrector',
            name='Chromatic Aberration Corrector',
            type=ToolType.CORRECTION,
            description='Corrects lateral chromatic aberration',
            capabilities=[
                ToolCapability(
                    name='lateral_ca_correction',
                    description='Removes color fringing at edges',
                    input_types=['image'],
                    output_types=['image'],
                    parameters={
                        'auto_detect': {'type': 'bool', 'default': True},
                        'red_shift': {'type': 'float', 'min': -5, 'max': 5, 'default': 0},
                        'blue_shift': {'type': 'float', 'min': -5, 'max': 5, 'default': 0}
                    },
                    performance={'speed': 0.9, 'quality': 0.95}
                )
            ],
            semantic_tags=['correction', 'chromatic', 'aberration', 'quality'],
            constraints={}
        ))
        
        # Analysis tools
        self.register_tool(OpticalTool(
            id='lens_analyzer',
            name='Lens Characteristic Analyzer',
            type=ToolType.ANALYSIS,
            description='Analyzes and extracts lens characteristics from images',
            capabilities=[
                ToolCapability(
                    name='extract_psf',
                    description='Extracts point spread function',
                    input_types=['image'],
                    output_types=['psf_data'],
                    parameters={
                        'method': {'type': 'string', 'options': ['blind', 'star', 'edge'], 'default': 'blind'}
                    }
                ),
                ToolCapability(
                    name='measure_vignetting',
                    description='Measures vignetting profile',
                    input_types=['image'],
                    output_types=['vignetting_map'],
                    parameters={}
                )
            ],
            semantic_tags=['analysis', 'measurement', 'calibration'],
            constraints={}
        ))
    
    def register_tool(self, tool: OpticalTool):
        """Register a new tool in the registry."""
        if tool.id in self.tools:
            self.logger.warning(f"Overwriting existing tool: {tool.id}")
        
        self.tools[tool.id] = tool
        self.logger.info(f"Registered tool: {tool.name} ({tool.id})")
    
    def get_tool(self, tool_id: str) -> Optional[OpticalTool]:
        """Get a specific tool by ID."""
        return self.tools.get(tool_id)
    
    def search_tools(self, query: str, 
                    tool_type: Optional[ToolType] = None,
                    tags: Optional[List[str]] = None,
                    limit: int = 10) -> List[OpticalTool]:
        """Search for tools matching criteria."""
        results = []
        
        for tool in self.tools.values():
            # Filter by type if specified
            if tool_type and tool.type != tool_type:
                continue
            
            # Filter by tags if specified
            if tags:
                tag_match = any(tag in tool.semantic_tags for tag in tags)
                if not tag_match:
                    continue
            
            # Calculate relevance score
            score = tool.matches_query(query) if query else 1.0
            
            if score > 0:
                results.append((score, tool))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in results[:limit]]
    
    def find_tools_by_capability(self, 
                                input_type: Optional[str] = None,
                                output_type: Optional[str] = None,
                                capability_name: Optional[str] = None) -> List[OpticalTool]:
        """Find tools with specific capabilities."""
        matching_tools = []
        
        for tool in self.tools.values():
            for cap in tool.capabilities:
                # Check input type
                if input_type and input_type not in cap.input_types:
                    continue
                
                # Check output type
                if output_type and output_type not in cap.output_types:
                    continue
                
                # Check capability name
                if capability_name and capability_name.lower() not in cap.name.lower():
                    continue
                
                matching_tools.append(tool)
                break  # Only add tool once
        
        return matching_tools
    
    def get_compatible_tools(self, output_type: str) -> List[OpticalTool]:
        """Find tools that can process a given output type."""
        return self.find_tools_by_capability(input_type=output_type)
    
    def export_registry(self, filename: str):
        """Export registry to JSON file."""
        registry_data = {
            'tools': [tool.to_dict() for tool in self.tools.values()],
            'metadata': {
                'version': '1.0',
                'tool_count': len(self.tools)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def import_registry(self, filename: str):
        """Import tools from JSON file."""
        with open(filename, 'r') as f:
            registry_data = json.load(f)
        
        for tool_data in registry_data.get('tools', []):
            # Reconstruct tool
            capabilities = []
            for cap_data in tool_data.get('capabilities', []):
                capabilities.append(ToolCapability(
                    name=cap_data['name'],
                    description=cap_data['description'],
                    input_types=cap_data['input_types'],
                    output_types=cap_data['output_types'],
                    parameters=cap_data.get('parameters', {}),
                    constraints=cap_data.get('constraints', {}),
                    performance=cap_data.get('performance', {})
                ))
            
            tool = OpticalTool(
                id=tool_data['id'],
                name=tool_data['name'],
                type=ToolType(tool_data['type']),
                description=tool_data['description'],
                capabilities=capabilities,
                semantic_tags=tool_data['semantic_tags'],
                constraints=tool_data.get('constraints', {}),
                metadata=tool_data.get('metadata', {})
            )
            
            self.register_tool(tool)
    
    def get_tool_chain_suggestion(self, goal: str) -> List[str]:
        """Suggest a tool chain to achieve a goal."""
        # This is a simple implementation - could be enhanced with ML
        suggestions = {
            'vintage portrait': ['lens_analyzer', 'helios_44_2', 'kodak_portra_400'],
            'clean modern': ['chromatic_corrector', 'sharpness_enhancer'],
            'film look': ['lens_analyzer', 'vintage_detector', 'kodak_portra_400'],
            'professional': ['chromatic_corrector', 'vignetting_corrector', 'color_calibrator']
        }
        
        goal_lower = goal.lower()
        for key, chain in suggestions.items():
            if key in goal_lower:
                # Validate that tools exist
                valid_chain = [tool_id for tool_id in chain if tool_id in self.tools]
                if valid_chain:
                    return valid_chain
        
        # Default suggestion
        return ['lens_analyzer']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        type_counts = {}
        tag_counts = {}
        capability_counts = {}
        
        for tool in self.tools.values():
            # Count by type
            type_name = tool.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count tags
            for tag in tool.semantic_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Count capabilities
            for cap in tool.capabilities:
                capability_counts[cap.name] = capability_counts.get(cap.name, 0) + 1
        
        return {
            'total_tools': len(self.tools),
            'tools_by_type': type_counts,
            'popular_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'common_capabilities': sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
