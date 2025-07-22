"""
Constraint specification system for optical physics.

Defines physical constraints that corrections must respect to maintain
optical realism and prevent physically impossible results.
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class PhysicalConstraint:
    """Represents a single physical constraint."""
    
    name: str
    description: str
    constraint_type: str  # 'boundary', 'relationship', 'conservation'
    validator: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value against this constraint."""
        if self.validator:
            return self.validator(value, **self.parameters)
        return True, None
    
    def to_dict(self) -> Dict:
        """Serialize constraint (excluding validator function)."""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.constraint_type,
            'parameters': self.parameters
        }


class ConstraintSpecification:
    """Define and validate physical constraints for lens behavior."""
    
    def __init__(self):
        self.constraints = self._initialize_constraints()
        
    def _initialize_constraints(self) -> Dict[str, PhysicalConstraint]:
        """Initialize standard optical constraints."""
        return {
            'spherical_aberration': PhysicalConstraint(
                name='spherical_aberration',
                description='Spherical aberration coefficient bounds',
                constraint_type='boundary',
                parameters={'min': 0, 'max': 5},
                validator=lambda x, min=0, max=5: (
                    (min <= x <= max, None) if min <= x <= max 
                    else (False, f"Spherical aberration {x} outside bounds [{min}, {max}]")
                )
            ),
            
            'chromatic_aberration': PhysicalConstraint(
                name='chromatic_aberration',
                description='Chromatic aberration bounds in pixels',
                constraint_type='boundary',
                parameters={'min': 0, 'max': 10},
                validator=lambda x, min=0, max=10: (
                    (min <= x <= max, None) if min <= x <= max
                    else (False, f"Chromatic aberration {x} outside bounds [{min}, {max}]")
                )
            ),
            
            'vignetting_profile': PhysicalConstraint(
                name='vignetting_profile',
                description='Vignetting must follow smooth radial decay',
                constraint_type='relationship',
                parameters={'max_edge_ratio': 0.3, 'smoothness': 2.0},
                validator=self._validate_vignetting_profile
            ),
            
            'diffraction_limit': PhysicalConstraint(
                name='diffraction_limit',
                description='Resolution limited by aperture diffraction',
                constraint_type='relationship',
                parameters={'wavelength': 550e-9},  # Green light in meters
                validator=self._validate_diffraction_limit
            ),
            
            'energy_conservation': PhysicalConstraint(
                name='energy_conservation',
                description='Total light energy must be conserved',
                constraint_type='conservation',
                parameters={'tolerance': 0.02},  # 2% tolerance
                validator=self._validate_energy_conservation
            )
        }
    
    def _validate_vignetting_profile(self, profile: np.ndarray, 
                                    max_edge_ratio: float = 0.3,
                                    smoothness: float = 2.0) -> Tuple[bool, Optional[str]]:
        """Validate that vignetting follows realistic radial decay."""
        if profile.ndim != 2:
            return False, "Vignetting profile must be 2D"
        
        # Check radial symmetry
        center = np.array(profile.shape) // 2
        y, x = np.ogrid[:profile.shape[0], :profile.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Check edge darkness
        edge_mean = profile[r > r.max() * 0.9].mean()
        center_mean = profile[r < r.max() * 0.1].mean()
        
        if edge_mean / center_mean < max_edge_ratio:
            return False, f"Vignetting too strong: edge/center ratio {edge_mean/center_mean:.2f}"
        
        return True, None
    
    def _validate_diffraction_limit(self, resolution: float, f_stop: float,
                                   wavelength: float = 550e-9) -> Tuple[bool, Optional[str]]:
        """Validate that resolution respects diffraction limits."""
        # Rayleigh criterion for circular aperture
        theoretical_limit = 1.22 * wavelength * f_stop
        
        if resolution < theoretical_limit:
            return False, f"Resolution {resolution} exceeds diffraction limit {theoretical_limit}"
        
        return True, None
    
    def _validate_energy_conservation(self, original: np.ndarray, 
                                    processed: np.ndarray,
                                    tolerance: float = 0.02) -> Tuple[bool, Optional[str]]:
        """Validate that total image energy is conserved."""
        original_energy = np.sum(original.astype(float))
        processed_energy = np.sum(processed.astype(float))
        
        ratio = processed_energy / (original_energy + 1e-10)
        
        if abs(1.0 - ratio) > tolerance:
            return False, f"Energy not conserved: ratio {ratio:.3f} exceeds tolerance {tolerance}"
        
        return True, None
    
    def add_constraint(self, constraint: PhysicalConstraint):
        """Add a custom constraint."""
        self.constraints[constraint.name] = constraint
    
    def validate_correction(self, original: np.ndarray, 
                          corrected: np.ndarray,
                          metadata: Dict[str, Any]) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Validate that corrections respect all physical constraints."""
        results = {}
        
        # Check applicable constraints based on metadata
        if 'vignetting_map' in metadata:
            results['vignetting_profile'] = self.constraints['vignetting_profile'].validate(
                metadata['vignetting_map']
            )
        
        if 'f_stop' in metadata and 'resolution' in metadata:
            results['diffraction_limit'] = self.constraints['diffraction_limit'].validate(
                metadata['resolution'], metadata['f_stop']
            )
        
        # Always check energy conservation
        results['energy_conservation'] = self.constraints['energy_conservation'].validate(
            original, corrected
        )
        
        # Check aberration bounds if provided
        for aberration in ['spherical_aberration', 'chromatic_aberration']:
            if aberration in metadata:
                results[aberration] = self.constraints[aberration].validate(
                    metadata[aberration]
                )
        
        return results
    
    def get_constraint_bounds(self, constraint_name: str) -> Dict[str, Any]:
        """Get the parameter bounds for a specific constraint."""
        if constraint_name in self.constraints:
            return self.constraints[constraint_name].parameters
        return {}
    
    def export_constraints(self, filename: str):
        """Export constraints to JSON (for sharing/versioning)."""
        export_data = {
            name: constraint.to_dict() 
            for name, constraint in self.constraints.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def validate_pipeline(self, pipeline_spec: Dict[str, Any]) -> bool:
        """Validate that a processing pipeline respects constraints."""
        # This will be used by the task graph system
        # For now, return True as a skeleton
        return True
