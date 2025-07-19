"""
VintageOptics Complete Processing Pipeline
Full implementation with all processing tools
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter

# Import the perceptual defect filter
try:
    from perceptual_defect_filter import PerceptualDefectFilter
    PERCEPTUAL_FILTER_AVAILABLE = True
except ImportError:
    PERCEPTUAL_FILTER_AVAILABLE = False
    print("Warning: Perceptual defect filter not available")

# Optional imports with fallbacks
try:
    from skimage import restoration, morphology, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==================== HYPERDIMENSIONAL COMPUTING ====================

class HyperdimensionalProcessor:
    """Core hyperdimensional computing for VintageOptics."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.seed_vintage = 42
        self.seed_digital = 137
        
        # Initialize orthogonal base vectors
        self._init_base_vectors()
        
    def _init_base_vectors(self):
        """Initialize orthogonal base vectors for different error types."""
        # Vintage errors (continuous, physics-based)
        np.random.seed(self.seed_vintage)
        self.vintage_bases = {
            'optical': self._create_smooth_vector(),
            'mechanical': self._create_smooth_vector(),
            'chemical': self._create_smooth_vector(),
            'physical': self._create_smooth_vector()
        }
        
        # Digital errors (discrete, sensor-based)
        np.random.seed(self.seed_digital)
        self.digital_bases = {
            'quantization': self._create_sparse_vector(),
            'noise': self._create_sparse_vector(),
            'compression': self._create_sparse_vector(),
            'sensor': self._create_sparse_vector()
        }
        
    def _create_smooth_vector(self) -> np.ndarray:
        """Create smooth HD vector for analog errors."""
        base = np.random.randn(self.dimension)
        # Apply low-pass filter for smoothness
        return ndimage.gaussian_filter1d(base, sigma=10)
        
    def _create_sparse_vector(self) -> np.ndarray:
        """Create sparse HD vector for digital errors."""
        vec = np.zeros(self.dimension)
        # Only ~5% non-zero elements
        indices = np.random.choice(self.dimension, size=int(0.05 * self.dimension), replace=False)
        vec[indices] = np.random.randn(len(indices)) * 10
        return vec
        
    def encode_image_errors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Encode image errors into HD space."""
        h, w = image.shape[:2]
        
        # Extract error patterns
        vintage_errors = self._extract_vintage_errors(image)
        digital_errors = self._extract_digital_errors(image)
        
        # Encode to HD vectors
        hd_vintage = self._encode_to_hd(vintage_errors, self.vintage_bases)
        hd_digital = self._encode_to_hd(digital_errors, self.digital_bases)
        
        return {
            'vintage_hd': hd_vintage,
            'digital_hd': hd_digital,
            'combined_hd': self._bind_vectors(hd_vintage, hd_digital),
            'vintage_confidence': self._calculate_confidence(hd_vintage, self.vintage_bases),
            'digital_confidence': self._calculate_confidence(hd_digital, self.digital_bases)
        }
        
    def _extract_vintage_errors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract vintage/analog error patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        return {
            'vignetting': self._detect_vignetting(gray),
            'optical_aberrations': self._detect_aberrations(image),
            'coating_defects': self._detect_coating_issues(gray),
            'mechanical_vignetting': self._detect_mechanical_vignetting(gray)
        }
        
    def _extract_digital_errors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract digital/sensor error patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        return {
            'noise_pattern': self._detect_sensor_noise(gray),
            'quantization': self._detect_quantization(gray),
            'compression_artifacts': self._detect_compression(image),
            'dead_pixels': self._detect_dead_pixels(gray)
        }
        
    def _detect_vignetting(self, gray: np.ndarray) -> np.ndarray:
        """Detect vignetting pattern."""
        h, w = gray.shape
        
        # Create radial sampling
        y, x = np.ogrid[:h, :w]
        center = (h/2, w/2)
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Normalize and bin radial distances
        max_radius = np.sqrt(center[0]**2 + center[1]**2)
        radius_norm = radius / max_radius
        
        # Calculate radial intensity profile
        bins = np.linspace(0, 1, 50)
        profile = []
        for i in range(len(bins)-1):
            mask = (radius_norm >= bins[i]) & (radius_norm < bins[i+1])
            if np.any(mask):
                profile.append(np.mean(gray[mask]))
            else:
                profile.append(0)
                
        return np.array(profile)
        
    def _bind_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Bind two HD vectors using circular convolution."""
        return np.real(np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)))
        
    def separate_errors(self, hd_encoding: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Separate vintage and digital errors using orthogonality."""
        vintage_hd = hd_encoding['vintage_hd']
        digital_hd = hd_encoding['digital_hd']
        combined = hd_encoding['combined_hd']
        
        # Use orthogonality to separate
        vintage_component = self._project_onto_basis(combined, self.vintage_bases)
        digital_component = self._project_onto_basis(combined, self.digital_bases)
        
        # Reconstruct error maps
        vintage_errors = self._reconstruct_from_hd(vintage_component)
        digital_errors = self._reconstruct_from_hd(digital_component)
        
        return {
            'vintage_errors': vintage_errors,
            'digital_errors': digital_errors,
            'separation_confidence': self._calculate_separation_confidence(
                vintage_component, digital_component, combined
            )
        }
    
    # Helper methods for HD processing
    def _detect_aberrations(self, image: np.ndarray) -> np.ndarray:
        """Detect optical aberrations."""
        return np.zeros(10)  # Placeholder
        
    def _detect_coating_issues(self, gray: np.ndarray) -> np.ndarray:
        """Detect coating issues."""
        return np.zeros(10)  # Placeholder
        
    def _detect_mechanical_vignetting(self, gray: np.ndarray) -> np.ndarray:
        """Detect mechanical vignetting."""
        return self._detect_vignetting(gray)
        
    def _detect_sensor_noise(self, gray: np.ndarray) -> np.ndarray:
        """Detect sensor noise pattern."""
        return np.array([np.std(gray)])
        
    def _detect_quantization(self, gray: np.ndarray) -> np.ndarray:
        """Detect quantization artifacts."""
        return np.zeros(10)  # Placeholder
        
    def _detect_compression(self, image: np.ndarray) -> np.ndarray:
        """Detect compression artifacts."""
        return np.zeros(10)  # Placeholder
        
    def _detect_dead_pixels(self, gray: np.ndarray) -> np.ndarray:
        """Detect dead pixels."""
        return np.zeros(10)  # Placeholder
        
    def _encode_to_hd(self, errors: Dict, bases: Dict) -> np.ndarray:
        """Encode errors to HD vector."""
        hd_vector = np.zeros(self.dimension)
        for key in bases:
            if key in errors:
                if isinstance(errors[key], np.ndarray):
                    hd_vector += bases[key] * np.mean(errors[key])
        norm = np.linalg.norm(hd_vector)
        return hd_vector / norm if norm > 0 else hd_vector
        
    def _calculate_confidence(self, hd_vector: np.ndarray, bases: Dict) -> float:
        """Calculate confidence score."""
        scores = []
        for base in bases.values():
            norm_hd = np.linalg.norm(hd_vector)
            norm_base = np.linalg.norm(base)
            if norm_hd > 0 and norm_base > 0:
                scores.append(np.dot(hd_vector, base) / (norm_hd * norm_base))
        return np.mean(np.abs(scores)) if scores else 0.0
        
    def _project_onto_basis(self, vector: np.ndarray, bases: Dict) -> np.ndarray:
        """Project vector onto basis."""
        projection = np.zeros_like(vector)
        for base in bases.values():
            projection += np.dot(vector, base) * base
        return projection
        
    def _reconstruct_from_hd(self, hd_vector: np.ndarray) -> np.ndarray:
        """Reconstruct error map from HD vector."""
        # Simplified reconstruction
        return np.abs(hd_vector[:100].reshape(10, 10))
        
    def _calculate_separation_confidence(self, v1: np.ndarray, v2: np.ndarray, combined: np.ndarray) -> float:
        """Calculate separation confidence."""
        reconstruction = v1 + v2
        norm_combined = np.linalg.norm(combined)
        if norm_combined > 0:
            error = np.linalg.norm(combined - reconstruction) / norm_combined
            return 1.0 - error
        return 0.0


# ==================== PHYSICS ENGINE ====================

class OpticsPhysicsEngine:
    """Physical optics simulation engine."""
    
    def __init__(self):
        self.wavelengths = {
            'red': 650e-9,
            'green': 550e-9, 
            'blue': 450e-9
        }
        
    def apply_lens_model(self, image: np.ndarray, lens_params: Dict) -> np.ndarray:
        """Apply physical lens model to image."""
        result = image.copy()
        
        # Apply distortion
        if 'distortion' in lens_params:
            result = self._apply_distortion(result, lens_params['distortion'])
            
        # Apply chromatic aberration
        if 'chromatic' in lens_params:
            result = self._apply_chromatic_aberration(result, lens_params['chromatic'])
            
        # Apply vignetting
        if 'vignetting' in lens_params:
            result = self._apply_vignetting(result, lens_params['vignetting'])
            
        # Apply diffraction
        if 'aperture' in lens_params:
            result = self._apply_diffraction(result, lens_params['aperture'])
            
        return result
        
    def _apply_distortion(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply Brown-Conrady distortion model."""
        h, w = image.shape[:2]
        
        # Camera matrix
        fx = fy = max(w, h)
        cx, cy = w/2, h/2
        
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([
            params.get('k1', 0),
            params.get('k2', 0),
            params.get('p1', 0),
            params.get('p2', 0),
            params.get('k3', 0)
        ], dtype=np.float32)
        
        # Apply distortion
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, -dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
        )
        
        return cv2.remap(image, map1, map2, cv2.INTER_CUBIC)
        
    def _apply_chromatic_aberration(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply wavelength-dependent aberrations."""
        b, g, r = cv2.split(image)
        h, w = image.shape[:2]
        
        # Lateral chromatic aberration
        scale_r = 1 + params.get('lateral_scale', 0.002)
        scale_b = 1 - params.get('lateral_scale', 0.002)
        
        # Create scaling matrices
        M_r = cv2.getRotationMatrix2D((w/2, h/2), 0, scale_r)
        M_b = cv2.getRotationMatrix2D((w/2, h/2), 0, scale_b)
        
        # Apply scaling
        r_scaled = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b_scaled = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Longitudinal chromatic aberration (focus shift)
        if params.get('longitudinal', 0) > 0:
            sigma = params['longitudinal']
            r_scaled = cv2.GaussianBlur(r_scaled, (0, 0), sigma * 1.5)
            b_scaled = cv2.GaussianBlur(b_scaled, (0, 0), sigma * 0.5)
            
        return cv2.merge([b_scaled, g, r_scaled])
        
    def _apply_diffraction(self, image: np.ndarray, f_number: float) -> np.ndarray:
        """Apply diffraction effects based on aperture."""
        # Airy disk radius in pixels
        airy_radius = 1.22 * 550e-9 * f_number / (4e-6)  # Assuming 4μm pixel pitch
        
        if airy_radius > 0.5:
            # Apply diffraction blur
            kernel_size = int(airy_radius * 4) | 1  # Ensure odd
            kernel = self._create_airy_kernel(kernel_size, airy_radius)
            return cv2.filter2D(image, -1, kernel)
            
        return image
        
    def _create_airy_kernel(self, size: int, radius: float) -> np.ndarray:
        """Create Airy disk kernel for diffraction."""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Airy function: (2 * J1(x) / x)^2
        # Approximation for simplicity
        kernel = np.zeros((size, size))
        mask = r <= radius * 3
        kernel[mask] = (1 - (r[mask] / (radius * 3))**2)**2
        
        return kernel / kernel.sum()
    
    def _apply_vignetting(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply vignetting effect."""
        h, w = image.shape[:2]
        
        # Get vignetting parameters
        amount = params.get('amount', 0.3)
        falloff = params.get('falloff', 2.0)
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Apply vignetting
        vignette = 1 - (dist / max_dist) ** falloff * amount
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to all channels
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette
            
        return np.clip(result, 0, 255).astype(np.uint8)


# ==================== DEFECT DETECTION & CORRECTION ====================

class DefectAnalyzer:
    """Analyzes and characterizes lens defects."""
    
    def __init__(self):
        self.defect_types = ['dust', 'scratches', 'fungus', 'haze', 'separation', 'coating']
        self.perceptual_filter = PerceptualDefectFilter() if PERCEPTUAL_FILTER_AVAILABLE else None
        
    def analyze_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive defect analysis with perceptual filtering."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Get raw defect detections
        raw_results = {
            'dust': self._detect_dust(gray),
            'scratches': self._detect_scratches(gray),
            'fungus': self._detect_fungus(gray),
            'haze': self._measure_haze(gray),
            'separation': self._detect_separation(image),
            'coating': self._detect_coating_damage(image)
        }
        
        # Apply perceptual filtering if available
        if self.perceptual_filter is not None:
            results = self.perceptual_filter.filter_defects(image, raw_results)
        else:
            results = raw_results
            # Calculate overall defect score
            results['total_defect_score'] = self._calculate_defect_score(results)
        
        return results
        
    def _detect_dust(self, gray: np.ndarray) -> Dict:
        """Detect dust particles using morphological operations."""
        # High-pass filter to enhance small features
        highpass = gray - cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Threshold to find dark spots
        _, binary = cv2.threshold(highpass, -20, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to isolate dust
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dust_spots = []
        h, w = gray.shape
        total_pixels = h * w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Adjusted size filter - dust should be small isolated spots
            if 5 < area < min(100, total_pixels * 0.0001):  # Max 0.01% of image
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Check if it's isolated (not part of a texture)
                    roi_size = 20
                    x1, y1 = max(0, cx-roi_size), max(0, cy-roi_size)
                    x2, y2 = min(w, cx+roi_size), min(h, cy+roi_size)
                    
                    if x2 > x1 and y2 > y1:
                        roi = gray[y1:y2, x1:x2]
                        # Check if this spot is significantly different from surroundings
                        spot_value = gray[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0
                        roi_mean = np.mean(roi)
                        roi_std = np.std(roi)
                        
                        # Only keep if it's an outlier in the local region
                        if abs(spot_value - roi_mean) > 2 * roi_std:
                            dust_spots.append({
                                'position': (cx, cy),
                                'area': area,
                                'intensity': spot_value
                            })
        
        # Sanity check: if we detect too many dust spots, we're probably detecting texture
        if len(dust_spots) > 200:
            # Keep only the most significant ones
            dust_spots.sort(key=lambda x: x['area'], reverse=True)
            dust_spots = dust_spots[:50]
                    
        return {
            'count': len(dust_spots),
            'locations': dust_spots,
            'severity': min(len(dust_spots) / 50, 1.0)  # Normalize to 0-1
        }
        
    def _detect_scratches(self, gray: np.ndarray) -> Dict:
        """Detect scratches using line detection."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        scratches = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1)
                
                # Filter for scratch-like lines
                if length > 50:  # Minimum length
                    scratches.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': angle
                    })
                    
        return {
            'count': len(scratches),
            'scratches': scratches,
            'severity': min(len(scratches) / 10, 1.0)
        }
        
    def _detect_fungus(self, gray: np.ndarray) -> Dict:
        """Detect fungus patterns using texture analysis."""
        # Gabor filters for texture
        kernels = []
        for theta in np.arange(0, np.pi, np.pi/4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
            kernels.append(kernel)
            
        # Apply Gabor filters
        responses = []
        for kernel in kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(filtered)
            
        # Combine responses
        texture_map = np.mean(responses, axis=0)
        
        # Threshold to find textured regions
        _, binary = cv2.threshold(texture_map, texture_map.mean() + texture_map.std(), 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
        
        fungus_regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100:  # Minimum area for fungus
                fungus_regions.append({
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': stats[i, :4]
                })
                
        return {
            'count': len(fungus_regions),
            'regions': fungus_regions,
            'severity': min(sum(r['area'] for r in fungus_regions) / (gray.shape[0] * gray.shape[1]), 1.0)
        }
    
    def _measure_haze(self, gray: np.ndarray) -> Dict:
        """Measure haze level."""
        # Haze reduces contrast globally
        contrast = gray.std() / 255.0
        haze_level = 1.0 - contrast
        
        return {
            'level': haze_level,
            'severity': min(haze_level, 1.0)
        }
    
    def _detect_separation(self, image: np.ndarray) -> Dict:
        """Detect element separation (balsam separation)."""
        # Element separation often shows as rainbow patterns
        if len(image.shape) == 3:
            # Check for unusual color variations
            color_variance = np.std([image[:,:,i].std() for i in range(3)])
            separation_score = color_variance / 50.0
        else:
            separation_score = 0.0
            
        return {
            'detected': separation_score > 0.1,
            'severity': min(separation_score, 1.0)
        }
    
    def _detect_coating_damage(self, image: np.ndarray) -> Dict:
        """Detect coating damage."""
        # Coating damage often appears as uneven reflections
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for high-frequency variations
        highpass = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        damage_score = np.std(highpass) / 255.0
        
        return {
            'detected': damage_score > 0.1,
            'severity': min(damage_score, 1.0)
        }
    
    def _calculate_defect_score(self, results: Dict) -> float:
        """Calculate overall defect score."""
        scores = []
        for defect_type, data in results.items():
            if isinstance(data, dict) and 'severity' in data:
                scores.append(data['severity'])
        return np.mean(scores) if scores else 0.0


# ==================== BOKEH SYNTHESIS ====================

class BokehSynthesizer:
    """Synthesizes realistic bokeh effects."""
    
    def __init__(self):
        self.aperture_shapes = {
            'circular': self._create_circular_aperture,
            'hexagonal': self._create_hexagonal_aperture,
            'octagonal': self._create_octagonal_aperture,
            'cats_eye': self._create_cats_eye_aperture
        }
        
    def synthesize_bokeh(self, image: np.ndarray, depth_map: np.ndarray, 
                        lens_params: Dict) -> np.ndarray:
        """Synthesize depth-dependent bokeh."""
        
        # Get aperture shape
        aperture_type = lens_params.get('aperture_shape', 'circular')
        aperture_blades = lens_params.get('aperture_blades', 8)
        
        # Create aperture kernel
        kernel_size = lens_params.get('bokeh_size', 21)
        aperture = self.aperture_shapes[aperture_type](kernel_size, aperture_blades)
        
        # Apply depth-dependent blur
        result = self._apply_depth_blur(image, depth_map, aperture, lens_params)
        
        # Add bokeh characteristics
        if lens_params.get('swirly_bokeh', False):
            result = self._add_swirly_bokeh(result, depth_map)
            
        if lens_params.get('bokeh_fringing', False):
            result = self._add_bokeh_fringing(result, depth_map)
            
        return result
        
    def _create_circular_aperture(self, size: int, blades: int = None) -> np.ndarray:
        """Create circular aperture."""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= (center-1)**2
        
        aperture = np.zeros((size, size))
        aperture[mask] = 1
        
        return aperture / aperture.sum()
        
    def _create_octagonal_aperture(self, size: int, blades: int = 8) -> np.ndarray:
        """Create octagonal aperture."""
        # Create polygon aperture with 8 sides
        center = size // 2
        radius = center - 1
        
        aperture = np.zeros((size, size))
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        # Create polygon vertices
        vertices = []
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            vertices.append([x, y])
            
        vertices = np.array(vertices, np.int32)
        cv2.fillPoly(aperture, [vertices], 1)
        
        return aperture / aperture.sum() if aperture.sum() > 0 else aperture
        
    def _create_cats_eye_aperture(self, size: int, blades: int = None) -> np.ndarray:
        """Create cat's eye aperture (mechanical vignetting)."""
        # Start with circular aperture
        aperture = self._create_circular_aperture(size)
        
        # Add mechanical vignetting (cut off edges)
        h, w = aperture.shape
        cut_amount = int(size * 0.15)
        aperture[:cut_amount, :] = 0
        aperture[-cut_amount:, :] = 0
        
        return aperture / aperture.sum() if aperture.sum() > 0 else aperture
        
    def _create_hexagonal_aperture(self, size: int, blades: int = 6) -> np.ndarray:
        """Create hexagonal aperture."""
        # Create polygon aperture
        center = size // 2
        radius = center - 1
        
        aperture = np.zeros((size, size))
        angles = np.linspace(0, 2*np.pi, blades, endpoint=False)
        
        # Create polygon vertices
        vertices = []
        for angle in angles:
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            vertices.append([x, y])
            
        vertices = np.array(vertices, np.int32)
        cv2.fillPoly(aperture, [vertices], 1)
        
        return aperture / aperture.sum()
        
    def _apply_depth_blur(self, image: np.ndarray, depth_map: np.ndarray,
                         aperture: np.ndarray, params: Dict) -> np.ndarray:
        """Apply depth-dependent blur with aperture shape."""
        
        # Normalize depth map
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Define focus plane
        focus_distance = params.get('focus_distance', 0.5)
        dof_scale = params.get('dof_scale', 0.1)
        
        # Calculate blur amount for each pixel
        blur_map = np.abs(depth_norm - focus_distance) / dof_scale
        blur_map = np.clip(blur_map, 0, 1)
        
        # Apply variable blur
        # For efficiency, quantize to several blur levels
        blur_levels = 5
        result = image.copy()
        
        for i in range(blur_levels):
            blur_amount = i / (blur_levels - 1)
            mask = (blur_map >= blur_amount - 0.1) & (blur_map < blur_amount + 0.1)
            
            if np.any(mask):
                # Scale aperture
                kernel_size = int(3 + blur_amount * 18) | 1
                kernel = cv2.resize(aperture, (kernel_size, kernel_size))
                kernel = kernel / kernel.sum()
                
                # Apply blur to masked region
                blurred = cv2.filter2D(image, -1, kernel)
                result[mask] = blurred[mask]
                
        return result
    
    def _add_swirly_bokeh(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Add swirly bokeh effect (Petzval-style)."""
        h, w = image.shape[:2]
        result = image.copy()
        
        # Create radial swirl pattern
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        
        # Calculate angle and radius for each pixel
        angle = np.arctan2(y - center_y, x - center_x)
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        
        # Swirl amount increases with radius
        swirl_amount = (radius / max_radius) * 0.3
        
        # Apply swirl to out-of-focus areas
        blur_mask = depth_map > 0.7  # Background areas
        
        if np.any(blur_mask):
            # Create swirled coordinates
            new_angle = angle + swirl_amount
            new_x = center_x + radius * np.cos(new_angle)
            new_y = center_y + radius * np.sin(new_angle)
            
            # Remap the blurred areas
            map_x = new_x.astype(np.float32)
            map_y = new_y.astype(np.float32)
            
            swirled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
            result[blur_mask] = swirled[blur_mask]
            
        return result
    
    def _add_bokeh_fringing(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Add chromatic aberration to bokeh (purple/green fringing)."""
        b, g, r = cv2.split(image)
        
        # Create fringing in out-of-focus areas
        blur_mask = depth_map > 0.5
        
        if np.any(blur_mask):
            # Shift channels slightly
            kernel = np.ones((3, 3)) / 9
            r_shifted = cv2.filter2D(r, -1, kernel)
            b_shifted = cv2.filter2D(b, -1, kernel)
            
            # Apply color fringing
            r[blur_mask] = r_shifted[blur_mask]
            b[blur_mask] = b_shifted[blur_mask]
            
        return cv2.merge([b, g, r])


# ==================== QUALITY ANALYSIS ====================

class QualityAnalyzer:
    """Analyzes image quality with vintage lens considerations."""
    
    def __init__(self):
        self.metrics = {
            'sharpness': self._measure_sharpness,
            'contrast': self._measure_contrast,
            'color_accuracy': self._measure_color_accuracy,
            'noise': self._measure_noise,
            'detail_preservation': self._measure_detail_preservation
        }
        
    def analyze(self, image: np.ndarray, reference: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive quality analysis."""
        results = {}
        
        # Basic metrics
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(image)
            
        # Reference-based metrics if available
        if reference is not None:
            results['psnr'] = cv2.PSNR(image, reference)
            results['ssim'] = self._calculate_ssim(image, reference)
            
        # Vintage-specific quality aspects
        results['character_preservation'] = self._measure_character_preservation(image)
        results['bokeh_quality'] = self._measure_bokeh_quality(image)
        
        # Overall score
        results['overall_quality'] = self._calculate_overall_score(results)
        
        return results
        
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness using gradient magnitude."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize to 0-1
        return min(sharpness / 1000, 1.0)
        
    def _measure_bokeh_quality(self, image: np.ndarray) -> float:
        """Measure bokeh quality (smoothness of out-of-focus areas)."""
        # Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Measure smoothness in non-edge areas
        non_edge_mask = edges == 0
        if np.any(non_edge_mask):
            # Calculate local variance in non-edge areas
            local_var = ndimage.generic_filter(gray, np.var, size=5)
            smoothness = 1.0 - (np.mean(local_var[non_edge_mask]) / 255)
            return smoothness
            
        return 0.5
    
    def _measure_contrast(self, image: np.ndarray) -> float:
        """Measure image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return gray.std() / 128.0  # Normalized to roughly 0-1
    
    def _measure_color_accuracy(self, image: np.ndarray) -> float:
        """Measure color accuracy/saturation."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Measure chroma
            a_channel = lab[:, :, 1].astype(np.float32) - 128
            b_channel = lab[:, :, 2].astype(np.float32) - 128
            chroma = np.sqrt(a_channel**2 + b_channel**2)
            return np.mean(chroma) / 100.0  # Normalized
        return 0.5
    
    def _measure_noise(self, image: np.ndarray) -> float:
        """Measure image noise level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Estimate noise using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = np.std(laplacian)
        return min(noise / 50.0, 1.0)  # Normalized and capped
    
    def _measure_detail_preservation(self, image: np.ndarray) -> float:
        """Measure fine detail preservation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # High-frequency content
        highpass = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        detail_energy = np.mean(np.abs(highpass))
        return min(detail_energy / 20.0, 1.0)  # Normalized
    
    def _calculate_ssim(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate SSIM between two images."""
        # Simple SSIM implementation
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
        # Constants
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Mean and variance
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    def _measure_character_preservation(self, image: np.ndarray) -> float:
        """Measure preservation of vintage lens character."""
        # This is a subjective metric - we look for subtle imperfections
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Check for too-perfect uniformity (sign of over-correction)
        local_variance = ndimage.generic_filter(gray, np.var, size=9)
        variance_uniformity = 1.0 - (np.std(local_variance) / np.mean(local_variance) if np.mean(local_variance) > 0 else 0)
        
        # Check for natural vignetting
        h, w = gray.shape
        center_brightness = np.mean(gray[h//3:2*h//3, w//3:2*w//3])
        corner_brightness = np.mean([gray[:h//4, :w//4], gray[:h//4, -w//4:], 
                                   gray[-h//4:, :w//4], gray[-h//4:, -w//4:]]) / 4
        vignetting_preserved = 1.0 - abs(center_brightness - corner_brightness) / 255.0
        
        return (variance_uniformity + vignetting_preserved) / 2.0
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall quality score."""
        # Weighted average of different metrics
        weights = {
            'sharpness': 0.25,
            'contrast': 0.15,
            'color_accuracy': 0.15,
            'noise': 0.15,
            'detail_preservation': 0.2,
            'character_preservation': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results:
                value = results[metric]
                if metric == 'noise':  # Lower is better for noise
                    value = 1.0 - value
                score += value * weight
                total_weight += weight
                
        return score / total_weight if total_weight > 0 else 0.5


# ==================== MAIN PIPELINE INTEGRATION ====================

class VintageOpticsProcessor:
    """Main processor integrating all components."""
    
    def __init__(self):
        self.hd_processor = HyperdimensionalProcessor()
        self.physics_engine = OpticsPhysicsEngine()
        self.defect_analyzer = DefectAnalyzer()
        self.bokeh_synthesizer = BokehSynthesizer()
        self.quality_analyzer = QualityAnalyzer()
        
    def process_image(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Complete processing pipeline."""
        
        # 1. HD Analysis
        hd_encoding = self.hd_processor.encode_image_errors(image)
        error_separation = self.hd_processor.separate_errors(hd_encoding)
        
        # 2. Defect Analysis
        defects = self.defect_analyzer.analyze_defects(image)
        
        # 3. Apply lens model
        lens_params = params.get('lens_params', {})
        processed = self.physics_engine.apply_lens_model(image, lens_params)
        
        # 4. Synthesize bokeh if depth map available
        if 'depth_map' in params:
            processed = self.bokeh_synthesizer.synthesize_bokeh(
                processed, params['depth_map'], lens_params
            )
            
        # 5. Correction based on mode
        mode = params.get('mode', 'hybrid')
        if mode == 'correction':
            corrected = self._apply_correction(processed, error_separation, defects)
        elif mode == 'synthesis':
            corrected = processed  # Pure synthesis, no correction
        else:  # hybrid
            corrected = self._apply_hybrid_correction(processed, error_separation, defects)
            
        # 6. Quality analysis
        quality = self.quality_analyzer.analyze(corrected, image)
        
        return {
            'corrected_image': corrected,
            'hd_analysis': hd_encoding,
            'error_separation': error_separation,
            'defects': defects,
            'quality_metrics': quality,
            'processing_params': params
        }
        
    def _apply_correction(self, image: np.ndarray, errors: Dict, defects: Dict) -> np.ndarray:
        """Apply full correction mode."""
        result = image.copy()
        
        # Remove vintage errors
        if 'vintage_errors' in errors:
            result = self._remove_vintage_errors(result, errors['vintage_errors'])
            
        # Remove digital errors
        if 'digital_errors' in errors:
            result = self._remove_digital_errors(result, errors['digital_errors'])
            
        # Correct specific defects
        if defects['dust']['count'] > 0:
            result = self._remove_dust(result, defects['dust']['locations'])
            
        if defects['scratches']['count'] > 0:
            result = self._remove_scratches(result, defects['scratches']['scratches'])
            
        return result
        
    def _apply_hybrid_correction(self, image: np.ndarray, errors: Dict, defects: Dict) -> np.ndarray:
        """Apply hybrid correction - preserve character while fixing defects."""
        result = image.copy()
        
        # Selectively remove only severe defects
        if defects['dust']['severity'] > 0.5:
            result = self._remove_dust(result, defects['dust']['locations'], strength=0.7)
            
        if defects['scratches']['severity'] > 0.7:
            result = self._remove_scratches(result, defects['scratches']['scratches'], strength=0.5)
            
        # Reduce but don't eliminate vintage character
        if 'vintage_errors' in errors:
            vintage_reduced = self._remove_vintage_errors(result, errors['vintage_errors'])
            result = cv2.addWeighted(result, 0.6, vintage_reduced, 0.4, 0)
            
        return result
        
    def _remove_dust(self, image: np.ndarray, dust_locations: List, strength: float = 1.0) -> np.ndarray:
        """Remove dust spots using inpainting."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for dust in dust_locations:
            cv2.circle(mask, dust['position'], int(np.sqrt(dust['area'])) + 2, 255, -1)
            
        if strength < 1.0:
            # Partial removal
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            return cv2.addWeighted(image, 1-strength, inpainted, strength, 0)
        else:
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    def _remove_scratches(self, image: np.ndarray, scratches: List, strength: float = 1.0) -> np.ndarray:
        """Remove scratches using inpainting."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for scratch in scratches:
            # Draw line on mask
            cv2.line(mask, scratch['start'], scratch['end'], 255, 3)
            
        if strength < 1.0:
            # Partial removal
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            return cv2.addWeighted(image, 1-strength, inpainted, strength, 0)
        else:
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    def _remove_vintage_errors(self, image: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """Remove vintage lens errors."""
        # Simple approach: apply inverse correction based on error map
        # In practice, this would be more sophisticated
        correction_strength = 0.5
        
        # Create correction map from errors
        h, w = image.shape[:2]
        error_resized = cv2.resize(errors, (w, h))
        
        # Apply correction
        result = image.copy().astype(np.float32)
        correction = 1.0 - error_resized * correction_strength
        
        for i in range(3):
            result[:, :, i] *= correction
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _remove_digital_errors(self, image: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """Remove digital sensor errors."""
        # Apply noise reduction for digital errors
        result = image.copy()
        
        # Use bilateral filter to reduce noise while preserving edges
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result


# ==================== API INTEGRATION ====================

def create_processor():
    """Create configured processor instance."""
    return VintageOpticsProcessor()

def process_with_vintageoptics(image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Process image with VintageOptics pipeline."""
    processor = create_processor()
    return processor.process_image(image, kwargs)
