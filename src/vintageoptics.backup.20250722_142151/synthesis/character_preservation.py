# src/vintageoptics/synthesis/character_preservation.py (continued)
            profile.flare_resistance = max(profile.flare_resistance, 0.9)
            profile.rendering_style = 'modern'
    
    def _apply_full_preservation(self, corrected: np.ndarray, 
                               original: np.ndarray,
                               profile: CharacterProfile) -> np.ndarray:
        """Apply full character preservation"""
        
        result = corrected.copy()
        strength = profile.character_strength
        
        # Preserve bokeh character in out-of-focus areas
        blur_map = self._detect_blur_regions(original)
        if np.any(blur_map > 0.1):
            bokeh_preserved = self._preserve_bokeh_character(
                corrected, original, blur_map, profile
            )
            result = result * (1 - blur_map[:, :, np.newaxis]) + \
                    bokeh_preserved * blur_map[:, :, np.newaxis]
        
        # Preserve rendering character
        result = self._preserve_rendering_character(result, original, profile, strength)
        
        # Preserve vignetting character
        result = self._preserve_vignetting_character(result, original, profile, strength)
        
        # Preserve aberration character
        result = self._preserve_aberration_character(result, original, profile, strength)
        
        return result
    
    def _apply_selective_preservation(self, corrected: np.ndarray,
                                    original: np.ndarray,
                                    profile: CharacterProfile,
                                    depth_map: Optional[np.ndarray]) -> np.ndarray:
        """Apply selective character preservation based on image regions"""
        
        result = corrected.copy()
        
        # Create preservation mask based on image content
        preservation_mask = self._create_preservation_mask(
            original, depth_map, profile
        )
        
        # Apply different preservation strengths to different regions
        if profile.bokeh_quality > 0.7 or profile.swirl_factor > 0.3:
            # Preserve bokeh in background
            if depth_map is not None:
                bokeh_mask = depth_map > 0.6  # Far regions
                bokeh_preserved = self._preserve_bokeh_character(
                    corrected, original, bokeh_mask.astype(np.float32), profile
                )
                result = np.where(
                    bokeh_mask[:, :, np.newaxis],
                    bokeh_preserved,
                    result
                )
        
        # Preserve character in non-critical areas
        character_preserved = self._apply_full_preservation(
            corrected, original, profile
        )
        
        # Blend based on preservation mask
        result = result * (1 - preservation_mask[:, :, np.newaxis]) + \
                character_preserved * preservation_mask[:, :, np.newaxis]
        
        return result
    
    def _apply_adaptive_preservation(self, corrected: np.ndarray,
                                   original: np.ndarray,
                                   profile: CharacterProfile,
                                   depth_map: Optional[np.ndarray]) -> np.ndarray:
        """Apply adaptive preservation based on image analysis"""
        
        # Analyze image content
        content_analysis = self._analyze_image_content(corrected)
        
        # Determine preservation strategy
        if content_analysis['is_portrait']:
            # Portrait: preserve bokeh, minimal correction in background
            return self._apply_portrait_preservation(
                corrected, original, profile, depth_map
            )
        elif content_analysis['is_landscape']:
            # Landscape: preserve atmosphere, vignetting
            return self._apply_landscape_preservation(
                corrected, original, profile
            )
        elif content_analysis['has_architecture']:
            # Architecture: less distortion preservation
            return self._apply_architecture_preservation(
                corrected, original, profile
            )
        else:
            # General adaptive preservation
            return self._apply_selective_preservation(
                corrected, original, profile, depth_map
            )
    
    def _apply_artistic_preservation(self, corrected: np.ndarray,
                                   original: np.ndarray,
                                   profile: CharacterProfile) -> np.ndarray:
        """Apply artistic preservation - enhance character"""
        
        result = corrected.copy()
        
        # Enhance desirable characteristics
        if profile.swirl_factor > 0.3:
            # Enhance swirly bokeh
            result = self._enhance_swirl_bokeh(result, original, profile)
        
        if profile.bokeh_quality > 0.7:
            # Enhance smooth bokeh
            result = self._enhance_bokeh_smoothness(result, original, profile)
        
        if profile.rendering_style == 'vintage':
            # Enhance vintage rendering
            result = self._enhance_vintage_rendering(result, profile)
        elif profile.rendering_style == 'dreamy':
            # Enhance dreamy look
            result = self._enhance_dreamy_rendering(result, profile)
        
        # Add artistic vignetting
        if profile.vignette_smoothness < 0.5:
            result = self._add_artistic_vignetting(result, profile)
        
        return result
    
    def _preserve_bokeh_character(self, corrected: np.ndarray,
                                original: np.ndarray,
                                blur_mask: np.ndarray,
                                profile: CharacterProfile) -> np.ndarray:
        """Preserve bokeh characteristics in blurred regions"""
        
        result = corrected.copy()
        
        # Extract bokeh regions
        if np.any(blur_mask > 0.1):
            # Preserve swirl if present
            if profile.swirl_factor > 0.2:
                swirl_preserved = self._preserve_swirl_pattern(
                    corrected, original, blur_mask, profile.swirl_factor
                )
                result = result * (1 - blur_mask[:, :, np.newaxis] * 0.7) + \
                        swirl_preserved * blur_mask[:, :, np.newaxis] * 0.7
            
            # Preserve bokeh shape
            if profile.cat_eye_factor > 0.2:
                shape_preserved = self._preserve_bokeh_shape(
                    corrected, original, blur_mask, profile
                )
                result = np.where(
                    blur_mask[:, :, np.newaxis] > 0.5,
                    shape_preserved,
                    result
                )
        
        return result
    
    def _preserve_rendering_character(self, corrected: np.ndarray,
                                    original: np.ndarray,
                                    profile: CharacterProfile,
                                    strength: float) -> np.ndarray:
        """Preserve rendering characteristics"""
        
        result = corrected.copy()
        
        # Apply contrast signature
        if profile.contrast_signature is not None:
            result = self._apply_contrast_curve(
                result, profile.contrast_signature, strength * 0.7
            )
        
        # Apply color cast
        if profile.color_cast is not None:
            cast_strength = strength * 0.5
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - cast_strength) + \
                                 result[:, :, c] * profile.color_cast[c] * cast_strength
        
        # Adjust saturation
        if abs(profile.saturation_response - 1.0) > 0.1:
            result = self._adjust_saturation(
                result, profile.saturation_response, strength * 0.6
            )
        
        return np.clip(result, 0, 255)
    
    def _preserve_vignetting_character(self, corrected: np.ndarray,
                                     original: np.ndarray,
                                     profile: CharacterProfile,
                                     strength: float) -> np.ndarray:
        """Preserve vignetting characteristics"""
        
        h, w = corrected.shape[:2]
        
        # Extract original vignetting pattern
        original_vignetting = self._extract_vignetting_pattern(original)
        corrected_vignetting = self._extract_vignetting_pattern(corrected)
        
        # Blend vignetting patterns
        target_vignetting = corrected_vignetting * (1 - strength) + \
                          original_vignetting * strength
        
        # Apply smoothness adjustment
        if profile.vignette_smoothness < 0.5:
            # Make vignetting less smooth (more abrupt)
            target_vignetting = np.power(target_vignetting, 
                                       2.0 - profile.vignette_smoothness)
        
        # Apply asymmetry if present
        if abs(profile.vignette_asymmetry) > 0.1:
            target_vignetting = self._apply_vignetting_asymmetry(
                target_vignetting, profile.vignette_asymmetry
            )
        
        # Apply to image
        result = corrected * target_vignetting[:, :, np.newaxis]
        
        return result
    
    def _preserve_aberration_character(self, corrected: np.ndarray,
                                     original: np.ndarray,
                                     profile: CharacterProfile,
                                     strength: float) -> np.ndarray:
        """Preserve aberration characteristics"""
        
        result = corrected.copy()
        
        # Preserve spherical aberration (soft focus effect)
        if profile.spherical_aberration > 0.1:
            sphere_strength = strength * min(profile.spherical_aberration, 0.5)
            soft_focus = self._apply_spherical_aberration(
                corrected, sphere_strength
            )
            result = result * (1 - sphere_strength) + soft_focus * sphere_strength
        
        # Preserve field curvature
        if profile.field_curvature > 0.1:
            curve_strength = strength * min(profile.field_curvature, 0.3)
            curved = self._apply_field_curvature(corrected, curve_strength)
            result = result * (1 - curve_strength) + curved * curve_strength
        
        return result
    
    # Helper methods
    
    def _detect_blur_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect out-of-focus regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate local variance (sharpness indicator)
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, 
                           (kernel_size, kernel_size))
        
        # Normalize
        blur_map = 1.0 - (local_var / (np.max(local_var) + 1e-6))
        
        # Smooth the map
        blur_map = cv2.GaussianBlur(blur_map, (21, 21), 5.0)
        
        return blur_map
    
    def _extract_bokeh_regions(self, image: np.ndarray, 
                             blur_map: np.ndarray) -> List[np.ndarray]:
        """Extract regions with bokeh"""
        regions = []
        
        # Find significantly blurred areas
        blur_thresh = blur_map > 0.6
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            blur_thresh.astype(np.uint8), connectivity=8
        )
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 100:  # Minimum region size
                mask = (labels == i)
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                region = image[y:y+h, x:x+w]
                regions.append(region)
        
        return regions
    
    def _find_bokeh_highlights(self, region: np.ndarray) -> List[np.ndarray]:
        """Find bokeh highlights in a region"""
        highlights = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        
        # Find bright spots
        _, thresh = cv2.threshold(gray, np.percentile(gray, 90), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 2000:  # Reasonable highlight size
                x, y, w, h = cv2.boundingRect(contour)
                highlight = region[y:y+h, x:x+w]
                highlights.append(highlight)
        
        return highlights
    
    def _measure_bokeh_smoothness(self, highlights: List[np.ndarray]) -> float:
        """Measure bokeh smoothness from highlights"""
        if not highlights:
            return 0.5
        
        smoothness_scores = []
        
        for highlight in highlights:
            if highlight.size == 0:
                continue
            
            # Analyze edge softness
            gray = cv2.cvtColor(highlight, cv2.COLOR_RGB2GRAY) if len(highlight.shape) == 3 else highlight
            
            # Calculate gradient
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(gx**2 + gy**2)
            
            # Smooth bokeh has gentle gradients
            edge_sharpness = np.mean(gradient) / (np.max(gray) - np.min(gray) + 1e-6)
            smoothness = 1.0 - min(edge_sharpness, 1.0)
            smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.5
    
    def _detect_swirl_pattern(self, highlights: List[np.ndarray]) -> float:
        """Detect swirly bokeh pattern"""
        # Simplified detection - would implement directional analysis
        return 0.0
    
    def _detect_cat_eye_pattern(self, highlights: List[np.ndarray]) -> float:
        """Detect cat-eye bokeh at frame edges"""
        # Simplified detection - would analyze highlight ellipticity
        return 0.0
    
    def _detect_bubble_bokeh(self, highlights: List[np.ndarray]) -> bool:
        """Detect bubble/donut bokeh"""
        # Simplified detection - would analyze highlight intensity profiles
        return False
    
    def _measure_contrast_curve(self, image: np.ndarray) -> np.ndarray:
        """Measure contrast response curve"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Create tone curve by analyzing local contrast at different brightness levels
        num_bins = 10
        curve = np.zeros(num_bins)
        
        for i in range(num_bins):
            low = i * 256 // num_bins
            high = (i + 1) * 256 // num_bins
            
            mask = (gray >= low) & (gray < high)
            if np.sum(mask) > 100:
                # Local contrast in this brightness range
                local_contrast = self._calculate_local_contrast(gray, mask)
                curve[i] = local_contrast
        
        # Normalize
        if np.max(curve) > 0:
            curve = curve / np.max(curve)
        
        return curve
    
    def _measure_color_bias(self, image: np.ndarray) -> np.ndarray:
        """Measure color cast/bias"""
        if len(image.shape) != 3:
            return np.array([1.0, 1.0, 1.0])
        
        # Calculate average color balance
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        
        # Normalize to neutral gray
        gray_value = np.mean(mean_rgb)
        if gray_value > 0:
            color_bias = mean_rgb / gray_value
        else:
            color_bias = np.array([1.0, 1.0, 1.0])
        
        return color_bias
    
    def _measure_saturation_response(self, image: np.ndarray) -> float:
        """Measure saturation characteristics"""
        if len(image.shape) != 3:
            return 1.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Analyze saturation distribution
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Compare to expected neutral response
        mean_sat = np.mean(saturation)
        expected_sat = 0.5  # Neutral expectation
        
        response = mean_sat / expected_sat if expected_sat > 0 else 1.0
        
        return np.clip(response, 0.5, 2.0)
    
    def _measure_sharpness(self, region: np.ndarray) -> float:
        """Measure image sharpness"""
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        
        # Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        return sharpness
    
    def _create_preservation_mask(self, image: np.ndarray,
                                depth_map: Optional[np.ndarray],
                                profile: CharacterProfile) -> np.ndarray:
        """Create mask for selective preservation"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Base preservation in corners/edges (less critical areas)
        edge_width = min(h, w) // 8
        mask[:edge_width, :] = 0.5  # Top
        mask[-edge_width:, :] = 0.5  # Bottom
        mask[:, :edge_width] = 0.5  # Left
        mask[:, -edge_width:] = 0.5  # Right
        
        # Increase preservation in blurred areas
        blur_map = self._detect_blur_regions(image)
        mask = np.maximum(mask, blur_map * 0.8)
        
        # Use depth map if available
        if depth_map is not None:
            # More preservation in far areas
            mask = np.maximum(mask, depth_map * 0.7)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (31, 31), 10.0)
        
        return mask
    
    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, bool]:
        """Analyze image content type"""
        # Simplified content detection
        return {
            'is_portrait': False,  # Would use face detection
            'is_landscape': False,  # Would analyze scene
            'has_architecture': False  # Would detect lines/structures
        }
    
    def _create_zeiss_contrast_curve(self) -> np.ndarray:
        """Create Zeiss-style contrast curve"""
        # Characteristic gentle S-curve
        curve = np.array([0.1, 0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97, 1.0])
        return curve
    
    def _extract_vignetting_pattern(self, image: np.ndarray) -> np.ndarray:
        """Extract vignetting pattern from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Create radial distance map
        h, w = gray.shape
        center = (w // 2, h // 2)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = min(center)
        dist_norm = dist / max_dist
        
        # Extract brightness falloff
        vignetting = np.ones_like(gray, dtype=np.float32)
        
        # Sample brightness at different radii
        num_samples = 20
        for i in range(num_samples):
            r_min = i / num_samples
            r_max = (i + 1) / num_samples
            mask = (dist_norm >= r_min) & (dist_norm < r_max)
            
            if np.any(mask):
                avg_brightness = np.mean(gray[mask])
                center_brightness = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
                
                if center_brightness > 0:
                    falloff = avg_brightness / center_brightness
                    vignetting[mask] = falloff
        
        return vignetting
    
    def _apply_contrast_curve(self, image: np.ndarray, 
                            curve: np.ndarray,
                            strength: float) -> np.ndarray:
        """Apply contrast curve to image"""
        # Create lookup table from curve
        lut = np.linspace(0, 255, len(curve))
        full_lut = np.interp(np.arange(256), lut, curve * 255)
        
        # Apply with strength
        result = image.copy()
        for c in range(image.shape[2] if len(image.shape) == 3 else 1):
            channel = image[:, :, c] if len(image.shape) == 3 else image
            adjusted = np.interp(channel, np.arange(256), full_lut)
            
            if len(image.shape) == 3:
                result[:, :, c] = channel * (1 - strength) + adjusted * strength
            else:
                result = channel * (1 - strength) + adjusted * strength
        
        return result
    
    def _adjust_saturation(self, image: np.ndarray, 
                         saturation_factor: float,
                         strength: float) -> np.ndarray:
        """Adjust image saturation"""
        if len(image.shape) != 3:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust saturation channel
        target_sat = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - strength) + target_sat * strength
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return result.astype(image.dtype)
