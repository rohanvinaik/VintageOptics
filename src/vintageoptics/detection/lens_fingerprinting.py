# src/vintageoptics/detection/lens_fingerprinting.py (continued)
            
            for i in range(1, num_labels):
                # Get defect info
                mask_single = (labels == i).astype(np.uint8) * 255
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = centroids[i]
                
                # Determine defect type from accumulated detections
                defect_types = {}
                for defect_list in all_defects:
                    for defect in defect_list:
                        overlap = cv2.bitwise_and(defect.mask, mask_single)
                        if np.sum(overlap) > area * 0.5:  # >50% overlap
                            defect_type = defect.defect_type.value
                            defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
                
                # Most common type
                if defect_types:
                    primary_type = max(defect_types, key=defect_types.get)
                else:
                    primary_type = 'unknown'
                
                permanent_defects.append({
                    'type': primary_type,
                    'location': (int(cx), int(cy)),
                    'size': int(area),
                    'persistence': float(defect_accumulator[int(cy), int(cx)]),
                    'mask': mask_single.tolist()  # Store as list for JSON serialization
                })
        
        return permanent_defects
    
    def _assess_coating_condition(self, images: List[np.ndarray]) -> float:
        """Assess overall coating condition (0=poor, 1=excellent)"""
        
        condition_scores = []
        
        for img in images:
            if len(img.shape) != 3:
                continue
            
            # Analyze color consistency
            lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Coating wear shows as color variance
            color_variance = np.var(a) + np.var(b)
            
            # Analyze reflection consistency  
            # Good coatings have uniform reflections
            reflection_map = self._detect_reflections(img)
            reflection_uniformity = 1.0 - np.std(reflection_map)
            
            # Combine metrics
            score = (reflection_uniformity * 0.7 + 
                    (1.0 - min(color_variance / 1000, 1.0)) * 0.3)
            condition_scores.append(score)
        
        return np.mean(condition_scores) if condition_scores else 0.5
    
    def _measure_optical_clarity(self, images: List[np.ndarray]) -> float:
        """Measure overall optical clarity (0=hazy, 1=clear)"""
        
        clarity_scores = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            # Measure local contrast
            kernel_size = 15
            local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
            local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, 
                               (kernel_size, kernel_size))
            contrast_map = np.sqrt(local_var) / (local_mean + 1e-6)
            
            # High clarity = high average contrast
            avg_contrast = np.mean(contrast_map)
            
            # Measure edge sharpness
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1] * 255)
            
            # Combine metrics
            clarity = avg_contrast * 0.6 + edge_density * 40 * 0.4
            clarity_scores.append(min(clarity, 1.0))
        
        return np.mean(clarity_scores) if clarity_scores else 0.5
    
    def _analyze_color_response(self, images: List[np.ndarray]) -> np.ndarray:
        """Analyze color response curves"""
        
        # Build histogram for each channel
        response_curves = np.zeros((3, 256))
        
        for img in images:
            if len(img.shape) != 3:
                continue
            
            for c in range(3):
                hist, _ = np.histogram(img[:, :, c], bins=256, range=(0, 256))
                response_curves[c] += hist
        
        # Normalize
        response_curves /= (np.sum(response_curves, axis=1, keepdims=True) + 1e-6)
        
        # Smooth curves
        from scipy.ndimage import gaussian_filter1d
        for c in range(3):
            response_curves[c] = gaussian_filter1d(response_curves[c], sigma=2)
        
        return response_curves
    
    def _analyze_contrast_curve(self, images: List[np.ndarray]) -> np.ndarray:
        """Analyze contrast transfer function"""
        
        # Sample contrast at different intensity levels
        num_bins = 20
        contrast_curve = np.zeros(num_bins)
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            # Divide intensity range into bins
            for i in range(num_bins):
                low = i * 256 // num_bins
                high = (i + 1) * 256 // num_bins
                
                # Find pixels in this intensity range
                mask = (gray >= low) & (gray < high)
                
                if np.sum(mask) > 100:  # Enough pixels
                    # Measure local contrast for these pixels
                    kernel = np.ones((5, 5)) / 25
                    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                    local_contrast = np.abs(gray - local_mean)
                    
                    avg_contrast = np.mean(local_contrast[mask])
                    contrast_curve[i] += avg_contrast
        
        contrast_curve /= len(images)
        
        # Normalize
        contrast_curve = contrast_curve / (np.max(contrast_curve) + 1e-6)
        
        return contrast_curve
    
    def _analyze_flare_pattern(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Analyze lens flare characteristics"""
        
        flare_patterns = []
        
        for img in images:
            # Detect bright light sources
            bright_spots = self._detect_bright_sources(img)
            
            if bright_spots:
                # Analyze flare around bright spots
                flare_map = self._extract_flare_pattern(img, bright_spots)
                flare_patterns.append(flare_map)
        
        if flare_patterns:
            # Average flare patterns
            avg_pattern = np.mean(flare_patterns, axis=0)
            return avg_pattern
        
        return None
    
    def _calculate_confidence_score(self, num_images: int, has_calibration: bool) -> float:
        """Calculate fingerprint confidence score"""
        
        # Base score from number of images
        image_score = min(num_images / 30, 1.0)  # Max out at 30 images
        
        # Calibration bonus
        calibration_bonus = 0.2 if has_calibration else 0.0
        
        # Combine scores
        confidence = image_score * 0.8 + calibration_bonus
        
        return min(confidence, 1.0)
    
    # Helper methods
    
    def _estimate_distortion_from_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Estimate distortion coefficients without calibration"""
        # Simplified estimation - in practice would use feature matching
        return np.array([0.1, -0.05, 0.0, 0.0, 0.0])
    
    def _measure_local_distortion(self, img: np.ndarray, center: Tuple[int, int], 
                                 radius: float) -> float:
        """Measure distortion at specific radius"""
        # Simplified - would use line detection in practice
        return radius / 1000.0
    
    def _measure_distortion_asymmetry(self, images: List[np.ndarray]) -> np.ndarray:
        """Measure asymmetry in distortion pattern"""
        return np.random.rand(4) * 0.1  # Placeholder
    
    def _measure_edge_shift(self, edges1: np.ndarray, edges2: np.ndarray) -> Tuple[float, float]:
        """Measure shift between edge maps"""
        if edges1.shape != edges2.shape:
            return (0.0, 0.0)
        
        # Use phase correlation for sub-pixel shift detection
        shift, error, diffphase = cv2.phaseCorrelate(
            edges1.astype(np.float32), 
            edges2.astype(np.float32)
        )
        
        return shift
    
    def _detect_bokeh_highlights(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect out-of-focus highlights for bokeh analysis"""
        highlights = []
        
        if len(img.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Find bright spots
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable highlight size
                # Extract highlight
                x, y, w, h = cv2.boundingRect(contour)
                highlight_img = img[y:y+h, x:x+w]
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x, -y))
                
                highlights.append((highlight_img, mask))
        
        return highlights
    
    def _analyze_highlight_shape(self, highlight: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Analyze individual highlight shape"""
        features = {
            'smoothness': 0.0,
            'circularity': 0.0,
            'eccentricity': 0.0
        }
        
        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return features
        
        contour = contours[0]
        
        # Smoothness - ratio of contour length to convex hull length
        hull = cv2.convexHull(contour)
        if cv2.arcLength(hull, True) > 0:
            features['smoothness'] = cv2.arcLength(contour, True) / cv2.arcLength(hull, True)
        
        # Circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            features['circularity'] = 4 * np.pi * area / (perimeter * perimeter)
        
        # Eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            axes = ellipse[1]
            if max(axes) > 0:
                features['eccentricity'] = min(axes) / max(axes)
        
        return features
    
    def _detect_swirl_pattern(self, shapes: List[Dict[str, float]]) -> float:
        """Detect swirly bokeh pattern"""
        # Placeholder - would analyze directional bias in highlights
        return 0.0
    
    def _detect_cat_eye_pattern(self, shapes: List[Dict[str, float]]) -> float:
        """Detect cat-eye bokeh pattern"""
        # Check for elliptical highlights at frame edges
        eccentricities = [s['eccentricity'] for s in shapes]
        return 1.0 - np.mean(eccentricities) if eccentricities else 0.0
    
    def _detect_bubble_bokeh(self, shapes: List[Dict[str, float]]) -> float:
        """Detect bubble/donut bokeh"""
        # Would analyze highlight intensity profiles
        return 0.0
    
    def _measure_shape_consistency(self, shapes: List[Dict[str, float]]) -> float:
        """Measure consistency of highlight shapes"""
        if len(shapes) < 2:
            return 1.0
        
        # Calculate variance in shape metrics
        circularities = [s['circularity'] for s in shapes]
        consistency = 1.0 - np.std(circularities)
        
        return max(0.0, consistency)
    
    def _estimate_center_from_symmetry(self, images: List[np.ndarray]) -> Tuple[float, float]:
        """Estimate optical center from symmetry analysis"""
        # Placeholder - would use iterative symmetry maximization
        h, w = images[0].shape[:2]
        return (0.0, 0.0)  # Assume no offset for now
    
    def _compare_regions(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """Compare two image regions"""
        if region1.shape != region2.shape:
            region2 = cv2.resize(region2, (region1.shape[1], region1.shape[0]))
        
        return np.mean(np.abs(region1.astype(np.float32) - region2.astype(np.float32)))
    
    def _detect_reflections(self, img: np.ndarray) -> np.ndarray:
        """Detect reflective areas in image"""
        # Convert to HSV
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # High value and low saturation indicates reflections
        _, s, v = cv2.split(hsv)
        
        reflection_map = ((v > 240) & (s < 30)).astype(np.float32)
        
        # Smooth
        reflection_map = cv2.GaussianBlur(reflection_map, (5, 5), 1.0)
        
        return reflection_map
    
    def _detect_bright_sources(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Detect bright light sources in image"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Find very bright spots
        _, bright = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        # Find centers
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sources = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                sources.append((cx, cy))
        
        return sources
    
    def _extract_flare_pattern(self, img: np.ndarray, 
                             sources: List[Tuple[int, int]]) -> np.ndarray:
        """Extract flare pattern around bright sources"""
        h, w = img.shape[:2]
        flare_map = np.zeros((h, w), dtype=np.float32)
        
        for cx, cy in sources:
            # Create radial gradient from source
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Flare intensity decreases with distance
            source_flare = np.exp(-dist / 100)
            flare_map += source_flare
        
        # Normalize
        flare_map = flare_map / (np.max(flare_map) + 1e-6)
        
        return flare_map
    
    def _extract_matching_features(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features for fingerprint matching"""
        features = {}
        
        # Basic distortion estimate
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Sample radial distortion
        radial_samples = []
        for r in np.linspace(0.1, 1.0, 10):
            radius = r * min(center)
            sample = self._measure_local_distortion(img, center, radius)
            radial_samples.append(sample)
        
        features['distortion'] = np.array(radial_samples)
        
        # Color response
        if len(img.shape) == 3:
            features['color'] = np.array([
                np.mean(img[:, :, 0]),
                np.mean(img[:, :, 1]),
                np.mean(img[:, :, 2])
            ])
        
        # Vignetting estimate
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        corners = [
            gray[:50, :50],
            gray[:50, -50:],
            gray[-50:, :50],
            gray[-50:, -50:]
        ]
        features['vignetting'] = np.mean([np.mean(c) for c in corners])
        
        return features
    
    def _calculate_match_score(self, test_features: Dict[str, np.ndarray],
                             fingerprint: OpticalFingerprint) -> float:
        """Calculate match score between test features and fingerprint"""
        scores = []
        
        # Compare distortion
        if 'distortion' in test_features:
            dist_score = 1.0 - np.mean(np.abs(
                test_features['distortion'] - fingerprint.distortion_signature[:10]
            ))
            scores.append(dist_score)
        
        # Compare color response
        if 'color' in test_features and fingerprint.color_response.size > 0:
            color_score = 1.0 - np.mean(np.abs(
                test_features['color'] / 255.0 - 
                np.mean(fingerprint.color_response, axis=1)
            ))
            scores.append(color_score)
        
        # Compare vignetting
        if 'vignetting' in test_features:
            vig_score = 1.0 - abs(
                test_features['vignetting'] / 255.0 - 
                np.mean(fingerprint.vignetting_signature)
            )
            scores.append(vig_score)
        
        return np.mean(scores) if scores else 0.0
