# src/vintageoptics/depth/bokeh_analyzer.py

class BokehAnalyzer:
    """Analyze and characterize bokeh rendering"""
    
    def analyze_bokeh_characteristics(self, image: np.ndarray, 
                                    blur_map: np.ndarray) -> Dict:
        """Analyze bokeh rendering characteristics"""
        
        # Identify bokeh regions
        bokeh_mask = blur_map > 0.6  # Significantly blurred areas
        
        if not np.any(bokeh_mask):
            return {'has_bokeh': False}
        
        # Extract bokeh highlights
        highlights = self._extract_bokeh_highlights(image, bokeh_mask)
        
        characteristics = {
            'has_bokeh': True,
            'bokeh_shape': self._analyze_bokeh_shape(highlights),
            'swirl_amount': self._detect_swirl(image, bokeh_mask),
            'smoothness': self._analyze_smoothness(image, bokeh_mask),
            'outlining': self._detect_outlining(highlights),
            'cat_eye': self._detect_cat_eye(highlights),
            'onion_rings': self._detect_onion_rings(highlights),
            'quality_score': 0.0  # Will be calculated
        }
        
        # Calculate overall bokeh quality
        characteristics['quality_score'] = self._calculate_bokeh_quality(characteristics)
        
        # Detect specific vintage bokeh types
        characteristics.update(self._classify_bokeh_type(characteristics))
        
        return characteristics
    
    def _extract_bokeh_highlights(self, image: np.ndarray, 
                                 bokeh_mask: np.ndarray) -> List[BokehHighlight]:
        """Extract individual bokeh balls/highlights"""
        
        # Convert to LAB for better highlight extraction
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Find bright spots in bokeh regions
        bokeh_region = l_channel * bokeh_mask
        
        # Adaptive thresholding for highlights
        thresh = np.percentile(bokeh_region[bokeh_mask], 90)
        highlight_mask = (bokeh_region > thresh).astype(np.uint8)
        
        # Find individual highlights
        contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        highlights = []
        for contour in contours:
            if cv2.contourArea(contour) > 20:  # Minimum size
                highlight = BokehHighlight(
                    contour=contour,
                    center=self._get_contour_center(contour),
                    shape_descriptor=self._analyze_highlight_shape(contour),
                    intensity_profile=self._extract_intensity_profile(l_channel, contour)
                )
                highlights.append(highlight)
        
        return highlights
    
    def _detect_swirl(self, image: np.ndarray, bokeh_mask: np.ndarray) -> float:
        """Detect swirly bokeh characteristics"""
        
        # Analyze optical flow in bokeh regions
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create synthetic rotation to test swirl response
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        # Small rotation
        M = cv2.getRotationMatrix2D(center, 2, 1)
        rotated = cv2.warpAffine(gray, M, (w, h))
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray, rotated, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Analyze flow patterns in bokeh regions
        bokeh_flow = flow[bokeh_mask]
        
        # Calculate rotational component
        if len(bokeh_flow) > 0:
            # Compute average angular deviation from expected
            flow_magnitude = np.sqrt(bokeh_flow[:, 0]**2 + bokeh_flow[:, 1]**2)
            flow_angle = np.arctan2(bokeh_flow[:, 1], bokeh_flow[:, 0])
            
            # Swirly bokeh shows non-uniform rotational patterns
            angle_variance = np.var(flow_angle)
            swirl_score = min(angle_variance / np.pi, 1.0)
        else:
            swirl_score = 0.0
        
        return swirl_score