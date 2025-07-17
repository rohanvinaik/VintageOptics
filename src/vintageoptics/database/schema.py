-- Complete database schema for VintageOptics v2.0

-- Core lens data
CREATE TABLE lenses (
    lens_id TEXT PRIMARY KEY,
    manufacturer TEXT NOT NULL,
    model TEXT NOT NULL,
    variant TEXT,
    lens_type TEXT CHECK(lens_type IN ('manual', 'electronic', 'adapted')),
    mount_type TEXT NOT NULL,
    
    -- Optical specifications
    focal_min_mm REAL NOT NULL,
    focal_max_mm REAL NOT NULL,
    aperture_min REAL NOT NULL,
    aperture_max REAL NOT NULL,
    
    -- Physical characteristics
    elements INTEGER,
    groups INTEGER,
    aperture_blades INTEGER,
    special_elements JSON,  -- ED, aspherical, etc.
    coatings JSON,
    
    -- Metadata
    production_years TEXT,
    serial_formats TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Electronic lens communication data
CREATE TABLE electronic_lens_data (
    lens_id TEXT PRIMARY KEY,
    protocol TEXT NOT NULL,  -- 'Canon EF', 'Nikon F', etc.
    electronic_id INTEGER,
    cpu_contacts INTEGER,
    data_pins JSON,
    firmware_versions JSON,
    communication_features JSON,
    motor_groups JSON,
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id)
);

-- 3D optical correction parameters
CREATE TABLE optical_parameters (
    param_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_id TEXT NOT NULL,
    
    -- Shooting parameters (3D lookup)
    focal_length_mm REAL NOT NULL,
    aperture_f REAL NOT NULL,
    focus_distance_m REAL NOT NULL,
    
    -- Sensor parameters
    sensor_size TEXT DEFAULT 'FF',
    sensor_shift_x REAL DEFAULT 0,
    sensor_shift_y REAL DEFAULT 0,
    
    -- Distortion parameters
    distortion_model TEXT DEFAULT 'brown_conrady',
    k1 REAL DEFAULT 0, k2 REAL DEFAULT 0, k3 REAL DEFAULT 0,
    p1 REAL DEFAULT 0, p2 REAL DEFAULT 0,
    
    -- Chromatic aberration
    ca_model TEXT DEFAULT 'linear',
    ca_red_scale REAL DEFAULT 1.0,
    ca_blue_scale REAL DEFAULT 1.0,
    ca_red_shift JSON,
    ca_blue_shift JSON,
    
    -- Vignetting
    vignetting_model TEXT DEFAULT 'polynomial',
    vignetting_params JSON,
    
    -- Advanced aberrations
    spherical_aberration REAL DEFAULT 0,
    coma JSON,
    astigmatism JSON,
    field_curvature REAL DEFAULT 0,
    
    -- Metadata
    measurement_method TEXT,
    confidence REAL DEFAULT 0.5,
    sample_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id),
    UNIQUE(lens_id, focal_length_mm, aperture_f, focus_distance_m, sensor_size)
);

-- Lens instances (individual copies)
CREATE TABLE lens_instances (
    instance_id TEXT PRIMARY KEY,
    lens_id TEXT NOT NULL,
    serial_number TEXT,
    
    -- Identification
    optical_fingerprint JSON,
    electronic_fingerprint JSON,
    
    -- Characteristics
    measured_variations JSON,  -- Deviations from reference
    defect_map JSON,          -- Known defects
    bokeh_character JSON,     -- Measured bokeh characteristics
    rendering_style JSON,     -- Color, contrast, etc.
    
    -- History
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    image_count INTEGER DEFAULT 0,
    
    -- User data
    owner_id TEXT,
    nickname TEXT,
    notes TEXT,
    favorite BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id)
);

-- Bokeh characteristics
CREATE TABLE bokeh_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_id TEXT NOT NULL,
    aperture_f REAL NOT NULL,
    
    -- Shape characteristics
    aperture_shape JSON,      -- Polygon vertices
    shape_uniformity REAL,
    
    -- Rendering characteristics
    smoothness REAL,
    swirl_factor REAL,
    cat_eye_factor REAL,
    onion_rings BOOLEAN,
    soap_bubbles BOOLEAN,
    
    -- Quality metrics
    highlight_quality REAL,
    transition_quality REAL,
    overall_score REAL,
    
    -- Sample data
    sample_images JSON,
    analysis_method TEXT,
    
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id)
);

-- Synthesis profiles
CREATE TABLE synthesis_profiles (
    profile_id TEXT PRIMARY KEY,
    lens_id TEXT NOT NULL,
    profile_name TEXT NOT NULL,
    
    -- Synthesis parameters
    distortion_strength REAL DEFAULT 1.0,
    vignetting_strength REAL DEFAULT 1.0,
    ca_strength REAL DEFAULT 1.0,
    bokeh_synthesis JSON,
    
    -- Character parameters
    contrast_curve JSON,
    color_response JSON,
    coating_simulation JSON,
    
    -- Artistic adjustments
    artistic_params JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id)
);

-- Calibration data
CREATE TABLE calibrations (
    calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_id TEXT NOT NULL,
    instance_id TEXT,
    
    -- Method and quality
    method TEXT NOT NULL,
    quality_score REAL,
    confidence REAL,
    
    -- Results
    parameters JSON,
    raw_data BLOB,
    
    -- Metadata
    image_count INTEGER,
    chart_type TEXT,
    environmental_conditions JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lens_id) REFERENCES lenses(lens_id)
);

-- Processing cache
CREATE TABLE processing_cache (
    cache_id TEXT PRIMARY KEY,
    image_hash TEXT NOT NULL,
    settings_hash TEXT NOT NULL,
    
    -- Cache data
    result_type TEXT,
    result_data BLOB,
    metadata JSON,
    
    -- Usage tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    size_bytes INTEGER,
    
    INDEX idx_lookup (image_hash, settings_hash)
);

-- User presets
CREATE TABLE user_presets (
    preset_id TEXT PRIMARY KEY,
    preset_name TEXT NOT NULL,
    category TEXT,
    
    -- Configuration
    base_lens_id TEXT,
    processing_mode TEXT,
    parameters JSON,
    
    -- Metadata
    description TEXT,
    tags JSON,
    is_public BOOLEAN DEFAULT FALSE,
    
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (base_lens_id) REFERENCES lenses(lens_id)
);

-- Create indexes
CREATE INDEX idx_optical_params_lookup 
    ON optical_parameters(lens_id, focal_length_mm, aperture_f);
CREATE INDEX idx_instance_serial 
    ON lens_instances(serial_number);
CREATE INDEX idx_cache_access 
    ON processing_cache(accessed_at);