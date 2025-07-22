# src/vintageoptics/synthesis/lens_library.py

class LensCharacteristicLibrary:
    """Comprehensive library of lens characteristics for synthesis"""
    
    def __init__(self, database_path: str):
        self.db_path = database_path
        self.profiles = self._load_all_profiles()
        self.presets = self._load_artistic_presets()
    
    def get_synthesis_profile(self, lens_name: str) -> LensSynthesisProfile:
        """Get complete synthesis profile for a lens"""
        
        base_profile = self.profiles.get(lens_name, {})
        
        return LensSynthesisProfile(
            optical_formula=base_profile.get('optical_formula'),
            distortion_model=self._get_distortion_model(base_profile),
            vignetting_model=self._get_vignetting_model(base_profile),
            chromatic_model=self._get_chromatic_model(base_profile),
            bokeh_model=self._get_bokeh_model(base_profile),
            rendering_character=self._get_rendering_character(base_profile),
            special_effects=self._get_special_effects(base_profile)
        )
    
    def create_artistic_preset(self, name: str, description: str,
                             base_lens: str, modifications: Dict) -> str:
        """Create custom artistic preset based on lens characteristics"""
        
        preset = {
            'name': name,
            'description': description,
            'base_lens': base_lens,
            'modifications': modifications,
            'created': datetime.now().isoformat()
        }
        
        preset_id = self._save_preset(preset)
        return preset_id