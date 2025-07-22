# src/vintageoptics/integrations/lensfun.py (continued)
            # Parse parameters
            for param in ['a', 'b', 'c', 'k1', 'k2', 'k3']:
                value = distortion_elem.get(param)
                if value is not None:
                    calib['distortion']['parameters'].append(float(value))
        
        # TCA (Chromatic Aberration)
        tca_elem = calib_elem.find('tca')
        if tca_elem is not None:
            calib['tca'] = {
                'model': tca_elem.get('model', 'linear'),
                'focal': float(tca_elem.get('focal', 0)),
                'parameters': []
            }
            
            # Parse red/blue parameters
            for param in ['kr', 'kb', 'vr', 'vb']:
                value = tca_elem.get(param)
                if value is not None:
                    calib['tca']['parameters'].append(float(value))
        
        # Vignetting
        vignetting_elem = calib_elem.find('vignetting')
        if vignetting_elem is not None:
            calib['vignetting'] = {
                'model': vignetting_elem.get('model', 'pa'),
                'focal': float(vignetting_elem.get('focal', 0)),
                'aperture': float(vignetting_elem.get('aperture', 0)),
                'distance': float(vignetting_elem.get('distance', 10.0)),
                'parameters': []
            }
            
            # Parse parameters
            for param in ['k1', 'k2', 'k3']:
                value = vignetting_elem.get(param)
                if value is not None:
                    calib['vignetting']['parameters'].append(float(value))
        
        return calib if calib else None


# Fallback implementation if lensfunpy is not available
class LensfunFallback:
    """Fallback implementation using XML parsing when lensfunpy is not available"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lenses = []
        self._load_database()
    
    def _load_database(self):
        """Load lens database from XML files"""
        if not os.path.exists(self.db_path):
            logger.error(f"Lensfun database path not found: {self.db_path}")
            return
        
        # Find all XML files
        xml_files = []
        for root, dirs, files in os.walk(self.db_path):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
        
        # Parse each file
        for xml_file in xml_files:
            lenses = LensfunXMLParser.parse_lens_xml(xml_file)
            self.lenses.extend(lenses)
        
        logger.info(f"Loaded {len(self.lenses)} lenses from XML files")
    
    def find_lens(self, maker: str, model: str) -> Optional[Dict[str, Any]]:
        """Find lens in database"""
        maker_lower = maker.lower()
        model_lower = model.lower()
        
        for lens in self.lenses:
            if (maker_lower in lens['maker'].lower() and 
                model_lower in lens['model'].lower()):
                return lens
        
        return None
