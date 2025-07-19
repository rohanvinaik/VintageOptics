#!/usr/bin/env python3
"""
Debug metadata extraction for the uploaded image
"""

from PIL import Image
from PIL.ExifTags import TAGS
import sys

def debug_metadata(image_path):
    """Extract and display all EXIF metadata"""
    try:
        img = Image.open(image_path)
        exifdata = img.getexif()
        
        print("=== METADATA DEBUG ===")
        print(f"Image format: {img.format}")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print("\n=== EXIF DATA ===")
        
        if exifdata:
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                print(f"{tag}: {value}")
        else:
            print("No EXIF data found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_metadata(sys.argv[1])
    else:
        print("Usage: python debug_metadata.py <image_path>")
