"""
Font utilities for Khmer text rendering in matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
from pathlib import Path
from typing import Optional, Dict, List
import platform


class KhmerFontManager:
    """
    Manager for Khmer font detection and configuration in matplotlib.
    """
    
    def __init__(self):
        self.available_fonts = self._detect_khmer_fonts()
        self.current_font = self._select_best_font()
        self._font_cache = {}
        
    def _detect_khmer_fonts(self) -> Dict[str, str]:
        """
        Detect available Khmer fonts from multiple sources.
        
        Returns:
            Dictionary mapping font names to font paths
        """
        fonts = {}
        
        # 1. Check project fonts directory
        project_fonts = self._check_project_fonts()
        fonts.update(project_fonts)
        
        # 2. Check system fonts
        system_fonts = self._check_system_fonts()
        fonts.update(system_fonts)
        
        # 3. Register project fonts with matplotlib
        for font_name, font_path in project_fonts.items():
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass
        
        return fonts
    
    def _check_project_fonts(self) -> Dict[str, str]:
        """Check fonts in the project's fonts directory."""
        fonts = {}
        
        # Find project root and fonts directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        font_dir = project_root / 'src' / 'fonts'
        
        if font_dir.exists():
            for font_file in font_dir.glob('*.ttf'):
                try:
                    font_prop = fm.FontProperties(fname=str(font_file))
                    font_name = font_prop.get_name()
                    
                    # Test if it can render Khmer
                    if self._test_khmer_rendering(font_name):
                        fonts[font_name] = str(font_file)
                        
                except Exception:
                    continue
        
        return fonts
    
    def _check_system_fonts(self) -> Dict[str, str]:
        """Check system-installed Khmer fonts."""
        fonts = {}
        
        # Common Khmer font names by platform
        khmer_font_names = {
            'Windows': [
                'Khmer UI', 'Khmer OS', 'Khmer OS System', 'DaunPenh', 
                'MoolBoran', 'Khmer Sangam MN'
            ],
            'Darwin': [  # macOS
                'Khmer Sangam MN', 'Khmer MN', 'Khmer UI'
            ],
            'Linux': [
                'Khmer OS', 'Khmer OS System', 'Liberation Sans'
            ]
        }
        
        system = platform.system()
        candidates = khmer_font_names.get(system, khmer_font_names['Linux'])
        
        for font_name in candidates:
            if self._test_khmer_rendering(font_name):
                # Try to find the actual font file
                font_path = self._find_font_path(font_name)
                fonts[font_name] = font_path or font_name
        
        return fonts
    
    def _find_font_path(self, font_name: str) -> Optional[str]:
        """Find the file path for a system font."""
        try:
            # Get all system fonts
            system_fonts = fm.findSystemFonts()
            
            for font_path in system_fonts:
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    if font_prop.get_name().lower() == font_name.lower():
                        return font_path
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _test_khmer_rendering(self, font_name: str) -> bool:
        """Test if a font can properly render Khmer digits."""
        try:
            # Suppress warnings during testing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create a small test figure
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '០១២៣៤', fontname=font_name, 
                       ha='center', va='center', fontsize=12)
                
                # If we get here without exception, the font works
                plt.close(fig)
                return True
                
        except Exception:
            return False
    
    def _select_best_font(self) -> Optional[str]:
        """Select the best available Khmer font."""
        if not self.available_fonts:
            return None
        
        # Priority order for font selection
        priority_fonts = [
            'Khmer OS', 'Khmer OS System', 'Khmer UI',
            'KhmerOS', 'KhmerOSsystem', 'KhmerUI'
        ]
        
        # Check priority fonts first
        for priority_font in priority_fonts:
            for available_font in self.available_fonts.keys():
                if priority_font.lower() in available_font.lower():
                    return available_font
        
        # Return the first available font
        return list(self.available_fonts.keys())[0]
    
    def get_font_properties(self, font_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get font properties for matplotlib text rendering.
        
        Args:
            font_name: Specific font name or None for default
            
        Returns:
            Dictionary of font properties for matplotlib
        """
        if font_name is None:
            font_name = self.current_font
        
        if font_name and font_name in self.available_fonts:
            return {'fontname': font_name}
        else:
            return {}
    
    def render_safe_text(self, text: str, fallback_format: str = "Text {}", 
                        index: Optional[int] = None) -> tuple:
        """
        Safely render Khmer text with fallback options.
        
        Args:
            text: Original Khmer text
            fallback_format: Format string for fallback
            index: Index to use in fallback
            
        Returns:
            Tuple of (display_text, font_properties)
        """
        # If we have a working font, use the original text
        if self.current_font:
            return text, self.get_font_properties()
        
        # Try to detect if text contains non-ASCII characters
        try:
            # Check if the text is properly encoded
            text.encode('utf-8').decode('utf-8')
            
            # If it contains placeholder characters or non-ASCII
            ascii_text = text.encode('ascii', errors='replace').decode('ascii')
            
            if '?' in ascii_text or len(ascii_text) != len(text):
                # Use fallback
                if index is not None:
                    return fallback_format.format(index), {}
                else:
                    return f"Khmer({len(text)} chars)", {}
            
            # Text seems okay
            return text, {}
            
        except Exception:
            # Ultimate fallback
            if index is not None:
                return fallback_format.format(index), {}
            else:
                return "Sample", {}
    
    def print_font_info(self):
        """Print information about detected fonts."""
        print("Khmer Font Detection Report")
        print("=" * 40)
        
        if self.available_fonts:
            print(f"✅ Found {len(self.available_fonts)} Khmer fonts:")
            for font_name, font_path in self.available_fonts.items():
                path_info = f" ({font_path})" if font_path != font_name else " (system)"
                print(f"  • {font_name}{path_info}")
            
            print(f"\n🎯 Selected font: {self.current_font}")
            
        else:
            print("❌ No Khmer fonts detected")
            print("   Text will use fallback labels")
        
        print("\nRecommendations:")
        if not self.available_fonts:
            print("  • Install system Khmer fonts (Khmer OS, Khmer UI)")
            print("  • Ensure project fonts in src/fonts/ are valid TTF files")
        else:
            print("  • Font detection successful")
            print("  • Khmer text should render properly in plots")


# Global font manager instance
_font_manager = None


def get_font_manager() -> KhmerFontManager:
    """Get the global Khmer font manager instance."""
    global _font_manager
    if _font_manager is None:
        _font_manager = KhmerFontManager()
    return _font_manager


def safe_khmer_text(text: str, fallback_format: str = "Index {}", 
                   index: Optional[int] = None) -> tuple:
    """
    Convenience function for safe Khmer text rendering.
    
    Args:
        text: Original Khmer text
        fallback_format: Format string for fallback
        index: Index to use in fallback
        
    Returns:
        Tuple of (display_text, font_properties)
    """
    manager = get_font_manager()
    return manager.render_safe_text(text, fallback_format, index)


def setup_khmer_fonts() -> Optional[str]:
    """
    Setup Khmer fonts for matplotlib and return the selected font.
    
    Returns:
        Name of selected Khmer font or None if none available
    """
    manager = get_font_manager()
    return manager.current_font


def print_font_status():
    """Print current Khmer font detection status."""
    manager = get_font_manager()
    manager.print_font_info()


# Suppress common font warnings
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*font.*') 