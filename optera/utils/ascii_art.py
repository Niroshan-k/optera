"""
Optera High-Impact ASCII Banner & Logo Generator
Compatible with Windows CP1252, UTF-8, Linux, and macOS terminals.
"""

import sys
from pathlib import Path
from typing import Optional, Union

# Global flag to ensure ASCII logo prints only ONCE per Python execution session
_BANNER_PRINTED = False

ASCII_LOGO_BANNER = r"""
================================================================================
  ___  ____ _____ _____ ____    _    
 / _ \|  _ \_   _| ____|  _ \  / \   
| | | | |_) || | |  _| | |_) |/ _ \  
| |_| |  __/ | | | |___|  _ </ ___ \ 
 \___/|_|    |_| |_____|_| \_\_/  \_\
  Supply Chain Optimization & Stochastic Risk Simulation Engine v1.0.2
  Institutional Swarm Intelligence (IPSO) & Event-Driven Monte Carlo
================================================================================
"""
ASCII_BANNER = ASCII_LOGO_BANNER


def convert_image_to_ascii(image_path: Union[str, Path], width: int = 60) -> Optional[str]:
    """
    Optional helper: Converts an image file (e.g. logo.png) into an ASCII string
    if PIL (Pillow) is installed.
    """
    try:
        from PIL import Image
        img_p = Path(image_path).resolve()
        if not img_p.exists():
            return None

        img = Image.open(img_p).convert("L")
        aspect_ratio = img.height / img.width
        height = int(width * aspect_ratio * 0.45)
        img = img.resize((width, height))

        pixels = img.getdata()
        chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
        new_pixels = [chars[pixel // 28] for pixel in pixels]
        
        ascii_img = []
        for i in range(0, len(new_pixels), width):
            ascii_img.append("".join(new_pixels[i:i + width]))
        return "\n".join(ascii_img)
    except Exception:
        return None


def print_ascii_banner(version: str = "1.0.2", force: bool = False) -> None:
    """
    Prints the official Optera ASCII logo banner safely to terminal.
    Ensures the banner prints ONCE when Optera starts.
    """
    global _BANNER_PRINTED
    if _BANNER_PRINTED and not force:
        return

    banner = ASCII_LOGO_BANNER.replace("v1.0.2", f"v{version}")
    try:
        print(banner)
        _BANNER_PRINTED = True
    except UnicodeEncodeError:
        safe_banner = banner.encode("ascii", "replace").decode("ascii")
        print(safe_banner)
        _BANNER_PRINTED = True
