"""
Pre-Flight System and Dependency Diagnostics for Linux GUI Operator.
Checks for required system packages, X11 display, screen capture tools, and API keys.
"""

import os
import shutil
import sys
from typing import Any, Dict, List, Tuple


def check_system_environment() -> Dict[str, Any]:
    """Inspects Linux X11 environment and required CLI tools."""
    diagnostics = {
        "display": False,
        "xdotool": False,
        "screen_capture": None,
        "tesseract": False,
        "xrandr": False,
        "warnings": [],
        "errors": [],
    }

    # 1. DISPLAY check
    display_var = os.environ.get("DISPLAY")
    if display_var:
        diagnostics["display"] = True
    else:
        diagnostics["errors"].append(
            "DISPLAY environment variable is not set. GUI automation requires an active X11 display."
        )

    # 2. Automation tools
    if shutil.which("xdotool"):
        diagnostics["xdotool"] = True
    else:
        diagnostics["warnings"].append(
            "xdotool is missing. Install with 'sudo apt install xdotool' for human-like event dispatch. "
            "Falling back to PyAutoGUI."
        )

    # 3. Screen capture
    if shutil.which("scrot"):
        diagnostics["screen_capture"] = "scrot"
    elif shutil.which("maim"):
        diagnostics["screen_capture"] = "maim"
    elif shutil.which("import"):
        diagnostics["screen_capture"] = "imagemagick"
    else:
        diagnostics["warnings"].append(
            "Neither scrot, maim, nor imagemagick is installed. Install scrot ('sudo apt install scrot') for "
            "zero-latency /dev/shm RAM capture. Falling back to PIL ImageGrab."
        )

    # 4. OCR
    if shutil.which("tesseract"):
        diagnostics["tesseract"] = True
    else:
        diagnostics["warnings"].append(
            "tesseract is not installed. Install with 'sudo apt install tesseract-ocr' for "
            "precise element snapping and checkbox centering."
        )

    # 5. Multi-monitor inspection
    if shutil.which("xrandr"):
        diagnostics["xrandr"] = True
    else:
        diagnostics["warnings"].append(
            "xrandr is not found. Multi-monitor auto-geometry will use default display resolution."
        )

    return diagnostics


def print_preflight_report(verbose: bool = True) -> bool:
    """Prints a friendly pre-flight diagnostics banner. Returns True if runnable."""
    diag = check_system_environment()

    if not verbose and not diag["errors"] and not diag["warnings"]:
        return True

    print("\033[1;36m[Pre-Flight Diagnostic]\033[0m Checking system environment:")
    print(f"  • X11 DISPLAY:      {'✓ ' + os.environ.get('DISPLAY', '') if diag['display'] else '✗ Missing'}")
    print(f"  • Screen Capture:   {'✓ ' + diag['screen_capture'] if diag['screen_capture'] else '⚠ PIL Fallback'}")
    print(f"  • Input Control:    {'✓ xdotool' if diag['xdotool'] else '⚠ PyAutoGUI only'}")
    print(f"  • OCR Snapping:     {'✓ tesseract' if diag['tesseract'] else '⚠ OCR disabled'}")
    print(f"  • Multi-Monitor:    {'✓ xrandr' if diag['xrandr'] else '⚠ Default fallback'}")

    if diag["warnings"]:
        print("\n\033[1;33m[Recommendations for Optimal Performance]:\033[0m")
        for w in diag["warnings"]:
            print(f"  • {w}")

    if diag["errors"]:
        print("\n\033[1;31m[Critical Setup Issues]:\033[0m")
        for e in diag["errors"]:
            print(f"  • {e}")
        return False

    print("")
    return True


if __name__ == "__main__":
    is_ok = print_preflight_report(verbose=True)
    sys.exit(0 if is_ok else 1)
