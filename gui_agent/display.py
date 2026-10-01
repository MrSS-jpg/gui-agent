import io
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image, ImageGrab


@dataclass
class DisplayGeometry:
    name: str
    is_primary: bool
    width: int
    height: int
    offset_x: int
    offset_y: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) global bounding box."""
        return (self.offset_x, self.offset_y, self.offset_x + self.width, self.offset_y + self.height)

    def contains_global(self, gx: int, gy: int) -> bool:
        """Checks if global coordinate (gx, gy) is inside this display's boundary."""
        return (self.offset_x <= gx < self.offset_x + self.width) and \
               (self.offset_y <= gy < self.offset_y + self.height)

    def __str__(self) -> str:
        tag = " [PRIMARY]" if self.is_primary else " [SECONDARY]"
        return f"{self.name}{tag}: {self.width}x{self.height}+{self.offset_x}+{self.offset_y}"


class DualMonitorLock:
    """
    Manages dual-monitor detection and strict geometry locking for antiX Linux / X11.
    Locks screen capture and OS input to the secondary monitor while strictly forbidding
    any interactions on the primary monitor.
    Provides visual scaling calibration, zero-drift coordinate translation,
    and lightning-fast RAM capture (/dev/shm) with 80% JPEG stream compression.
    """

    def __init__(self, target_secondary: Optional[str] = None):
        self.monitors = self._detect_monitors()
        self.primary, self.secondary = self._assign_displays(target_secondary)
        self.last_capture_size: Tuple[int, int] = (self.secondary.width, self.secondary.height)
        self.last_frame_bytes: Optional[bytes] = None

    def _detect_monitors(self) -> List[DisplayGeometry]:
        """Queries xrandr via subprocess to inspect connected screens and geometries."""
        try:
            output = subprocess.check_output(["xrandr"], text=True)
        except Exception:
            # Fallback if xrandr cannot be run
            return [
                DisplayGeometry("LVDS-1", True, 1366, 768, 0, 0),
                DisplayGeometry("VGA-1", False, 1920, 1080, 1366, 0),
            ]

        monitors: List[DisplayGeometry] = []
        pattern = re.compile(r"^(\S+)\s+connected\s+(?:(primary)\s+)?(\d+)x(\d+)\+(\d+)\+(\d+)")
        for line in output.splitlines():
            if " connected " in line:
                m = pattern.search(line)
                if m:
                    monitors.append(DisplayGeometry(
                        name=m.group(1),
                        is_primary=bool(m.group(2)),
                        width=int(m.group(3)),
                        height=int(m.group(4)),
                        offset_x=int(m.group(5)),
                        offset_y=int(m.group(6))
                    ))

        if not monitors:
            return [
                DisplayGeometry("LVDS-1", True, 1366, 768, 0, 0),
                DisplayGeometry("VGA-1", False, 1920, 1080, 1366, 0),
            ]

        return monitors

    def _assign_displays(self, target_secondary: Optional[str]) -> Tuple[DisplayGeometry, DisplayGeometry]:
        """Identifies primary display and secondary display with strict VGA-1 geometry enforcement."""
        primary = next((m for m in self.monitors if m.is_primary), self.monitors[0])

        if target_secondary:
            sec = next((m for m in self.monitors if m.name == target_secondary), None)
            if not sec:
                if target_secondary == "VGA-1":
                    sec = DisplayGeometry("VGA-1", False, 1920, 1080, 1366, 0)
                else:
                    available = ", ".join(m.name for m in self.monitors)
                    raise ValueError(f"Requested monitor '{target_secondary}' not found. Available: {available}")
            else:
                if sec.name == "VGA-1":
                    sec.width = 1920
                    sec.height = 1080
                    sec.offset_x = 1366
                    sec.offset_y = 0
            return primary, sec

        # Prioritize VGA-1 if present in connected monitors
        vga = next((m for m in self.monitors if m.name == "VGA-1"), None)
        if vga:
            vga.width = 1920
            vga.height = 1080
            vga.offset_x = 1366
            vga.offset_y = 0
            return primary, vga

        non_primaries = [m for m in self.monitors if not m.is_primary]
        if non_primaries:
            return primary, non_primaries[0]

        if len(self.monitors) >= 2:
            return self.monitors[0], self.monitors[1]

        return primary, DisplayGeometry("VGA-1", False, 1920, 1080, 1366, 0)

    def capture_secondary_frame(self) -> Tuple[Image.Image, bytes]:
        """
        Captures the screen cropped specifically to the bounding box of Monitor 2 (VGA-1).
        Secondary monitor VGA-1 geometry is strictly: Width=1920, Height=1080, X-offset=1366, Y-offset=0.
        Uses fast in-memory /dev/shm tmpfs RAM capture via scrot, maim, or PIL ImageGrab
        (zero physical disk I/O) and compresses outgoing payloads to 80% JPEG quality
        to reduce base64 upload latency across remote vision endpoints.
        """
        shm_path = "/dev/shm/agent_frame.jpg"
        sec = self.secondary

        # Secondary monitor VGA-1 geometry is strictly: Width=1920, Height=1080, X-offset=1366, Y-offset=0
        if sec.name == "VGA-1":
            offset_x, offset_y, width, height = 1366, 0, 1920, 1080
        else:
            offset_x, offset_y, width, height = sec.offset_x, sec.offset_y, sec.width, sec.height

        env = os.environ.copy()
        env["DISPLAY"] = os.environ.get("DISPLAY", ":0.0")
        if "XAUTHORITY" not in env or not env["XAUTHORITY"]:
            env["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")

        # 1. Primary Strategy: scrot with exact bounding box -a 1366,0,1920,1080 -q 80 /dev/shm/agent_frame.jpg
        # -o flag ensures existing file in /dev/shm is overwritten without creating numbered duplicates
        try:
            cmd = [
                "scrot",
                "-o",
                "-a", f"{offset_x},{offset_y},{width},{height}",
                "-q", "80",
                shm_path,
            ]
            res = subprocess.run(cmd, env=env, capture_output=True, timeout=1.0)
            if res.returncode == 0 and os.path.exists(shm_path) and os.path.getsize(shm_path) > 0:
                with open(shm_path, "rb") as f:
                    jpeg_bytes = f.read()
                img = Image.open(io.BytesIO(jpeg_bytes))
                # Guard against uncropped full-desktop captures
                if img.width > width or img.height > height:
                    img = img.crop((offset_x, offset_y, offset_x + width, offset_y + height))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80, optimize=True)
                    jpeg_bytes = buf.getvalue()
                    with open(shm_path, "wb") as f:
                        f.write(jpeg_bytes)
                self.last_capture_size = (img.width, img.height)
                self.last_frame_bytes = jpeg_bytes
                return img, jpeg_bytes
        except Exception:
            pass

        # 2. Strategy: maim with -g 1920x1080+1366+0
        try:
            cmd_maim = [
                "maim",
                "-g", f"{width}x{height}+{offset_x}+{offset_y}",
                shm_path,
            ]
            res = subprocess.run(cmd_maim, env=env, capture_output=True, timeout=1.0)
            if res.returncode == 0 and os.path.exists(shm_path) and os.path.getsize(shm_path) > 0:
                with open(shm_path, "rb") as f:
                    raw_bytes = f.read()
                img = Image.open(io.BytesIO(raw_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True)
                jpeg_bytes = buf.getvalue()
                with open(shm_path, "wb") as f:
                    f.write(jpeg_bytes)
                self.last_capture_size = (img.width, img.height)
                self.last_frame_bytes = jpeg_bytes
                return img, jpeg_bytes
        except Exception:
            pass

        # 3. Strategy: ImageMagick import with root window and crop (1920x1080+1366+0)
        try:
            cmd_import = [
                "import",
                "-window", "root",
                "-crop", f"{width}x{height}+{offset_x}+{offset_y}",
                shm_path,
            ]
            res = subprocess.run(cmd_import, env=env, capture_output=True, timeout=1.0)
            if res.returncode == 0 and os.path.exists(shm_path) and os.path.getsize(shm_path) > 0:
                with open(shm_path, "rb") as f:
                    raw_bytes = f.read()
                img = Image.open(io.BytesIO(raw_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True)
                jpeg_bytes = buf.getvalue()
                with open(shm_path, "wb") as f:
                    f.write(jpeg_bytes)
                self.last_capture_size = (img.width, img.height)
                self.last_frame_bytes = jpeg_bytes
                return img, jpeg_bytes
        except Exception:
            pass

        # 4. Strategy: PIL ImageGrab - specifically crop (1366, 0, 1366 + 1920, 1080)
        crop_bbox = (offset_x, offset_y, offset_x + width, offset_y + height)
        img = ImageGrab.grab(bbox=crop_bbox)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (width, height):
            img = img.crop(crop_bbox)

        self.last_capture_size = (img.width, img.height)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        jpeg_bytes = buffer.getvalue()
        self.last_frame_bytes = jpeg_bytes
        try:
            with open(shm_path, "wb") as f:
                f.write(jpeg_bytes)
        except Exception:
            pass
        return img, jpeg_bytes

    def wait_for_screen_stabilization(self, timeout: float = 1.0, poll_interval: float = 0.25) -> bool:
        """
        Adaptive polling helper: checks for visual stability or settles after UI actions.
        Returns True when consecutive frame captures match or timeout expires.
        """
        start_time = time.time()
        prev_bytes = self.last_frame_bytes
        time.sleep(poll_interval)

        while (time.time() - start_time) < timeout:
            _, curr_bytes = self.capture_secondary_frame()
            if prev_bytes and curr_bytes == prev_bytes:
                return True
            prev_bytes = curr_bytes
            time.sleep(poll_interval)
        return True

    def translate_and_validate(self, rel_x: int | float, rel_y: int | float) -> Tuple[int, int]:
        """
        Translates relative coordinates from Monitor 2's frame of reference
        to global X11 coordinate space:
          global_x = offset_x + relative_x
          global_y = offset_y + relative_y

        Enforces visual scale alignment between screenshot native resolution
        and physical display resolution, and ensures clicks never land inside Primary monitor.
        """
        cap_w, cap_h = self.last_capture_size
        scale_x = (self.secondary.width / cap_w) if cap_w > 0 else 1.0
        scale_y = (self.secondary.height / cap_h) if cap_h > 0 else 1.0

        # Auto-convert normalized floats [0.0, 1.0] if returned by model
        if isinstance(rel_x, float) and 0.0 <= rel_x <= 1.0:
            rx = int(round(rel_x * (self.secondary.width - 1)))
        else:
            rx = int(round(rel_x * scale_x))

        if isinstance(rel_y, float) and 0.0 <= rel_y <= 1.0:
            ry = int(round(rel_y * (self.secondary.height - 1)))
        else:
            ry = int(round(rel_y * scale_y))

        # Check relative boundaries within Monitor 2
        if rx < 0 or rx >= self.secondary.width or ry < 0 or ry >= self.secondary.height:
            raise ValueError(
                f"Relative coordinate ({rx}, {ry}) exceeds Monitor 2 dimensions "
                f"({self.secondary.width}x{self.secondary.height})."
            )

        # Global translation
        global_x = self.secondary.offset_x + rx
        global_y = self.secondary.offset_y + ry

        # Check Monitor 2 global containment
        if not self.secondary.contains_global(global_x, global_y):
            raise ValueError(
                f"Calculated coordinate ({global_x}, {global_y}) falls outside Monitor 2 boundary "
                f"[{self.secondary.offset_x}, {self.secondary.offset_x + self.secondary.width}) x "
                f"[{self.secondary.offset_y}, {self.secondary.offset_y + self.secondary.height})."
            )

        # Strict safety check: Never permit mouse in primary display
        if self.primary.contains_global(global_x, global_y):
            raise PermissionError(
                f"CRITICAL SAFETY VIOLATION: Coordinates ({global_x}, {global_y}) fall into "
                f"Primary display ({self.primary.name})! Action blocked."
            )

        return global_x, global_y
