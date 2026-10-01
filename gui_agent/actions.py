import io
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import shutil
import pyautogui
from gui_agent.display import DualMonitorLock

# Terminal ANSI Color codes
BOLD_CYAN = "\033[1;36m"
BOLD_GREEN = "\033[1;32m"
BOLD_YELLOW = "\033[1;33m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"

# Enable PyAutoGUI fail-safe: moving mouse to corner raises FailSafeException
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class EmergencyStopException(Exception):
    """Raised when an emergency killswitch is triggered (corner slam, sentinel file, or stop signal)."""
    pass


class OSActionHandler:
    """
    Executes OS-level UI input operations strictly locked to the secondary monitor.
    Provides tool callable schemas compatible with Google GenAI SDK, Agnes AI, and OpenAI/OpenRouter APIs.
    Features dynamic target snapping with OCR heuristics, human-like xdotool event dispatch,
    adaptive visual settling, and direct CLI command execution.
    Includes active emergency fail-safe checking before every input event.
    """

    FAILSAFE_SENTINEL_PATHS = [
        "/dev/shm/gui_agent_stop",
        "/tmp/gui_agent_stop",
    ]

    def __init__(self, monitor_lock: DualMonitorLock):
        self.lock = monitor_lock
        self.is_completed = False
        self.completion_summary = ""
        self.action_history: List[str] = []
        self._has_xdotool = shutil.which("xdotool") is not None
        self._has_tesseract = shutil.which("tesseract") is not None

    def check_failsafe(self) -> None:
        """
        Emergency Killswitch Checker:
        1. Checks for sentinel stop files (/dev/shm/gui_agent_stop or /tmp/gui_agent_stop).
        2. Checks for cursor slammed into any screen corner (top-left or other edges).
        3. Raises EmergencyStopException to immediately halt the agentic loop.
        """
        for stop_path in self.FAILSAFE_SENTINEL_PATHS:
            if os.path.exists(stop_path):
                try:
                    os.remove(stop_path)
                except Exception:
                    pass
                raise EmergencyStopException(f"Emergency stop file detected ({stop_path})! Halting execution immediately.")

        try:
            cx, cy = pyautogui.position()
            # Corner slam check: within 5 pixels of (0, 0)
            if cx <= 5 and cy <= 5:
                raise EmergencyStopException(f"Emergency stop: Mouse cursor pushed into top-left corner ({cx}, {cy})!")
            scr_w, scr_h = pyautogui.size()
            if (cx >= scr_w - 5 and cy <= 5) or (cx <= 5 and cy >= scr_h - 5) or (cx >= scr_w - 5 and cy >= scr_h - 5):
                raise EmergencyStopException(f"Emergency stop: Mouse cursor pushed into screen corner ({cx}, {cy})!")
        except EmergencyStopException:
            raise
        except Exception:
            pass

    def snap_to_target(self, x: int, y: int, radius_x: int = 60, radius_y: int = 30) -> Tuple[int, int]:
        """
        High-Precision Element Snapping:
        Inspects a local window around (x, y) on Monitor 2 using local tesseract OCR.
        - Automatically calculates exact bounding box center rather than clicking margins.
        - For form checkboxes and radio buttons, identifies adjacent label text on the same row
          and defaults the click target to the associated label text row for reliable trigger propagation.
        """
        try:
            gx, gy = self.lock.translate_and_validate(x, y)
            left = max(self.lock.secondary.offset_x, gx - radius_x)
            top = max(self.lock.secondary.offset_y, gy - radius_y)
            right = min(self.lock.secondary.offset_x + self.lock.secondary.width, gx + radius_x)
            bottom = min(self.lock.secondary.offset_y + self.lock.secondary.height, gy + radius_y)

            if right <= left or bottom <= top:
                return x, y

            from PIL import ImageGrab
            crop = ImageGrab.grab(bbox=(left, top, right, bottom))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")

            p = subprocess.Popen(
                ["tesseract", "stdin", "stdout", "--oem", "1", "-l", "eng", "tsv"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = p.communicate(input=buf.getvalue(), timeout=0.8)
            if p.returncode == 0:
                lines = stdout.decode("utf-8", errors="ignore").splitlines()
                words_info = []
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 12 and parts[11].strip():
                        try:
                            w_left = int(parts[6])
                            w_top = int(parts[7])
                            w_w = int(parts[8])
                            w_h = int(parts[9])
                            w_text = parts[11].strip()
                            g_cx = left + w_left + w_w / 2.0
                            g_cy = top + w_top + w_h / 2.0
                            words_info.append((g_cx, g_cy, w_left, w_top, w_w, w_h, w_text))
                        except Exception:
                            continue

                if words_info:
                    # 1. Checkbox/Radio detection: if click is slightly to the left of a text label on the same row
                    for g_cx, g_cy, w_left, w_top, w_w, w_h, w_text in words_info:
                        row_diff_y = abs(g_cy - gy)
                        col_diff_x = g_cx - gx
                        # Adjacent label within 10-60px to the right on the same horizontal row (within 16px vertical)
                        if 10 <= col_diff_x <= 60 and row_diff_y <= 16:
                            snapped_rx = int(round(g_cx - self.lock.secondary.offset_x))
                            snapped_ry = int(round(g_cy - self.lock.secondary.offset_y))
                            if 0 <= snapped_rx < self.lock.secondary.width and 0 <= snapped_ry < self.lock.secondary.height:
                                print(f"{BOLD_YELLOW}[Form Snapping]:{RESET} Checkbox/Radio detected. Snapped ({x}, {y}) -> ({snapped_rx}, {snapped_ry}) onto label '{w_text}'")
                                return snapped_rx, snapped_ry

                    # 2. General bounding box centering: find nearest text/input element center
                    best_dist = float("inf")
                    best_center = None
                    matched_text = ""
                    for g_cx, g_cy, _, _, _, _, w_text in words_info:
                        dist = ((g_cx - gx) ** 2 + (g_cy - gy) ** 2) ** 0.5
                        if dist < best_dist and dist <= 35:
                            best_dist = dist
                            best_center = (g_cx, g_cy)
                            matched_text = w_text

                    if best_center:
                        snapped_rx = int(round(best_center[0] - self.lock.secondary.offset_x))
                        snapped_ry = int(round(best_center[1] - self.lock.secondary.offset_y))
                        if 0 <= snapped_rx < self.lock.secondary.width and 0 <= snapped_ry < self.lock.secondary.height:
                            if (snapped_rx, snapped_ry) != (x, y):
                                print(f"{BOLD_YELLOW}[Target Snapping]:{RESET} Snapped ({x}, {y}) -> ({snapped_rx}, {snapped_ry}) on '{matched_text}'")
                                return snapped_rx, snapped_ry
        except Exception:
            pass
        return x, y

    def _snap_coordinates_ocr(self, x: int, y: int) -> Tuple[int, int]:
        """Backwards-compatible alias for snap_to_target."""
        return self.snap_to_target(x, y)

    def click(self, x: int, y: int) -> str:
        """
        Move mouse and left-click at specified relative coordinates on Monitor 2.
        Simulates human-like DOM registration: executes xdotool mousemove --sync {x} {y}
        followed by a 40ms micro-pause before xdotool click 1 so web browsers register hover
        states, focus indicators, and click events without dropping inputs.

        Args:
            x: Relative horizontal pixel coordinate on Monitor 2 (0 to width-1).
            y: Relative vertical pixel coordinate on Monitor 2 (0 to height-1).
        """
        self.check_failsafe()
        try:
            x, y = self.snap_to_target(x, y)
            gx, gy = self.lock.translate_and_validate(x, y)
        except Exception as err:
            msg = f"Click failed safety validation: {err}"
            print(f"{BOLD_RED}[Safety Block]:{RESET} {msg}")
            return msg

        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Clicking ({x}, {y}) on Screen 2 (Global: {gx}, {gy})...")
        try:
            if self._has_xdotool:
                subprocess.run(["xdotool", "mousemove", "--sync", str(gx), str(gy)], check=True)
                time.sleep(0.04)  # 40ms micro-pause for browser DOM focus & hover registration
                subprocess.run(["xdotool", "click", "1"], check=True)
            else:
                pyautogui.moveTo(gx, gy, duration=0.08)
                time.sleep(0.04)
                pyautogui.click(gx, gy)
        except Exception:
            pyautogui.moveTo(gx, gy, duration=0.08)
            time.sleep(0.04)
            pyautogui.click(gx, gy)

        time.sleep(0.05)
        result = f"Successfully clicked at ({x}, {y}) on Monitor 2 (Global: {gx}, {gy})."
        self.action_history.append(result)
        return result

    def double_click(self, x: int, y: int) -> str:
        """
        Move mouse and double-click at specified relative coordinates on Monitor 2.

        Args:
            x: Relative horizontal pixel coordinate on Monitor 2 (0 to width-1).
            y: Relative vertical pixel coordinate on Monitor 2 (0 to height-1).
        """
        self.check_failsafe()
        try:
            x, y = self.snap_to_target(x, y)
            gx, gy = self.lock.translate_and_validate(x, y)
        except Exception as err:
            msg = f"Double click failed safety validation: {err}"
            print(f"{BOLD_RED}[Safety Block]:{RESET} {msg}")
            return msg

        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Double-clicking ({x}, {y}) on Screen 2 (Global: {gx}, {gy})...")
        try:
            if self._has_xdotool:
                subprocess.run(["xdotool", "mousemove", "--sync", str(gx), str(gy)], check=True)
                time.sleep(0.04)
                subprocess.run(["xdotool", "click", "--repeat", "2", "--delay", "40", "1"], check=True)
            else:
                pyautogui.moveTo(gx, gy, duration=0.08)
                time.sleep(0.04)
                pyautogui.doubleClick(gx, gy)
        except Exception:
            pyautogui.moveTo(gx, gy, duration=0.08)
            time.sleep(0.04)
            pyautogui.doubleClick(gx, gy)

        time.sleep(0.05)
        result = f"Successfully double-clicked at ({x}, {y}) on Monitor 2 (Global: {gx}, {gy})."
        self.action_history.append(result)
        return result

    def right_click(self, x: int, y: int) -> str:
        """
        Move mouse and right-click (open context menu) at specified relative coordinates on Monitor 2.

        Args:
            x: Relative horizontal pixel coordinate on Monitor 2 (0 to width-1).
            y: Relative vertical pixel coordinate on Monitor 2 (0 to height-1).
        """
        self.check_failsafe()
        try:
            x, y = self.snap_to_target(x, y)
            gx, gy = self.lock.translate_and_validate(x, y)
        except Exception as err:
            msg = f"Right click failed safety validation: {err}"
            print(f"{BOLD_RED}[Safety Block]:{RESET} {msg}")
            return msg

        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Right-clicking ({x}, {y}) on Screen 2 (Global: {gx}, {gy})...")
        try:
            if self._has_xdotool:
                subprocess.run(["xdotool", "mousemove", "--sync", str(gx), str(gy)], check=True)
                time.sleep(0.04)
                subprocess.run(["xdotool", "click", "3"], check=True)
            else:
                pyautogui.moveTo(gx, gy, duration=0.08)
                time.sleep(0.04)
                pyautogui.rightClick(gx, gy)
        except Exception:
            pyautogui.moveTo(gx, gy, duration=0.08)
            time.sleep(0.04)
            pyautogui.rightClick(gx, gy)

        time.sleep(0.05)
        result = f"Successfully right-clicked at ({x}, {y}) on Monitor 2 (Global: {gx}, {gy})."
        self.action_history.append(result)
        return result

    def type_text(self, text: str, enter: bool = False) -> str:
        """
        Send keystrokes to the currently active/focused window.

        Args:
            text: Text string to type.
            enter: Whether to press Return/Enter after typing.
        """
        self.check_failsafe()
        enter_str = " (with Enter)" if enter else ""
        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Typing \"{text}\"{enter_str} to active window...")

        try:
            pyautogui.write(text, interval=0.01)
        except Exception:
            subprocess.run(["xdotool", "type", "--delay", "15", text], check=True)

        if enter:
            time.sleep(0.04)
            pyautogui.press("enter")

        time.sleep(0.05)
        result = f"Successfully typed text '{text}' (enter={enter})."
        self.action_history.append(result)
        return result

    def press_key(self, key: str) -> str:
        """
        Press a single key or key combination (e.g. 'Super', 'enter', 'tab', 'escape', 'ctrl+c', 'ctrl+t', 'ctrl+l').

        Args:
            key: Single key name or modifier combo.
        """
        self.check_failsafe()
        key_norm = key.strip()
        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Pressing key '{key_norm}'...")

        k_lower = key_norm.lower()

        # Handle Super/Win key explicitly for X11/antiX Linux
        if k_lower in ("super", "win", "windows", "super_l", "super_r"):
            try:
                subprocess.run(["xdotool", "key", "Super_L"], check=True)
                res = "Successfully pressed Super key via xdotool."
                time.sleep(0.05)
                self.action_history.append(res)
                return res
            except Exception:
                pass

        if "+" in k_lower:
            parts = [p.strip().lower() for p in k_lower.split("+")]
            mod_map = {
                "control": "ctrl",
                "return": "enter",
                "esc": "escape",
                "super": "super",
                "win": "super",
            }
            parts = [mod_map.get(p, p) for p in parts]
            try:
                pyautogui.hotkey(*parts)
                res = f"Successfully executed key combo '{key_norm}' via pyautogui."
            except Exception:
                if self._has_xdotool:
                    subprocess.run(["xdotool", "key", key_norm], check=True)
                    res = f"Successfully executed key combo '{key_norm}' via xdotool."
                else:
                    res = f"Failed to press key combo '{key_norm}': xdotool not available."
        else:
            aliases = {
                "return": "enter",
                "esc": "escape",
                "page_up": "pageup",
                "page_down": "pagedown",
                "super": "winleft",
                "win": "winleft",
            }
            target = aliases.get(k_lower, k_lower)
            if target in pyautogui.KEY_NAMES:
                pyautogui.press(target)
                res = f"Successfully pressed key '{key_norm}' via pyautogui."
            else:
                try:
                    if self._has_xdotool:
                        subprocess.run(["xdotool", "key", key_norm], check=True)
                        res = f"Successfully pressed key '{key_norm}' via xdotool."
                    else:
                        res = f"Failed to press key '{key_norm}': key not in PyAutoGUI and xdotool missing."
                except Exception as exc:
                    res = f"Failed to press key '{key_norm}': {exc}"

        time.sleep(0.05)
        self.action_history.append(res)
        return res

    def wait(self, seconds: float = 0.5) -> str:
        """
        Pause execution adaptively to allow UI settling or animations.
        Uses adaptive screen stabilization check with a tight 0.5s default delay.
        Monitors emergency killswitch throughout the wait period.

        Args:
            seconds: Number of seconds to pause (default: 0.5).
        """
        self.check_failsafe()
        duration = max(0.1, min(float(seconds), 10.0))
        print(f"{BOLD_CYAN}[Agent Action]:{RESET} Waiting {duration:.1f}s (adaptive settling)...")
        start = time.time()
        while (time.time() - start) < duration:
            self.check_failsafe()
            time.sleep(min(0.05, duration - (time.time() - start)))
        self.lock.wait_for_screen_stabilization(timeout=0.2, poll_interval=0.1)
        res = f"Successfully waited {duration:.1f} second(s) with screen stabilization."
        self.action_history.append(res)
        return res

    def run_command(self, command: str) -> str:
        """
        Executes a non-blocking or short bash command on antiX Linux when CLI execution is faster than GUI clicks.

        Args:
            command: Shell command string to execute in bash (e.g. 'curl ...', 'xdotool ...', 'pgrep ...').
        """
        self.check_failsafe()
        cmd_str = command.strip()
        print(f"{BOLD_CYAN}[Shell Command]:{RESET} Executing '{cmd_str}'...")

        try:
            res = subprocess.run(
                ["bash", "-c", cmd_str],
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            code = res.returncode

            result_str = f"Exit code: {code}"
            if out:
                result_str += f"\nStdout: {out[:1000]}"
            if err:
                result_str += f"\nStderr: {err[:500]}"
        except subprocess.TimeoutExpired:
            result_str = f"Command '{cmd_str}' timed out after 15 seconds."
        except Exception as exc:
            result_str = f"Failed to execute command '{cmd_str}': {exc}"

        print(f"{BOLD_YELLOW}[Shell Output]:{RESET} {result_str[:200]}")
        self.action_history.append(result_str)
        return result_str

    def task_complete(self, summary: str) -> str:
        """
        End loop when testing assertion or task goal passes.

        Args:
            summary: Detailed summary of test results and observations confirming task completion.
        """
        self.is_completed = True
        self.completion_summary = summary
        print(f"\n{BOLD_GREEN}[Task Complete]:{RESET} {summary}\n")
        res = f"Task completed successfully. Summary: {summary}"
        self.action_history.append(res)
        return res

    def execute_tool_by_name(self, name: str, args: Dict[str, Any]) -> str:
        """Executes a tool dynamically by name and arguments."""
        tool_map = {
            "click": lambda: self.click(int(args.get("x", 0)), int(args.get("y", 0))),
            "double_click": lambda: self.double_click(int(args.get("x", 0)), int(args.get("y", 0))),
            "right_click": lambda: self.right_click(int(args.get("x", 0)), int(args.get("y", 0))),
            "type_text": lambda: self.type_text(str(args.get("text", "")), bool(args.get("enter", False))),
            "press_key": lambda: self.press_key(str(args.get("key", ""))),
            "wait": lambda: self.wait(float(args.get("seconds", 0.5))),
            "run_command": lambda: self.run_command(str(args.get("command", ""))),
            "task_complete": lambda: self.task_complete(str(args.get("summary", "Done"))),
        }

        func = tool_map.get(name)
        if not func:
            msg = f"Unknown action '{name}'. Available: {list(tool_map.keys())}"
            print(f"{BOLD_RED}[Error]:{RESET} {msg}")
            return msg

        try:
            return func()
        except Exception as e:
            msg = f"Execution error in {name}: {e}"
            print(f"{BOLD_RED}[Execution Error]:{RESET} {msg}")
            return msg

    def get_tool_callables(self) -> list:
        """Returns list of functional tools for Google GenAI SDK."""
        return [
            self.click,
            self.double_click,
            self.right_click,
            self.type_text,
            self.press_key,
            self.wait,
            self.run_command,
            self.task_complete,
        ]

    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns tool definitions formatted for OpenAI / Agnes / Groq / Bytez / OpenRouter function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Move mouse and left-click at relative coordinates (x, y) on Monitor 2 with human DOM registration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Relative pixel X coordinate on Monitor 2"},
                            "y": {"type": "integer", "description": "Relative pixel Y coordinate on Monitor 2"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "double_click",
                    "description": "Move mouse and double-click at relative coordinates (x, y) on Monitor 2.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Relative pixel X coordinate on Monitor 2"},
                            "y": {"type": "integer", "description": "Relative pixel Y coordinate on Monitor 2"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "right_click",
                    "description": "Open context menu at relative coordinates (x, y) on Monitor 2.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "Relative pixel X coordinate on Monitor 2"},
                            "y": {"type": "integer", "description": "Relative pixel Y coordinate on Monitor 2"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text into active focused element on Monitor 2.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text string to type"},
                            "enter": {"type": "boolean", "description": "Whether to press Enter after typing (default: false)"},
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "press_key",
                    "description": "Press a key or combination (e.g. 'Super', 'enter', 'tab', 'escape', 'ctrl+c', 'ctrl+t', 'ctrl+l').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Key name or hotkey combination"},
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wait",
                    "description": "Pause execution adaptively to allow UI settling or page loading.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seconds": {"type": "number", "description": "Pause duration in seconds (default: 0.5)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Executes a non-blocking or short bash command on antiX Linux when CLI execution is faster than GUI clicks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The bash shell command to execute"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "task_complete",
                    "description": "Call when test assertion or task goal is achieved and verified on Monitor 2.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of observations and results"},
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]
