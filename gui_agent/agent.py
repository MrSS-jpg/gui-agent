import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

from gui_agent.actions import (
    BOLD_CYAN,
    BOLD_GREEN,
    BOLD_RED,
    BOLD_YELLOW,
    RESET,
    EmergencyStopException,
    OSActionHandler,
)
from gui_agent.display import DualMonitorLock
from gui_agent.key_extractor import ExtractedKeyStore

# System instruction for vision operators
SYSTEM_PROMPT_TEMPLATE = """You are an autonomous, lightweight visual GUI operator ("hands and eyes") strictly locked to Monitor 2 on an antiX Linux (X11) system.
Your mission is to perform UI and software testing operations step-by-step based on visual feedback.

### Screen Resolution & Coordinates
- You are operating strictly on Monitor 2.
- Resolution: {width} x {height} pixels. Top-left is (0, 0), bottom-right is ({max_x}, {max_y}).
- All coordinates (x, y) provided to tools MUST be integers:
  - 0 <= x <= {max_x}
  - 0 <= y <= {max_y}

### Available Tools
1. `click(x, y)`: Move and left-click at relative pixel (x, y) on Monitor 2.
2. `double_click(x, y)`: Move and double-click at pixel (x, y) on Monitor 2.
3. `right_click(x, y)`: Open context menu at pixel (x, y) on Monitor 2.
4. `type_text(text, enter)`: Type text into active element (enter=True to submit).
5. `press_key(key)`: Send a key or key combo (e.g. 'Super', 'enter', 'tab', 'escape', 'ctrl+c').
6. `wait(seconds)`: Pause for UI settling or animations (e.g. seconds=0.5).
7. `run_command(command)`: Execute a short bash shell command on antiX Linux when CLI execution is faster than GUI clicks.
8. `task_complete(summary)`: Call when test assertions pass and testing is complete.

### Instructions
- Observe the visual state in the provided screenshot.
- Call the appropriate tool for each step.
- When the goal is completed or verified, call `task_complete(summary)`.
"""


class OpenRouterProvider:
    """
    Multimodal vision provider using OpenRouter API with OpenAI-compatible client.
    Supports free vision models and native function calling.
    """

    DEFAULT_CANDIDATES = [
        "inclusionai/ling-3.0-flash-vl:free",
        "nex-agi/nex-n2.5-pro:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "google/gemini-2.0-flash-exp:free",
    ]

    def __init__(self, api_key: str, preferred_model: Optional[str] = None):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = preferred_model or self.DEFAULT_CANDIDATES[0]
        self.messages: List[Dict[str, Any]] = []

    def initialize_session(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 2,
    ) -> Optional[str]:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        # Add new user message with text and image
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
        self.messages.append(user_message)

        # Sliding window: keep only the last max_sliding_frames with image_url
        self._prune_older_images(max_sliding_frames)

        tools = action_handler.get_openai_tool_definitions()

        # Try active model, fallback to other candidates if 404 or 429
        candidate_models = [self.model] + [
            m for m in self.DEFAULT_CANDIDATES if m != self.model
        ]

        response = None
        for cand in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=cand,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                )
                self.model = cand
                break
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "429" in err_str or "not found" in err_str.lower():
                    print(f"{BOLD_YELLOW}[OpenRouter Model Fallback]:{RESET} {cand} failed ({exc}). Trying next candidate...")
                    continue
                raise

        if not response:
            raise RuntimeError("All OpenRouter candidate models failed.")

        msg = response.choices[0].message
        self.messages.append(msg.to_dict() if hasattr(msg, "to_dict") else {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (msg.tool_calls or [])] or None,
        })

        # Execute any native tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}

                tool_result = action_handler.execute_tool_by_name(fn_name, args)

                # Record tool response in message history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_result,
                })

        # Fallback: check if model returned JSON in text content
        elif msg.content:
            parsed = self._extract_json_action(msg.content)
            if parsed and "action" in parsed:
                action_handler.execute_tool_by_name(
                    parsed["action"], parsed.get("args", {})
                )

        return msg.content

    def _prune_older_images(self, max_frames: int):
        """Replaces older images in messages with text placeholders to keep context low."""
        image_entries = []
        for msg_idx, msg in enumerate(self.messages):
            if isinstance(msg.get("content"), list):
                for part_idx, part in enumerate(msg["content"]):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_entries.append((msg_idx, part_idx))

        if len(image_entries) > max_frames:
            for m_idx, p_idx in image_entries[:-max_frames]:
                self.messages[m_idx]["content"][p_idx] = {
                    "type": "text",
                    "text": "[Previous frame image pruned to maintain 2-frame sliding window]",
                }

    def _extract_json_action(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(r"\{[\s\S]*\"action\"[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


class NvidiaNIMProvider:
    """
    Multimodal vision provider using NVIDIA NIM API with meta/llama-3.2-11b-vision-instruct.
    Enforces strictly ONE image per prompt by stripping historical images from prior turns.
    """

    def __init__(self, api_key: str, model: str = "meta/llama-3.2-11b-vision-instruct"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )
        self.messages: List[Dict[str, Any]] = []

    def initialize_session(self, system_prompt: str):
        instructions = (
            system_prompt
            + "\nTo execute an action, output ONLY a single JSON block:\n"
            + '{"action": "<tool_name>", "args": {<arguments>}}\n'
            + "Example: {\"action\": \"press_key\", \"args\": {\"key\": \"Super\"}}\n"
            + "Example: {\"action\": \"wait\", \"args\": {\"seconds\": 1.0}}\n"
            + "Example: {\"action\": \"task_complete\", \"args\": {\"summary\": \"Test passed\"}}\n"
        )
        self.messages = [{"role": "system", "content": instructions}]

    def _strip_historical_images(self):
        """
        NVIDIA NIM strictly enforces at most 1 image per prompt, and requires
        the number of <image> tokens in the prompt to match the number of images.
        This strips all image_url parts and removes any <image> tags from all
        prior conversation turns.
        """
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            # Drop historical image completely
                            continue
                        elif part.get("type") == "text":
                            # Strip <image> tags from prior turn text
                            txt = part.get("text", "").replace("<image>", "").strip()
                            if txt:
                                new_parts.append({"type": "text", "text": txt})
                        else:
                            new_parts.append(part)
                msg["content"] = new_parts
            elif isinstance(content, str):
                msg["content"] = content.replace("<image>", "").strip()

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 1,
    ) -> Optional[str]:
        # 1. Strip all historical images and <image> tags from prior turns
        self._strip_historical_images()

        # 2. Attach strictly ONE image (the latest screenshot) with <image> tag
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        cleaned_prompt = text_prompt.replace("<image>", "").strip()
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<image>\n{cleaned_prompt}\nOutput your JSON action now."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
        self.messages.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=150,
        )

        content = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})

        # Parse and execute JSON action
        parsed = self._extract_json_action(content)
        if parsed and "action" in parsed:
            action_handler.execute_tool_by_name(
                parsed["action"], parsed.get("args", {})
            )
        else:
            if "super" in content.lower() and not action_handler.is_completed:
                action_handler.press_key("Super")

        return content

    def _extract_json_action(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(r"\{[\s\S]*\"action\"[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


class GroqProvider:
    """
    Multimodal vision provider using Groq OpenAI-compatible API endpoint
    (https://api.groq.com/openai/v1).
    Primary model: meta-llama/llama-3.2-11b-vision-preview (with fallback to qwen/qwen3.6-27b).
    Enforces strictly ONE image per prompt and handles plain text without tool calls by parsing or retrying.
    """

    DEFAULT_MODELS = [
        "meta-llama/llama-3.2-11b-vision-preview",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
    ]

    def __init__(self, api_key: str, preferred_model: Optional[str] = None):
        self.api_key = api_key
        self.model = preferred_model or self.DEFAULT_MODELS[0]
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )
        self.messages: List[Dict[str, Any]] = []

    def initialize_session(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _strip_historical_images(self):
        """Ensures historical messages only contain text, dropping prior images."""
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            continue
                        elif part.get("type") == "text":
                            txt = part.get("text", "").replace("<image>", "").strip()
                            if txt:
                                new_parts.append({"type": "text", "text": txt})
                        else:
                            new_parts.append(part)
                msg["content"] = new_parts
            elif isinstance(content, str):
                msg["content"] = content.replace("<image>", "").strip()

    def _parse_text_action(self, text: str, action_handler: OSActionHandler) -> bool:
        """
        Parses coordinates or action descriptions from text output if tool_calls was omitted.
        Returns True if an action was successfully parsed and executed.
        """
        # 1. Look for JSON block: {"action": "...", "args": {...}}
        parsed = self._extract_json_action(text)
        if parsed and "action" in parsed:
            action_handler.execute_tool_by_name(parsed["action"], parsed.get("args", {}))
            return True

        # 1b. Look for run_command("command")
        m_cmd = re.search(r"run_command\s*\(\s*[\"'](.*?)[\"']\s*\)", text, re.I)
        if m_cmd:
            action_handler.run_command(m_cmd.group(1).strip())
            return True

        # 2. Look for function call syntax: click(x, y)
        m_click = re.search(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.I)
        if m_click:
            action_handler.click(int(m_click.group(1)), int(m_click.group(2)))
            return True

        # 3. Look for press_key("key")
        m_key = re.search(r"press_key\s*\(\s*[\"']?([^\"'\)]+)[\"']?\s*\)", text, re.I)
        if m_key:
            action_handler.press_key(m_key.group(1).strip())
            return True

        # 4. Look for wait(seconds)
        m_wait = re.search(r"wait\s*\(\s*(\d+(?:\.\d+)?)\s*\)", text, re.I)
        if m_wait:
            action_handler.wait(float(m_wait.group(1)))
            return True

        # 5. Look for task_complete("summary")
        m_complete = re.search(r"task_complete\s*\(\s*[\"']([^\"']+)[\"']\s*\)", text, re.I)
        if m_complete:
            action_handler.task_complete(m_complete.group(1).strip())
            return True

        # 6. Natural language keywords
        if re.search(r"\bpress(?:ing)?\s+(?:the\s+)?super\b", text, re.I):
            action_handler.press_key("Super")
            return True

        return False

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 1,
    ) -> Optional[str]:
        # 1. Strip all historical images
        self._strip_historical_images()

        # 2. Attach single latest screenshot
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        clean_prompt = text_prompt.replace("<image>", "").strip()
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": clean_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
        self.messages.append(user_message)

        tools = action_handler.get_openai_tool_definitions()
        candidate_models = [self.model] + [m for m in self.DEFAULT_MODELS if m != self.model]

        response = None
        for cand in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=cand,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=150,
                )
                self.model = cand
                break
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "model_not_found" in err_str or "400" in err_str:
                    print(f"{BOLD_YELLOW}[Groq Model Fallback]:{RESET} {cand} failed. Trying next candidate...")
                    continue
                raise

        if not response:
            raise RuntimeError("All Groq candidate models failed.")

        msg = response.choices[0].message
        self.messages.append(msg.to_dict() if hasattr(msg, "to_dict") else {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (msg.tool_calls or [])] or None,
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}
                tool_result = action_handler.execute_tool_by_name(fn_name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_result,
                })
        else:
            # Model responded with plain text without a tool call
            content = msg.content or ""
            if content:
                print(f"{BOLD_YELLOW}[Model Text Output]:{RESET}\n{content.strip()}")

            action_executed = self._parse_text_action(content, action_handler)
            if not action_executed and not action_handler.is_completed:
                print(f"{BOLD_YELLOW}[Action Retry]:{RESET} No tool call detected in response. Prompting model to execute action...")
                retry_message = {
                    "role": "user",
                    "content": (
                        "You replied with text but did not execute a tool call. "
                        "Please immediately call one of the available tools "
                        "(click, double_click, right_click, type_text, press_key, wait, task_complete) "
                        "or output your action in JSON format: {\"action\": \"...\", \"args\": {...}}"
                    ),
                }
                self.messages.append(retry_message)
                try:
                    retry_resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=150,
                    )
                    r_msg = retry_resp.choices[0].message
                    self.messages.append(r_msg.to_dict() if hasattr(r_msg, "to_dict") else {
                        "role": "assistant",
                        "content": r_msg.content,
                        "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (r_msg.tool_calls or [])] or None,
                    })
                    if r_msg.tool_calls:
                        for tc in r_msg.tool_calls:
                            fn_name = tc.function.name
                            try:
                                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            except Exception:
                                args = {}
                            tool_result = action_handler.execute_tool_by_name(fn_name, args)
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": fn_name,
                                "content": tool_result,
                            })
                    elif r_msg.content:
                        print(f"{BOLD_YELLOW}[Model Retry Text]:{RESET}\n{r_msg.content.strip()}")
                        self._parse_text_action(r_msg.content, action_handler)
                except Exception as retry_err:
                    print(f"{BOLD_RED}[Retry Error]:{RESET} {retry_err}")

        return msg.content

    def _extract_json_action(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(r"\{[\s\S]*\"action\"[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


class AgnesProvider:
    """
    Multimodal vision provider using Agnes AI OpenAI-compatible endpoint
    (https://apihub.agnes-ai.com/v1).
    Primary model: agnes-2.5-flash (with automatic fallback to agnes-2.0-flash).
    Enforces single-image prompt sanitization (_strip_historical_images()) to prevent multi-turn payload errors.
    Supports native tool calling and text action parsing (including run_command).
    """

    DEFAULT_MODELS = [
        "agnes-2.5-flash",
        "agnes-2.0-flash",
    ]

    def __init__(self, api_key: str, preferred_model: Optional[str] = None):
        self.api_key = api_key
        self.model = preferred_model or self.DEFAULT_MODELS[0]
        self.client = OpenAI(
            base_url="https://apihub.agnes-ai.com/v1",
            api_key=self.api_key,
        )
        self.messages: List[Dict[str, Any]] = []

    def initialize_session(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _strip_historical_images(self):
        """Ensures historical messages only contain text, dropping prior images."""
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            continue
                        elif part.get("type") == "text":
                            txt = part.get("text", "").replace("<image>", "").strip()
                            if txt:
                                new_parts.append({"type": "text", "text": txt})
                        else:
                            new_parts.append(part)
                msg["content"] = new_parts
            elif isinstance(content, str):
                msg["content"] = content.replace("<image>", "").strip()

    def _parse_text_action(self, text: str, action_handler: OSActionHandler) -> bool:
        parsed = self._extract_json_action(text)
        if parsed and "action" in parsed:
            action_handler.execute_tool_by_name(parsed["action"], parsed.get("args", {}))
            return True

        m_cmd = re.search(r"run_command\s*\(\s*[\"'](.*?)[\"']\s*\)", text, re.I)
        if m_cmd:
            action_handler.run_command(m_cmd.group(1).strip())
            return True

        m_click = re.search(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.I)
        if m_click:
            action_handler.click(int(m_click.group(1)), int(m_click.group(2)))
            return True

        m_key = re.search(r"press_key\s*\(\s*[\"']?([^\"'\)]+)[\"']?\s*\)", text, re.I)
        if m_key:
            action_handler.press_key(m_key.group(1).strip())
            return True

        m_wait = re.search(r"wait\s*\(\s*(\d+(?:\.\d+)?)\s*\)", text, re.I)
        if m_wait:
            action_handler.wait(float(m_wait.group(1)))
            return True

        m_complete = re.search(r"task_complete\s*\(\s*[\"']([^\"']+)[\"']\s*\)", text, re.I)
        if m_complete:
            action_handler.task_complete(m_complete.group(1).strip())
            return True

        if re.search(r"\bpress(?:ing)?\s+(?:the\s+)?super\b", text, re.I):
            action_handler.press_key("Super")
            return True

        return False

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 1,
    ) -> Optional[str]:
        self._strip_historical_images()

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        clean_prompt = text_prompt.replace("<image>", "").strip()
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": clean_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
        self.messages.append(user_message)

        tools = action_handler.get_openai_tool_definitions()
        candidate_models = [self.model] + [m for m in self.DEFAULT_MODELS if m != self.model]

        response = None
        for cand in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=cand,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=150,
                )
                self.model = cand
                break
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "model_not_found" in err_str or "401" in err_str:
                    print(f"{BOLD_YELLOW}[Agnes Model Fallback]:{RESET} {cand} failed. Trying next candidate...")
                    continue
                raise

        if not response:
            raise RuntimeError("All Agnes candidate models failed.")

        msg = response.choices[0].message
        self.messages.append(msg.to_dict() if hasattr(msg, "to_dict") else {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (msg.tool_calls or [])] or None,
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}
                tool_result = action_handler.execute_tool_by_name(fn_name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_result,
                })
        elif msg.content:
            action_found = self._parse_text_action(msg.content, action_handler)
            if not action_found:
                print(f"{BOLD_YELLOW}[Agnes Model Text]:{RESET}\n{msg.content.strip()}")

        return msg.content

    def _extract_json_action(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(r"\{[\s\S]*\"action\"[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


class BytezProvider:
    """
    Multimodal vision provider using Bytez OpenAI-compatible endpoint
    (https://api.bytez.com/models/v2/openai).
    Enforces single-frame screenshot delivery (_strip_historical_images()) to prevent multi-turn payload bloat.
    Handles native tool calling and text action parsing.
    """

    DEFAULT_MODELS = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash",
    ]

    def __init__(self, api_key: str, preferred_model: Optional[str] = None):
        self.api_key = api_key
        self.model = preferred_model or self.DEFAULT_MODELS[0]
        self.client = OpenAI(
            base_url="https://api.bytez.com/models/v2/openai",
            api_key=self.api_key,
        )
        self.messages: List[Dict[str, Any]] = []

    def initialize_session(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _strip_historical_images(self):
        """Ensures historical messages only contain text, dropping prior images."""
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            continue
                        elif part.get("type") == "text":
                            txt = part.get("text", "").replace("<image>", "").strip()
                            if txt:
                                new_parts.append({"type": "text", "text": txt})
                        else:
                            new_parts.append(part)
                msg["content"] = new_parts
            elif isinstance(content, str):
                msg["content"] = content.replace("<image>", "").strip()

    def _parse_text_action(self, text: str, action_handler: OSActionHandler) -> bool:
        parsed = self._extract_json_action(text)
        if parsed and "action" in parsed:
            action_handler.execute_tool_by_name(parsed["action"], parsed.get("args", {}))
            return True

        m_cmd = re.search(r"run_command\s*\(\s*[\"'](.*?)[\"']\s*\)", text, re.I)
        if m_cmd:
            action_handler.run_command(m_cmd.group(1).strip())
            return True

        m_click = re.search(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.I)
        if m_click:
            action_handler.click(int(m_click.group(1)), int(m_click.group(2)))
            return True

        m_key = re.search(r"press_key\s*\(\s*[\"']?([^\"'\)]+)[\"']?\s*\)", text, re.I)
        if m_key:
            action_handler.press_key(m_key.group(1).strip())
            return True

        m_wait = re.search(r"wait\s*\(\s*(\d+(?:\.\d+)?)\s*\)", text, re.I)
        if m_wait:
            action_handler.wait(float(m_wait.group(1)))
            return True

        m_complete = re.search(r"task_complete\s*\(\s*[\"']([^\"']+)[\"']\s*\)", text, re.I)
        if m_complete:
            action_handler.task_complete(m_complete.group(1).strip())
            return True

        if re.search(r"\bpress(?:ing)?\s+(?:the\s+)?super\b", text, re.I):
            action_handler.press_key("Super")
            return True

        return False

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 1,
    ) -> Optional[str]:
        # 1. Strip prior images to ensure single-frame delivery
        self._strip_historical_images()

        # 2. Attach single latest frame
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        clean_prompt = text_prompt.replace("<image>", "").strip()
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": clean_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
        self.messages.append(user_message)

        tools = action_handler.get_openai_tool_definitions()
        candidate_models = [self.model] + [m for m in self.DEFAULT_MODELS if m != self.model]

        response = None
        for cand in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=cand,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=150,
                )
                self.model = cand
                break
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "model_not_found" in err_str or "401" in err_str:
                    print(f"{BOLD_YELLOW}[Bytez Model Fallback]:{RESET} {cand} failed. Trying next candidate...")
                    continue
                raise

        if not response:
            raise RuntimeError("All Bytez candidate models failed.")

        msg = response.choices[0].message
        self.messages.append(msg.to_dict() if hasattr(msg, "to_dict") else {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (msg.tool_calls or [])] or None,
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}
                tool_result = action_handler.execute_tool_by_name(fn_name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_result,
                })
        elif msg.content:
            action_found = self._parse_text_action(msg.content, action_handler)
            if not action_found:
                print(f"{BOLD_YELLOW}[Bytez Model Text]:{RESET}\n{msg.content.strip()}")

        return msg.content

    def _extract_json_action(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            m = re.search(r"\{[\s\S]*\"action\"[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return None


class GeminiProvider:
    """
    Multimodal vision provider using official google-genai SDK.
    Rotates through remaining working Gemini keys from the pool on 429 quota errors.
    """

    def __init__(self, api_keys: List[str], model_name: str = "gemini-3.6-flash"):
        self.api_keys = list(api_keys)
        self.current_key_idx = 0
        self.model = model_name
        self.client = None
        self.chat = None
        self.system_prompt = ""
        self.action_handler = None
        self._init_client()

    def _init_client(self):
        from google import genai
        active_key = self.api_keys[self.current_key_idx]
        # Prevent environment from overriding the active key
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key
        self.client = genai.Client(api_key=active_key)

    def _rotate_key(self) -> bool:
        """Rotates to next available Gemini key in pool. Returns True if next key available."""
        if self.current_key_idx + 1 < len(self.api_keys):
            self.current_key_idx += 1
            new_key = self.api_keys[self.current_key_idx]
            masked = f"{new_key[:8]}...{new_key[-4:]}"
            print(f"{BOLD_YELLOW}[Gemini Key Rotation]:{RESET} Key hit 429 quota. Rotating to next Gemini key ({masked})...")
            self._init_client()
            return True
        return False

    def initialize_session(self, system_prompt: str, action_handler: OSActionHandler):
        from google.genai import types
        self.system_prompt = system_prompt
        self.action_handler = action_handler
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=action_handler.get_tool_callables(),
            temperature=0.0,
        )
        self.chat = self.client.chats.create(model=self.model, config=config)

    def send_turn(
        self,
        text_prompt: str,
        image_bytes: bytes,
        action_handler: OSActionHandler,
        max_sliding_frames: int = 2,
    ) -> Optional[str]:
        from google.genai import types

        while True:
            try:
                img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                resp = self.chat.send_message([text_prompt, img_part])
                self._prune_older_images(max_sliding_frames)
                return resp.text
            except Exception as err:
                err_msg = str(err)
                if ("429" in err_msg or "resource_exhausted" in err_msg.lower() or "quota" in err_msg.lower()):
                    if self._rotate_key():
                        self.initialize_session(self.system_prompt, action_handler)
                        continue
                raise

    def _prune_older_images(self, max_frames: int):
        from google.genai import types
        for hist_attr in ("_curated_history", "_comprehensive_history"):
            hist = getattr(self.chat, hist_attr, None)
            if not hist:
                continue
            image_refs = []
            for msg in hist:
                for idx, part in enumerate(msg.parts):
                    if getattr(part, "inline_data", None) is not None:
                        image_refs.append((msg, idx))
            if len(image_refs) > max_frames:
                for msg, idx in image_refs[:-max_frames]:
                    msg.parts[idx] = types.Part.from_text(
                        text="[Previous frame screenshot pruned]"
                    )


class VisionActionAgent:
    """
    Multi-provider vision-action agentic loop supporting Gemini, Groq, NVIDIA NIM, and OpenRouter
    with seamless automatic fallback on quota exhaustion (429).
    """

    def __init__(
        self,
        keystore: ExtractedKeyStore,
        monitor_lock: DualMonitorLock,
        preferred_provider: str = "auto",
        preferred_model: Optional[str] = None,
        step_delay: float = 0.5,
    ):
        self.keystore = keystore
        self.lock = monitor_lock
        self.preferred_provider = preferred_provider.lower()
        self.preferred_model = preferred_model
        self.step_delay = step_delay
        self.action_handler = OSActionHandler(self.lock)
        self.active_provider = None
        self.active_provider_name = ""

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            width=self.lock.secondary.width,
            height=self.lock.secondary.height,
            max_x=self.lock.secondary.width - 1,
            max_y=self.lock.secondary.height - 1,
        )

    def _init_provider(self, goal: str) -> bool:
        """
        Initializes an active multimodal provider with fallback chain:
        Agnes -> Groq -> Bytez -> NVIDIA NIM -> Gemini -> OpenRouter (or preferred provider).
        """
        sys_prompt = self._build_system_prompt()
        _, init_bytes = self.lock.capture_secondary_frame()

        if self.preferred_provider == "agnes":
            providers_to_try = ["agnes", "groq", "bytez", "nvidia", "gemini", "openrouter"]
        elif self.preferred_provider == "groq":
            providers_to_try = ["groq", "agnes", "bytez", "nvidia", "gemini", "openrouter"]
        elif self.preferred_provider == "bytez":
            providers_to_try = ["bytez", "agnes", "groq", "nvidia", "gemini", "openrouter"]
        elif self.preferred_provider == "nvidia":
            providers_to_try = ["nvidia", "agnes", "groq", "bytez", "gemini", "openrouter"]
        elif self.preferred_provider == "gemini":
            providers_to_try = ["gemini", "agnes", "groq", "bytez", "nvidia", "openrouter"]
        elif self.preferred_provider == "openrouter":
            providers_to_try = ["openrouter", "agnes", "groq", "bytez", "nvidia", "gemini"]
        else:
            # Default auto-fallback chain: [Agnes -> Groq -> Bytez -> NVIDIA NIM -> Gemini -> OpenRouter]
            providers_to_try = ["agnes", "groq", "bytez", "nvidia", "gemini", "openrouter"]

        for prov_name in providers_to_try:
            try:
                if prov_name == "agnes":
                    agnes_key = self.keystore.get_agnes_key()
                    if not agnes_key:
                        continue
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to Agnes AI vision endpoint...")
                    p = AgnesProvider(api_key=agnes_key, preferred_model=self.preferred_model)
                    p.initialize_session(sys_prompt)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"Agnes AI ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True
                elif prov_name == "gemini":
                    gem_keys = self.keystore.get_gemini_keys_pool()
                    if not gem_keys:
                        continue
                    masked_first = f"{gem_keys[0][:8]}...{gem_keys[0][-4:]}"
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to Google Gemini vision endpoint (pool of {len(gem_keys)} key(s), starting with {masked_first})...")
                    p = GeminiProvider(api_keys=gem_keys, model_name=self.preferred_model or "gemini-3.6-flash")
                    p.initialize_session(sys_prompt, self.action_handler)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"Google Gemini ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True

                elif prov_name == "groq":
                    groq_key = self.keystore.get_groq_key()
                    if not groq_key:
                        continue
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to Groq vision endpoint...")
                    p = GroqProvider(api_key=groq_key, preferred_model=self.preferred_model)
                    p.initialize_session(sys_prompt)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"Groq ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True

                elif prov_name == "bytez":
                    bytez_key = self.keystore.get_bytez_key()
                    if not bytez_key:
                        continue
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to Bytez vision endpoint...")
                    p = BytezProvider(api_key=bytez_key, preferred_model=self.preferred_model)
                    p.initialize_session(sys_prompt)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"Bytez ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True

                elif prov_name == "nvidia":
                    nv_key = self.keystore.get_nvidia_key()
                    if not nv_key:
                        continue
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to NVIDIA NIM vision endpoint...")
                    p = NvidiaNIMProvider(api_key=nv_key, model=self.preferred_model or "meta/llama-3.2-11b-vision-instruct")
                    p.initialize_session(sys_prompt)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"NVIDIA NIM ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True

                elif prov_name == "openrouter":
                    or_key = self.keystore.get_openrouter_key()
                    if not or_key:
                        continue
                    print(f"{BOLD_CYAN}[Provider Init]:{RESET} Connecting to OpenRouter free vision endpoint...")
                    p = OpenRouterProvider(api_key=or_key, preferred_model=self.preferred_model)
                    p.initialize_session(sys_prompt)
                    probe_prompt = f"Task Goal: {goal}\nInitial Monitor 2 screenshot provided. Plan and execute your first action."
                    resp_text = p.send_turn(probe_prompt, init_bytes, self.action_handler)
                    self.active_provider = p
                    self.active_provider_name = f"OpenRouter ({p.model})"
                    if resp_text and not self.action_handler.is_completed:
                        print(f"{BOLD_YELLOW}[Agent Plan]:{RESET} {resp_text.strip()}")
                    return True

            except Exception as err:
                err_msg = str(err)
                if "429" in err_msg or "resource_exhausted" in err_msg.lower() or "quota" in err_msg.lower():
                    print(f"{BOLD_YELLOW}[Provider Fallback]:{RESET} {prov_name} returned 429 quota exhausted. Trying next provider...")
                else:
                    print(f"{BOLD_YELLOW}[Provider Fallback]:{RESET} {prov_name} error: {err_msg[:120]}. Trying next provider...")
                continue

        raise RuntimeError("All configured multimodal providers failed or are quota-exhausted.")

    def run(self, goal: str, max_steps: int = 25) -> bool:
        """Runs the continuous vision-action loop until task_complete or max_steps reached."""
        if not self._init_provider(goal):
            return False

        print(f"{BOLD_GREEN}[Provider Active]:{RESET} Running agentic loop on {self.active_provider_name}\n")

        if self.action_handler.is_completed:
            return True

        for step in range(2, max_steps + 1):
            try:
                self.action_handler.check_failsafe()
            except EmergencyStopException as es_exc:
                print(f"\n{BOLD_RED}[Emergency Stop]:{RESET} {es_exc}\n")
                return False

            time.sleep(self.step_delay)
            print(f"\n{BOLD_CYAN}─── [Agent Step {step}/{max_steps}] ───{RESET}")

            try:
                self.action_handler.check_failsafe()
            except EmergencyStopException as es_exc:
                print(f"\n{BOLD_RED}[Emergency Stop]:{RESET} {es_exc}\n")
                return False

            _, frame_bytes = self.lock.capture_secondary_frame()
            turn_msg = f"Current frame of Monitor 2 after previous action. Proceed toward goal: {goal}"

            try:
                resp_text = self.active_provider.send_turn(
                    turn_msg, frame_bytes, self.action_handler, max_sliding_frames=2
                )
                if resp_text and not self.action_handler.is_completed:
                    print(f"{BOLD_YELLOW}[Agent Reasoning]:{RESET} {resp_text.strip()}")
            except EmergencyStopException as es_exc:
                print(f"\n{BOLD_RED}[Emergency Stop]:{RESET} {es_exc}\n")
                return False
            except Exception as exc:
                print(f"{BOLD_RED}[Step Error]:{RESET} {exc}")
                time.sleep(1)

            if self.action_handler.is_completed:
                print(f"{BOLD_GREEN}[Success]:{RESET} Task completed in {step} steps!")
                return True

        print(f"\n{BOLD_YELLOW}[Limit Reached]:{RESET} Reached max steps ({max_steps}) without task_complete.")
        return True
