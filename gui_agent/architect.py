import os
import re
from typing import Optional
from openai import OpenAI

from gui_agent.actions import BOLD_CYAN, BOLD_GREEN, BOLD_YELLOW, BOLD_RED, RESET
from gui_agent.key_extractor import ExtractedKeyStore, extract_all_keys

ARCHITECT_SYSTEM_PROMPT = """You are the "Prompt Architect" middle-man for an autonomous visual GUI operator locked strictly to Monitor 2 (1920x1080) on antiX Linux (X11).
Your role: take brief, vague, or high-level human instructions (e.g. "Take test on screen 2", "Open chatgpt and say hi", "Search antix linux on firefox") and translate them into an explicit, deterministic, step-by-step execution prompt.

Every refined prompt MUST include:
1. Primary Goal & Visual Anchors: what windows, buttons, input fields, or icons to locate on Monitor 2.
2. Exact Keyboard Shortcuts:
   - 'Ctrl+T' for opening new browser tab
   - 'Ctrl+L' for focusing browser address bar
   - 'Enter' to submit forms / searches
   - 'Ctrl+W' to close tab
   - 'Super' to open application launcher
3. Deterministic Step-by-Step Execution Sequence: exact sequence of click, wait, and type actions.
4. Fallback Scrolling & Settling: instructions to scroll (PageDown or Down arrow) if content is below fold, and wait 1.0s for animations/page loading.
5. Completion Verification: clear visual assertion for when to call task_complete(summary).

Output ONLY the structured execution guide. Be direct, concise, and deterministic."""


class PromptArchitect:
    """
    High-speed text translation engine using Cerebras or Mistral (with Groq/OpenRouter fallback).
    Transforms vague human instructions into deterministic, structured sub-agent execution plans.
    """

    def __init__(self, keystore: Optional[ExtractedKeyStore] = None):
        self.keystore = keystore or extract_all_keys()

    def refine_prompt(self, user_goal: str) -> str:
        """
        Translates raw goal into a detailed, step-by-step vision operator guide.
        """
        user_prompt = (
            f"Human Request: \"{user_goal}\"\n\n"
            "Translate this request into an explicit, deterministic step-by-step visual execution plan "
            "for the Monitor 2 autonomous agent."
        )

        # 1. Attempt Cerebras (ultra-fast Llama-3.3-70b / Qwen)
        for key in self.keystore.cerebras_keys:
            if not key:
                continue
            for model_name in ["llama-3.3-70b", "qwen-3.8-27b", "gemma-4-31b", "gpt-oss-120b"]:
                try:
                    client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=key, timeout=10.0)
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.2,
                    )
                    content = resp.choices[0].message.content
                    if content and len(content.strip()) > 30:
                        print(f"{BOLD_GREEN}[Prompt Architect]:{RESET} Refined prompt via Cerebras ({model_name}).")
                        return content.strip()
                except Exception:
                    continue

        # 2. Attempt Mistral
        for key in self.keystore.mistral_keys:
            if not key:
                continue
            for model_name in ["mistral-small-latest", "mistral-medium-latest", "open-mistral-7b"]:
                try:
                    client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=key, timeout=10.0)
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.2,
                    )
                    content = resp.choices[0].message.content
                    if content and len(content.strip()) > 30:
                        print(f"{BOLD_GREEN}[Prompt Architect]:{RESET} Refined prompt via Mistral ({model_name}).")
                        return content.strip()
                except Exception:
                    continue

        # 3. Attempt Groq (Ultra-fast fallback)
        groq_key = self.keystore.get_groq_key()
        if groq_key:
            for model_name in ["qwen/qwen3.8-27b", "meta-llama/llama-3.2-11b-vision-preview"]:
                try:
                    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key, timeout=10.0)
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.2,
                    )
                    content = resp.choices[0].message.content
                    if content and len(content.strip()) > 30:
                        print(f"{BOLD_GREEN}[Prompt Architect]:{RESET} Refined prompt via Groq ({model_name}).")
                        return content.strip()
                except Exception:
                    continue

        # 4. Deterministic Rule-Based Fallback
        print(f"{BOLD_YELLOW}[Prompt Architect]:{RESET} API models busy; applying deterministic blueprint...")
        return self._heuristic_refine(user_goal)

    def _heuristic_refine(self, goal: str) -> str:
        g_lower = goal.lower()
        steps = []
        steps.append(f"Objective: {goal}")
        steps.append("Environment: antiX Linux X11, strictly locked to Monitor 2 (1920x1080).")
        steps.append("\nExecution Steps:")

        if any(w in g_lower for w in ["browser", "firefox", "chrome", "chatgpt", "youtube", "search", "web", "url"]):
            steps.append("1. Check if browser window is currently focused or visible on Monitor 2.")
            steps.append("2. If browser is open, press 'Ctrl+L' to focus the address bar, or 'Ctrl+T' for a new tab.")
            steps.append("3. If browser is not open, press 'Super', type 'firefox' or browser name, and press 'Enter'.")
            steps.append("4. Wait 1.5 seconds for UI settling.")
            steps.append(f"5. Enter the target URL or action for: '{goal}', then press 'Enter'.")
            steps.append("6. If content is below the fold, use 'PageDown' or arrow down to scroll.")
            steps.append("7. Once the page is loaded and target content is visually confirmed, call task_complete(summary).")
        elif "terminal" in g_lower or "bash" in g_lower:
            steps.append("1. Press 'Super' to open launcher, type 'terminal', and press 'Enter'.")
            steps.append("2. Wait 1.0s for terminal window to settle.")
            steps.append(f"3. Type the required command for '{goal}' and press 'Enter'.")
            steps.append("4. Confirm command execution in terminal output, then call task_complete(summary).")
        else:
            steps.append(f"1. Visually inspect Monitor 2 for relevant UI components related to: '{goal}'.")
            steps.append("2. Click on the target input element or button using precise pixel coordinates.")
            steps.append("3. Wait 1.0s for visual feedback and state update.")
            steps.append("4. Repeat action until objective is achieved, then call task_complete(summary).")

        return "\n".join(steps)
