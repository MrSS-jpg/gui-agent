import argparse
import sys
import pyautogui

from gui_agent.actions import (
    BOLD_CYAN,
    BOLD_GREEN,
    BOLD_RED,
    BOLD_YELLOW,
    RESET,
    EmergencyStopException,
)
from gui_agent.agent import VisionActionAgent
from gui_agent.display import DualMonitorLock
from gui_agent.key_extractor import extract_all_keys
from gui_agent.preflight import check_system_environment, print_preflight_report

BANNER = f"""{BOLD_CYAN}
╔═══════════════════════════════════════════════════════════════╗
║            antiX Linux X11 Autonomous GUI Operator            ║
║              Locked Strictly to Secondary Monitor             ║
║         Multi-Provider: Gemini / Groq / NIM / OpenRouter      ║
╚═══════════════════════════════════════════════════════════════╝{RESET}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous visual GUI operator locked strictly to Monitor 2 for UI/software testing."
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help="The testing goal or task description (e.g. 'Press Super, wait 1 second, press Super again').",
    )
    parser.add_argument(
        "--task",
        dest="task_flag",
        default=None,
        help="Alternative flag to specify testing goal.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "agnes", "groq", "bytez", "nvidia", "gemini", "openrouter"],
        help="Multimodal provider (default: auto with quota fallback).",
    )
    parser.add_argument(
        "--architect",
        action="store_true",
        help="Auto-refine raw goal into a deterministic step-by-step plan using Prompt Architect.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'agnes-2.5-flash', 'qwen/qwen3.6-27b', 'meta/llama-3.2-11b-vision-instruct').",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum vision-action steps before stopping (default: 25).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between actions to allow UI settling (default: 0.5s).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run pre-flight system diagnostics and check installed tools.",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        default=None,
        help="Path to ODT or text file containing keys (default: auto-discover or .env).",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Explicit name of the secondary monitor to lock onto (e.g. 'VGA-1').",
    )

    args = parser.parse_args()

    if args.check:
        is_ok = print_preflight_report(verbose=True)
        sys.exit(0 if is_ok else 1)

    task_goal = args.goal or args.task_flag
    if not task_goal:
        print(BANNER)
        parser.print_help()
        print(f"\n{BOLD_RED}[Error]: Please provide a testing goal or instruction.{RESET}")
        print(f"Example: run-agent \"Press Super, wait 1 second, press Super again\"\n")
        sys.exit(1)

    print(BANNER)

    # 0. Pre-Flight System Sanity
    print_preflight_report(verbose=False)

    # 1. API Key Extraction & Bootstrapping
    try:
        keystore = extract_all_keys(args.key_file)
        agnes_count = len(keystore.agnes_keys)
        or_count = len(keystore.openrouter_keys)
        nv_status = "Available" if keystore.get_nvidia_key() else "None"
        gem_count = len(keystore.gemini_keys)
        groq_count = len(keystore.groq_keys)
        bytez_count = len(keystore.bytez_keys)
        cer_count = len(keystore.cerebras_keys)
        mis_count = len(keystore.mistral_keys)

        print(f"{BOLD_GREEN}[Bootstrap]:{RESET} Loaded API keys from {keystore.source_file or 'local config'}:")
        print(f"             • Agnes AI:   {agnes_count} key(s)")
        print(f"             • Groq:       {groq_count} key(s)")
        print(f"             • Bytez:      {bytez_count} key(s)")
        print(f"             • NVIDIA NIM: {nv_status}")
        print(f"             • Gemini:     {gem_count} key(s) (exhausted ...U604 omitted)")
        print(f"             • OpenRouter: {or_count} key(s)")
        print(f"             • Cerebras:   {cer_count} key(s)")
        print(f"             • Mistral:    {mis_count} key(s)")
    except Exception as exc:
        print(f"{BOLD_RED}[Bootstrap Error]:{RESET} {exc}")
        sys.exit(1)

    # 2. Dual-Monitor Geometry Locking
    try:
        lock = DualMonitorLock(target_secondary=args.monitor)
        print(f"{BOLD_CYAN}[Monitor Locking]:{RESET} Primary Display:   {lock.primary}")
        print(
            f"{BOLD_CYAN}[Monitor Locking]:{RESET} Secondary Display: {lock.secondary} {BOLD_GREEN}[LOCKED]{RESET}"
        )
        print(
            f"{BOLD_CYAN}[Monitor Locking]:{RESET} Allowed Global Bounds: X=[{lock.secondary.offset_x}, {lock.secondary.offset_x + lock.secondary.width - 1}], "
            f"Y=[{lock.secondary.offset_y}, {lock.secondary.offset_y + lock.secondary.height - 1}]"
        )
        print(f"{BOLD_CYAN}[Safety Guard]:{RESET} Primary Display coordinates are strictly barred.")
        print(f"{BOLD_CYAN}[Safety Guard]:{RESET} PyAutoGUI fail-safe active (slam mouse to screen corner to abort).\n")
    except Exception as exc:
        print(f"{BOLD_RED}[Monitor Locking Error]:{RESET} {exc}")
        sys.exit(1)

    # 3. Optional Prompt Architect Translation
    if args.architect:
        try:
            from gui_agent.architect import PromptArchitect
            print(f"{BOLD_CYAN}[Prompt Architect]:{RESET} Translating raw request into deterministic execution plan...")
            pa = PromptArchitect(keystore)
            task_goal = pa.refine_prompt(task_goal)
        except Exception as exc:
            print(f"{BOLD_YELLOW}[Prompt Architect Fallback]:{RESET} Could not refine prompt: {exc}")

    # 4. Multi-Provider Vision-Action Agentic Loop
    try:
        print(f"{BOLD_YELLOW}[Task Objective]:{RESET} {task_goal}\n")
        agent = VisionActionAgent(
            keystore=keystore,
            monitor_lock=lock,
            preferred_provider=args.provider,
            preferred_model=args.model,
            step_delay=args.delay,
        )
        success = agent.run(goal=task_goal, max_steps=args.max_steps)
        sys.exit(0 if success else 2)

    except EmergencyStopException as exc:
        print(f"\n{BOLD_RED}[Emergency Killswitch]:{RESET} {exc}\n")
        sys.exit(130)

    except pyautogui.FailSafeException:
        print(f"\n{BOLD_RED}[Emergency Stop]:{RESET} Mouse pushed to corner! PyAutoGUI fail-safe triggered. Halting agent immediately.\n")
        sys.exit(130)

    except KeyboardInterrupt:
        print(f"\n{BOLD_YELLOW}[User Interrupt]:{RESET} Process terminated by user (Ctrl+C).\n")
        sys.exit(130)

    except Exception as exc:
        print(f"\n{BOLD_RED}[Runtime Error]:{RESET} {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
