# 🖥️ Autonomous Visual GUI Operator for Linux (X11)

> **An autonomous, multimodal vision-action desktop operator ("hands and eyes") designed for automated UI testing, software QA, and desktop control on Linux.**  
> Powered by an adaptive failover engine across Agnes AI, Groq, Bytez, NVIDIA NIM, Google Gemini, and OpenRouter with sub-second visual settling and pixel-perfect element snapping.

---

```
                       ┌──────────────────────────────────────────────┐
                       │          HUMAN GOAL / TEST MISSION           │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │      PROMPT ARCHITECT (Cerebras / Mistral)    │
                       │     Translates goal into atomic step plans   │
                       └──────────────────────┬───────────────────────┘
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         ADAPTIVE MULTIMODAL FAILOVER ENGINE                            │
 │  Agnes AI (2.5-Flash) ──► Groq ──► Bytez ──► NVIDIA NIM ──► Google Gemini ──► OpenRouter  │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │       ZERO-LATENCY RAM CAPTURE (/dev/shm tmpfs)         │
             │     scrot -a 1366,0,1920,1080 -q 80 (80% JPEG stream)   │
             └────────────────────────┬────────────────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────────────┐
             │         PRECISION SNAPPING & DOM REGISTRATION           │
             │   • Tesseract TSV OCR Bounding Box Centering            │
             │   • Adjacent Label Row Snapping for Checkboxes/Radios   │
             │   • xdotool mousemove --sync + 40ms Human Micro-Pause   │
             │   • Direct CLI Tool: run_command(bash)                  │
             └────────────────────────┬────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │     EMERGENCY KILLSWITCH     │
                       │  • Corner Slam (0,0)         │
                       │  • Sentinel File Check       │
                       │  • Dashboard Abort Button    │
                       └──────────────────────────────┘
```

---

## ⚡ Prerequisites

This agent controls X11 desktop windows and captures low-latency screenshots using standard Linux utilities.

On Ubuntu, Debian, antiX, Linux Mint, or Raspberry Pi OS:
```bash
sudo apt update && sudo apt install -y xdotool scrot tesseract-ocr x11-xserver-utils
```

*Optional but supported alternatives:*
- `maim` or ImageMagick (`import`) for fallback screen capture.

---

## 🚀 3-Step Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/gui-agent.git
cd gui-agent
```

### 2. Set Up Python Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

**Launch the Interactive Web Dashboard:**
```bash
./run_dashboard.sh
```
*Opens [http://127.0.0.1:8765](http://127.0.0.1:8765) in your default browser with live Monitor preview, streaming SSE logs, and interactive key manager.*

**Or Run via CLI:**
```bash
./run_agent.sh "Open terminal, check memory usage with free -m, and report"
```

---

## 🔑 Bring-Your-Own-Key (BYOK) System

The agent operates completely client-side without storing keys on any external server. You can provide API keys via any of three convenient methods:

### Method A: Environment File (`.env`)
Copy `.env.example` to `.env` in the project root:
```bash
cp .env.example .env
```
Populate your active keys:
```env
AGNES_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
BYTEZ_API_KEY=...
NVIDIA_API_KEY=nvapi-...
OPENROUTER_API_KEY=sk-or-v1-...
CEREBRAS_API_KEY=csk-...
MISTRAL_API_KEY=...
```

### Method B: Live Web UI Key Center
Navigate to [http://127.0.0.1:8765](http://127.0.0.1:8765) and use the **API Key & Provider Center** card to paste keys directly. Keys are saved locally to `config.json` (which is strictly gitignored).

### Method C: Environment Variables
Export variables directly in your shell or CI/CD runner:
```bash
export AGNES_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

---

## 🛡️ Safety & Emergency Killswitch

GUI automation requires strict safety boundaries to prevent unexpected behavior. The agent features **triple-redundant safety mechanisms**:

1. **Screen Corner Slam**: Slamming your physical mouse into any corner of the screen (e.g. `(0, 0)`) immediately triggers an `EmergencyStopException` and aborts all actions within 50ms.
2. **Dashboard Abort Button**: Clicking **"Abort / Stop Agent"** in the web panel instantly drops a stop sentinel file in `/dev/shm/gui_agent_stop` and sends `SIGTERM` / `SIGKILL` to the process group.
3. **Graceful Ctrl+C**: Hitting `Ctrl+C` in the terminal cleanly halts all subprocesses without leaving orphan background workers.
4. **Primary Monitor Barricade**: In dual-monitor setups, all cursor coordinates within the primary screen boundary (`LVDS-1` or `Display 1`) are strictly rejected with a `PermissionError`.

---

## 🛠️ CLI Options & Diagnostics

```bash
# Run pre-flight environment diagnostics
./run_agent.sh --check

# Run with a specific provider
./run_agent.sh --provider agnes "Navigate to docs.python.org"
./run_agent.sh --provider groq "Search antiX Linux release notes"

# Run with Prompt Architect goal translation
./run_agent.sh --architect "Test the user login form"

# Tune step delay and maximum steps
./run_agent.sh --delay 0.5 --max-steps 15 "Click the save icon"
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Free for open-source and commercial use.
