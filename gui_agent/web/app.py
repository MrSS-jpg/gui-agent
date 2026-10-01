import asyncio
import io
import json
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from gui_agent.actions import OSActionHandler
from gui_agent.architect import PromptArchitect
from gui_agent.display import DualMonitorLock
from gui_agent.key_extractor import DEFAULT_CONFIG_PATH, ExtractedKeyStore, extract_all_keys

app = FastAPI(title="antiX GUI Operator Control Panel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Shared Agent Process State
class AgentState:
    process: Optional[subprocess.Popen] = None
    goal: str = ""
    refined_goal: str = ""
    active_provider: str = "auto"
    max_steps: int = 25
    delay: float = 0.5
    current_step: int = 0
    is_running: bool = False
    log_history: List[str] = []
    log_subscribers: List[asyncio.Queue] = []

state = AgentState()


def broadcast_log(line: str):
    state.log_history.append(line)
    if len(state.log_history) > 1000:
        state.log_history.pop(0)
    for q in list(state.log_subscribers):
        try:
            q.put_nowait(line)
        except Exception:
            pass


class RunRequest(BaseModel):
    goal: str
    use_architect: bool = True
    provider: str = "auto"
    model: Optional[str] = None
    max_steps: int = 25
    delay: float = 0.5


class KeysUpdateRequest(BaseModel):
    agnes_keys: List[str] = []
    groq_keys: List[str] = []
    gemini_keys: List[str] = []
    bytez_keys: List[str] = []
    nvidia_keys: List[str] = []
    openrouter_keys: List[str] = []
    cerebras_keys: List[str] = []
    mistral_keys: List[str] = []


class ArchitectRequest(BaseModel):
    goal: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_file):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    target = os.path.join(STATIC_DIR, file_path)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Static file not found.")
    return FileResponse(target)


_cached_monitor_lock: Optional[DualMonitorLock] = None


def get_monitor_lock() -> DualMonitorLock:
    global _cached_monitor_lock
    if _cached_monitor_lock is None:
        _cached_monitor_lock = DualMonitorLock(target_secondary="VGA-1")
    return _cached_monitor_lock


@app.get("/api/status")
async def get_status():
    lock_info = "VGA-1 (1920x1080) [LOCKED]"
    try:
        lock = get_monitor_lock()
        lock_info = f"{lock.secondary.name} ({lock.secondary.width}x{lock.secondary.height}) [LOCKED]"
    except Exception as e:
        lock_info = f"VGA-1 (1920x1080) [LOCKED] ({e})"

    return {
        "running": state.is_running,
        "pid": state.process.pid if state.process and state.is_running else None,
        "goal": state.goal,
        "refined_goal": state.refined_goal,
        "active_provider": state.active_provider,
        "current_step": state.current_step,
        "max_steps": state.max_steps,
        "secondary_monitor": lock_info,
    }


@app.get("/api/screenshot")
async def get_screenshot():
    try:
        lock = get_monitor_lock()
        _, img_bytes = lock.capture_secondary_frame()
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception as exc:
        if os.path.exists("/dev/shm/agent_frame.jpg"):
            try:
                with open("/dev/shm/agent_frame.jpg", "rb") as f:
                    return Response(
                        content=f.read(),
                        media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
                    )
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to capture secondary monitor frame: {exc}")


@app.post("/api/architect")
async def refine_prompt(req: ArchitectRequest):
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Empty goal.")
    try:
        keystore = extract_all_keys()
        architect = PromptArchitect(keystore)
        refined = architect.refine_prompt(req.goal.strip())
        return {"refined_goal": refined}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run")
async def run_mission(req: RunRequest):
    if state.is_running:
        raise HTTPException(status_code=400, detail="Agent is already running. Stop it first.")

    target_goal = req.goal.strip()
    if not target_goal:
        raise HTTPException(status_code=400, detail="Mission goal cannot be empty.")

    state.goal = target_goal
    state.refined_goal = ""
    state.active_provider = req.provider
    state.max_steps = req.max_steps
    state.delay = req.delay
    state.current_step = 0
    state.log_history.clear()

    final_goal = target_goal
    if req.use_architect:
        broadcast_log("[Dashboard]: Refining mission prompt with Prompt Architect...")
        try:
            keystore = extract_all_keys()
            architect = PromptArchitect(keystore)
            final_goal = architect.refine_prompt(target_goal)
            state.refined_goal = final_goal
            broadcast_log("[Dashboard]: Refined Prompt:\n" + final_goal)
        except Exception as exc:
            broadcast_log(f"[Dashboard]: Architect fallback ({exc}), using raw goal.")

    # Launch run-agent CLI runner with current Python environment
    python_exe = sys.executable or "python3"
    cmd = [
        python_exe,
        "-m",
        "gui_agent.cli",
        "--provider",
        req.provider,
        "--max-steps",
        str(req.max_steps),
        "--delay",
        str(req.delay),
        final_goal,
    ]
    if req.model:
        cmd.extend(["--model", req.model])

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pkg_root}:{env.get('PYTHONPATH', '')}"
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0.0")

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,
        )
        state.process = p
        state.is_running = True

        asyncio.create_task(_monitor_process(p))
        return {"status": "started", "pid": p.pid}
    except Exception as exc:
        state.is_running = False
        raise HTTPException(status_code=500, detail=f"Failed to start agent process: {exc}")


async def _monitor_process(p: subprocess.Popen):
    loop = asyncio.get_event_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, p.stdout.readline)
            if not line:
                break
            line_str = line.rstrip()
            broadcast_log(line_str)
        p.wait()
    except Exception as e:
        broadcast_log(f"[Dashboard Error]: {e}")
    finally:
        state.is_running = False
        broadcast_log(f"[Dashboard]: Agent process exited with code {p.returncode}.")


@app.post("/api/stop")
async def stop_mission():
    # 1. Drop emergency stop sentinel file to trigger agent's internal killswitch immediately
    for p in ["/dev/shm/gui_agent_stop", "/tmp/gui_agent_stop"]:
        try:
            with open(p, "w") as f:
                f.write("STOP")
        except Exception:
            pass

    if not state.is_running or not state.process:
        return {"status": "not_running"}

    try:
        # Kill the entire process group
        os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
        broadcast_log("[Dashboard]: Sent SIGTERM to agent process group.")
    except Exception:
        try:
            state.process.terminate()
        except Exception:
            pass

    # Wait shortly and force kill if still alive
    await asyncio.sleep(0.4)
    if state.process and state.process.poll() is None:
        try:
            os.killpg(os.getpgid(state.process.pid), signal.SIGKILL)
            broadcast_log("[Dashboard]: Forced SIGKILL to agent process group.")
        except Exception:
            pass

    state.is_running = False
    return {"status": "stopped"}


@app.on_event("shutdown")
def on_shutdown():
    # Clean up any running agent processes when dashboard server is shut down
    for p in ["/dev/shm/gui_agent_stop", "/tmp/gui_agent_stop"]:
        try:
            with open(p, "w") as f:
                f.write("STOP")
        except Exception:
            pass

    if state.process and state.is_running:
        try:
            os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
        except Exception:
            try:
                state.process.terminate()
            except Exception:
                pass


@app.get("/api/events")
async def sse_events(request: Request):
    queue: asyncio.Queue = asyncio.Queue()
    state.log_subscribers.append(queue)

    async def event_generator():
        # Send backlog first
        for line in state.log_history:
            yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"
                except asyncio.TimeoutError:
                    yield f": heartbeat\n\n"
        finally:
            if queue in state.log_subscribers:
                state.log_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/keys")
async def get_keys():
    keystore = extract_all_keys()
    keys_data = keystore.to_dict()

    # Determine health heuristics
    health: Dict[str, str] = {}
    for prov, klist in [
        ("agnes", keystore.agnes_keys),
        ("groq", keystore.groq_keys),
        ("gemini", keystore.get_gemini_keys_pool()),
        ("bytez", keystore.bytez_keys),
        ("nvidia", keystore.nvidia_keys or ([keystore.nvidia_key] if keystore.nvidia_key else [])),
        ("openrouter", keystore.openrouter_keys),
        ("cerebras", keystore.cerebras_keys),
        ("mistral", keystore.mistral_keys),
    ]:
        if not klist:
            health[prov] = "missing"
        else:
            health[prov] = "active"

    return {
        "keys": keys_data,
        "health": health,
        "config_file": DEFAULT_CONFIG_PATH,
    }


@app.post("/api/keys")
async def update_keys(req: KeysUpdateRequest):
    keystore = extract_all_keys()
    keystore.agnes_keys = [k.strip() for k in req.agnes_keys if k.strip()]
    keystore.groq_keys = [k.strip() for k in req.groq_keys if k.strip()]
    keystore.gemini_keys = [k.strip() for k in req.gemini_keys if k.strip()]
    keystore.bytez_keys = [k.strip() for k in req.bytez_keys if k.strip()]
    keystore.nvidia_keys = [k.strip() for k in req.nvidia_keys if k.strip()]
    if keystore.nvidia_keys:
        keystore.nvidia_key = keystore.nvidia_keys[0]
    keystore.openrouter_keys = [k.strip() for k in req.openrouter_keys if k.strip()]
    keystore.cerebras_keys = [k.strip() for k in req.cerebras_keys if k.strip()]
    keystore.mistral_keys = [k.strip() for k in req.mistral_keys if k.strip()]

    try:
        keystore.save_to_config(DEFAULT_CONFIG_PATH)
        return {"status": "saved", "path": DEFAULT_CONFIG_PATH}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save keys: {exc}")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("gui_agent.web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
