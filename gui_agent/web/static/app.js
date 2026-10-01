// ==============================================================================
// antiX GUI Operator Dashboard - Local Reactive Logic
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statusBadge = document.getElementById("agent-status-badge");
  const statusText = document.getElementById("status-text");
  const monitorBadge = document.getElementById("monitor-badge");
  const missionPrompt = document.getElementById("mission-prompt");
  const useArchitect = document.getElementById("use-architect");
  const btnPreviewArchitect = document.getElementById("btn-preview-architect");
  const architectPreviewBox = document.getElementById("architect-preview-box");
  const providerSelect = document.getElementById("provider-select");
  const delaySlider = document.getElementById("delay-slider");
  const delayVal = document.getElementById("delay-val");
  const stepsSlider = document.getElementById("steps-slider");
  const stepsVal = document.getElementById("steps-val");
  const btnRun = document.getElementById("btn-run");
  const btnStop = document.getElementById("btn-stop");
  const terminalOutput = document.getElementById("terminal-output");
  const terminalLinesCount = document.getElementById("terminal-lines-count");
  const btnClearTerminal = document.getElementById("btn-clear-terminal");
  const screenshotImg = document.getElementById("screenshot-img");
  const btnRefreshScreen = document.getElementById("btn-refresh-screen");
  const keysContainer = document.getElementById("keys-container");
  const btnSaveKeys = document.getElementById("btn-save-keys");
  const toast = document.getElementById("toast");

  let lineCount = 0;
  let isRunning = false;
  let keysData = {};

  // Sliders display update
  delaySlider.addEventListener("input", (e) => {
    delayVal.textContent = parseFloat(e.target.value).toFixed(1) + "s";
  });

  stepsSlider.addEventListener("input", (e) => {
    stepsVal.textContent = e.target.value;
  });

  // Toast helper
  function showToast(msg, isError = false) {
    toast.textContent = msg;
    toast.style.borderColor = isError ? "var(--accent-red)" : "var(--accent-cyan)";
    toast.style.display = "block";
    setTimeout(() => {
      toast.style.display = "none";
    }, 3000);
  }

  // Clear Terminal
  btnClearTerminal.addEventListener("click", () => {
    terminalOutput.textContent = "";
    lineCount = 0;
    terminalLinesCount.textContent = "0 lines";
  });

  // Refresh Screenshot
  function refreshScreenshot() {
    screenshotImg.src = "/api/screenshot?t=" + Date.now();
  }
  btnRefreshScreen.addEventListener("click", refreshScreenshot);
  setInterval(refreshScreenshot, 4000);

  // Preview Prompt Architect
  btnPreviewArchitect.addEventListener("click", async () => {
    const raw = missionPrompt.value.trim();
    if (!raw) {
      showToast("Please enter an objective first.", true);
      return;
    }
    btnPreviewArchitect.disabled = true;
    btnPreviewArchitect.textContent = "Refining...";
    try {
      const resp = await fetch("/api/architect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: raw }),
      });
      const data = await resp.json();
      if (resp.ok && data.refined_goal) {
        architectPreviewBox.textContent = data.refined_goal;
        architectPreviewBox.style.display = "block";
      } else {
        showToast(data.detail || "Architect error", true);
      }
    } catch (err) {
      showToast("Architect request failed: " + err, true);
    } finally {
      btnPreviewArchitect.disabled = false;
      btnPreviewArchitect.textContent = "Preview Refinement";
    }
  });

  // Append Log Line
  function appendLog(text) {
    if (terminalOutput.textContent.startsWith("Ready.")) {
      terminalOutput.textContent = "";
    }
    terminalOutput.textContent += text + "\n";
    lineCount++;
    terminalLinesCount.textContent = lineCount + " lines";
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
  }

  // SSE Log Streaming
  function initSSE() {
    const evtSource = new EventSource("/api/events");
    evtSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === "log" && payload.text) {
          appendLog(payload.text);
        }
      } catch (err) {
        // Heartbeat or raw line
      }
    };
    evtSource.onerror = () => {
      setTimeout(initSSE, 3000);
    };
  }
  initSSE();

  // Status Poller
  async function pollStatus() {
    try {
      const resp = await fetch("/api/status");
      if (!resp.ok) return;
      const data = await resp.json();

      isRunning = data.running;
      btnRun.disabled = isRunning;
      btnStop.disabled = !isRunning;

      if (data.secondary_monitor) {
        monitorBadge.textContent = data.secondary_monitor;
      }

      if (isRunning) {
        statusBadge.className = "status-badge running";
        statusText.textContent = `RUNNING (PID ${data.pid || "Active"})`;
      } else {
        statusBadge.className = "status-badge stopped";
        statusText.textContent = "IDLE / READY";
      }
    } catch (e) {
      // Offline / server rebooting
    }
  }
  setInterval(pollStatus, 2000);
  pollStatus();

  // Launch Mission
  btnRun.addEventListener("click", async () => {
    const goal = missionPrompt.value.trim();
    if (!goal) {
      showToast("Please enter a testing goal or instruction.", true);
      return;
    }

    btnRun.disabled = true;
    appendLog(`[Dashboard]: Initiating run with provider '${providerSelect.value}'...`);

    try {
      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal: goal,
          use_architect: useArchitect.checked,
          provider: providerSelect.value,
          max_steps: parseInt(stepsSlider.value),
          delay: parseFloat(delaySlider.value),
        }),
      });
      const data = await resp.json();
      if (resp.ok) {
        showToast("Mission started successfully.");
        pollStatus();
      } else {
        showToast(data.detail || "Launch failed", true);
        btnRun.disabled = false;
      }
    } catch (err) {
      showToast("Network error: " + err, true);
      btnRun.disabled = false;
    }
  });

  // Stop / Abort Agent
  btnStop.addEventListener("click", async () => {
    btnStop.disabled = true;
    appendLog("[Dashboard]: Sending stop signal to active agent...");
    try {
      const resp = await fetch("/api/stop", { method: "POST" });
      const data = await resp.json();
      if (resp.ok) {
        showToast("Agent process terminated.");
        pollStatus();
      } else {
        showToast(data.detail || "Stop failed", true);
      }
    } catch (err) {
      showToast("Stop error: " + err, true);
    }
  });

  // ----------------------------------------------------------------------------
  // API Key Management Center
  // ----------------------------------------------------------------------------
  const PROVIDER_NAMES = {
    agnes_keys: "Agnes AI API Keys",
    groq_keys: "Groq API Keys",
    gemini_keys: "Google Gemini API Keys",
    bytez_keys: "Bytez API Keys",
    nvidia_keys: "NVIDIA NIM API Keys",
    openrouter_keys: "OpenRouter API Keys",
    cerebras_keys: "Cerebras API Keys",
    mistral_keys: "Mistral API Keys",
  };

  async function loadKeys() {
    try {
      const resp = await fetch("/api/keys");
      if (!resp.ok) return;
      const data = await resp.json();
      keysData = data.keys || {};
      renderKeysAccordion(data.health || {});
    } catch (err) {
      console.error("Failed to load keys", err);
    }
  }

  function renderKeysAccordion(health) {
    keysContainer.innerHTML = "";

    for (const [providerProp, label] of Object.entries(PROVIDER_NAMES)) {
      const row = document.createElement("div");
      row.className = "provider-row";
      row.dataset.prop = providerProp;

      const keysList = keysData[providerProp] || [];
      const provShort = providerProp.replace("_keys", "");
      const isHealthy = (health[provShort] === "active") && keysList.length > 0;

      const header = document.createElement("div");
      header.className = "provider-row-header";
      header.innerHTML = `
        <div class="provider-title">
          <span>${label}</span>
          <span style="font-size: 11px; font-weight: normal; color: ${isHealthy ? "var(--accent-emerald)" : "var(--text-dim)"};">
            ${isHealthy ? "● Active (" + keysList.length + ")" : "○ Empty / Missing"}
          </span>
        </div>
        <button type="button" class="btn btn-secondary btn-sm btn-add-key" data-prop="${providerProp}">+ Add Key</button>
      `;

      const listContainer = document.createElement("div");
      listContainer.className = "provider-keys-list";
      listContainer.id = "list-" + providerProp;

      if (keysList.length === 0) {
        listContainer.appendChild(createKeyInputRow(providerProp, ""));
      } else {
        keysList.forEach((k) => {
          listContainer.appendChild(createKeyInputRow(providerProp, k));
        });
      }

      row.appendChild(header);
      row.appendChild(listContainer);
      keysContainer.appendChild(row);
    }

    // Attach Add Key listeners
    document.querySelectorAll(".btn-add-key").forEach((btn) => {
      btn.addEventListener("click", () => {
        const prop = btn.dataset.prop;
        const container = document.getElementById("list-" + prop);
        if (container) {
          container.appendChild(createKeyInputRow(prop, ""));
        }
      });
    });
  }

  function createKeyInputRow(prop, val) {
    const div = document.createElement("div");
    div.className = "key-input-row";

    const input = document.createElement("input");
    input.type = "text";
    input.value = val;
    input.placeholder = `Enter ${PROVIDER_NAMES[prop]}...`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-key";
    removeBtn.textContent = "✕";
    removeBtn.title = "Remove Key";
    removeBtn.addEventListener("click", () => {
      div.remove();
    });

    div.appendChild(input);
    div.appendChild(removeBtn);
    return div;
  }

  // Save Changes to config.json
  btnSaveKeys.addEventListener("click", async () => {
    btnSaveKeys.disabled = true;
    btnSaveKeys.textContent = "Saving...";

    const payload = {};
    for (const prop of Object.keys(PROVIDER_NAMES)) {
      const container = document.getElementById("list-" + prop);
      const inputs = container ? container.querySelectorAll("input") : [];
      const collected = [];
      inputs.forEach((inp) => {
        const v = inp.value.trim();
        if (v) collected.push(v);
      });
      payload[prop] = collected;
    }

    try {
      const resp = await fetch("/api/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (resp.ok) {
        showToast("Keys saved to local config.json!");
        loadKeys();
      } else {
        showToast(data.detail || "Failed to save keys", true);
      }
    } catch (err) {
      showToast("Save error: " + err, true);
    } finally {
      btnSaveKeys.disabled = false;
      btnSaveKeys.textContent = "Save Changes";
    }
  });

  loadKeys();
});
