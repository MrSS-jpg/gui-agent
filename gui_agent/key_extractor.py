import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

EXHAUSTED_KEY_SUFFIX = "U604"  # Known exhausted legacy Gemini key AIzaSy...U604
DEFAULT_CONFIG_PATH = os.environ.get(
    "GUI_AGENT_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
)


def load_env_file(dotenv_path: Optional[str] = None):
    """Simple, zero-dependency .env loader that populates os.environ if keys are not already set."""
    if not dotenv_path:
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.path.expanduser("~"),
        ]
        for d in search_dirs:
            p = os.path.join(d, ".env")
            if os.path.isfile(p):
                dotenv_path = p
                break

    if not dotenv_path or not os.path.isfile(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


@dataclass
class ExtractedKeyStore:
    agnes_keys: List[str] = field(default_factory=list)
    openrouter_keys: List[str] = field(default_factory=list)
    nvidia_key: Optional[str] = None
    nvidia_keys: List[str] = field(default_factory=list)
    cerebras_keys: List[str] = field(default_factory=list)
    mistral_keys: List[str] = field(default_factory=list)
    explorium_keys: List[str] = field(default_factory=list)
    gemini_keys: List[str] = field(default_factory=list)
    groq_keys: List[str] = field(default_factory=list)
    bytez_keys: List[str] = field(default_factory=list)
    source_file: str = ""

    def get_agnes_key(self) -> Optional[str]:
        """Returns Agnes AI key from environment or extracted keys."""
        env_key = os.environ.get("AGNES_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.agnes_keys[0] if self.agnes_keys else None

    def get_agnes_keys_pool(self) -> List[str]:
        """Returns all available Agnes AI keys."""
        pool: List[str] = []
        env_key = os.environ.get("AGNES_API_KEY")
        if env_key and env_key.strip():
            pool.append(env_key.strip())
        for k in self.agnes_keys:
            if k and k.strip() and k.strip() not in pool:
                pool.append(k.strip())
        return pool

    def get_openrouter_key(self) -> Optional[str]:
        """Returns OpenRouter key from environment or extracted keys."""
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.openrouter_keys[0] if self.openrouter_keys else None

    def get_nvidia_key(self) -> Optional[str]:
        """Returns NVIDIA NIM key from environment or extracted keys."""
        env_key = os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        if self.nvidia_keys:
            return self.nvidia_keys[0]
        return self.nvidia_key

    def get_groq_key(self) -> Optional[str]:
        """Returns Groq key from environment or extracted keys."""
        env_key = os.environ.get("GROQ_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.groq_keys[0] if self.groq_keys else None

    def get_bytez_key(self) -> Optional[str]:
        """Returns Bytez key from environment or extracted keys."""
        env_key = os.environ.get("BYTEZ_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.bytez_keys[0] if self.bytez_keys else None

    def get_bytez_keys_pool(self) -> List[str]:
        """Returns all available Bytez keys."""
        pool: List[str] = []
        env_key = os.environ.get("BYTEZ_API_KEY")
        if env_key and env_key.strip():
            pool.append(env_key.strip())
        for k in self.bytez_keys:
            if k and k.strip() and k.strip() not in pool:
                pool.append(k.strip())
        return pool

    def get_cerebras_key(self) -> Optional[str]:
        """Returns Cerebras key from environment or extracted keys."""
        env_key = os.environ.get("CEREBRAS_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.cerebras_keys[0] if self.cerebras_keys else None

    def get_mistral_key(self) -> Optional[str]:
        """Returns Mistral key from environment or extracted keys."""
        env_key = os.environ.get("MISTRAL_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return self.mistral_keys[0] if self.mistral_keys else None

    def get_gemini_keys_pool(self) -> List[str]:
        """
        Returns a list of all non-exhausted Gemini keys for rotation.
        Pools keys from environment (GEMINI_API_KEY, GOOGLE_API_KEY) and the .odt file,
        excluding known exhausted keys (...U604).
        """
        pool: List[str] = []
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.environ.get(env_var)
            if val and val.strip() and not val.endswith(EXHAUSTED_KEY_SUFFIX):
                if val.strip() not in pool:
                    pool.append(val.strip())

        for key in self.gemini_keys:
            if not key.endswith(EXHAUSTED_KEY_SUFFIX) and key not in pool:
                pool.append(key)

        return pool

    def get_gemini_key(self) -> Optional[str]:
        """
        Returns an active, non-exhausted Gemini API key.
        Checks pool of extracted and environment keys.
        """
        pool = self.get_gemini_keys_pool()
        return pool[0] if pool else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agnes_keys": self.agnes_keys,
            "groq_keys": self.groq_keys,
            "gemini_keys": self.gemini_keys,
            "bytez_keys": self.bytez_keys,
            "nvidia_keys": self.nvidia_keys or ([self.nvidia_key] if self.nvidia_key else []),
            "openrouter_keys": self.openrouter_keys,
            "cerebras_keys": self.cerebras_keys,
            "mistral_keys": self.mistral_keys,
            "explorium_keys": self.explorium_keys,
            "source_file": self.source_file,
        }

    def save_to_config(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Saves current keys to local JSON configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_config(self, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
        """Loads and overrides keys from config.json if present."""
        if not os.path.isfile(config_path):
            return False
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "agnes_keys" in data and isinstance(data["agnes_keys"], list):
                    self.agnes_keys = [k for k in data["agnes_keys"] if k]
                if "groq_keys" in data and isinstance(data["groq_keys"], list):
                    self.groq_keys = [k for k in data["groq_keys"] if k]
                if "gemini_keys" in data and isinstance(data["gemini_keys"], list):
                    self.gemini_keys = [k for k in data["gemini_keys"] if k]
                if "bytez_keys" in data and isinstance(data["bytez_keys"], list):
                    self.bytez_keys = [k for k in data["bytez_keys"] if k]
                if "nvidia_keys" in data and isinstance(data["nvidia_keys"], list):
                    self.nvidia_keys = [k for k in data["nvidia_keys"] if k]
                    if self.nvidia_keys:
                        self.nvidia_key = self.nvidia_keys[0]
                if "openrouter_keys" in data and isinstance(data["openrouter_keys"], list):
                    self.openrouter_keys = [k for k in data["openrouter_keys"] if k]
                if "cerebras_keys" in data and isinstance(data["cerebras_keys"], list):
                    self.cerebras_keys = [k for k in data["cerebras_keys"] if k]
                if "mistral_keys" in data and isinstance(data["mistral_keys"], list):
                    self.mistral_keys = [k for k in data["mistral_keys"] if k]
                return True
        except Exception:
            pass
        return False


def extract_all_keys(
    custom_path: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> ExtractedKeyStore:
    """
    Extracts active API keys for Agnes AI, Groq, Gemini, Bytez, NVIDIA NIM, OpenRouter,
    Cerebras, Mistral, and Explorium.
    Priority:
      1. Environment variables (.env file or system environment).
      2. Local config file (config.json) if present.
      3. ODT document key-store if provided or discovered.
    """
    load_env_file()

    home = os.path.expanduser("~")
    cwd = os.getcwd()
    env_key_file = os.environ.get("GUI_AGENT_KEY_FILE")

    candidate_paths = [
        custom_path,
        env_key_file,
        os.path.join(cwd, "keys.odt"),
        os.path.join(cwd, "untitled1.odt"),
        os.path.join(cwd, "Untitled 1.odt"),
        os.path.join(home, "Documents", "keys.odt"),
        os.path.join(home, "Documents", "untitled1.odt"),
        os.path.join(home, "Documents", "Untitled 1.odt"),
        os.path.join(home, "keys.odt"),
        os.path.join(home, "untitled1.odt"),
    ]

    odt_file = None
    for path in candidate_paths:
        if path and os.path.isfile(path):
            odt_file = path
            break

    keystore = ExtractedKeyStore(source_file=odt_file or "")

    if odt_file:
        try:
            with zipfile.ZipFile(odt_file, "r") as z:
                if "content.xml" in z.namelist():
                    xml_content = z.read("content.xml").decode("utf-8", errors="ignore")
                    _parse_xml_keys(xml_content, keystore)
        except Exception:
            pass

    # Support all standard provider environment variables (BYOK system)
    env_agnes = os.environ.get("AGNES_API_KEY")
    if env_agnes and env_agnes.strip() and env_agnes.strip() not in keystore.agnes_keys:
        keystore.agnes_keys.insert(0, env_agnes.strip())

    env_groq = os.environ.get("GROQ_API_KEY")
    if env_groq and env_groq.strip() and env_groq.strip() not in keystore.groq_keys:
        keystore.groq_keys.insert(0, env_groq.strip())

    env_gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_gemini and env_gemini.strip() and env_gemini.strip() not in keystore.gemini_keys:
        keystore.gemini_keys.insert(0, env_gemini.strip())

    env_bytez = os.environ.get("BYTEZ_API_KEY")
    if env_bytez and env_bytez.strip() and env_bytez.strip() not in keystore.bytez_keys:
        keystore.bytez_keys.insert(0, env_bytez.strip())

    env_nvidia = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if env_nvidia and env_nvidia.strip():
        keystore.nvidia_key = env_nvidia.strip()
        if env_nvidia.strip() not in keystore.nvidia_keys:
            keystore.nvidia_keys.insert(0, env_nvidia.strip())

    env_openrouter = os.environ.get("OPENROUTER_API_KEY")
    if env_openrouter and env_openrouter.strip() and env_openrouter.strip() not in keystore.openrouter_keys:
        keystore.openrouter_keys.insert(0, env_openrouter.strip())

    env_cerebras = os.environ.get("CEREBRAS_API_KEY")
    if env_cerebras and env_cerebras.strip() and env_cerebras.strip() not in keystore.cerebras_keys:
        keystore.cerebras_keys.insert(0, env_cerebras.strip())

    env_mistral = os.environ.get("MISTRAL_API_KEY")
    if env_mistral and env_mistral.strip() and env_mistral.strip() not in keystore.mistral_keys:
        keystore.mistral_keys.insert(0, env_mistral.strip())

    # Ensure single nvidia_key is in nvidia_keys list
    if keystore.nvidia_key and keystore.nvidia_key not in keystore.nvidia_keys:
        keystore.nvidia_keys.append(keystore.nvidia_key)

    # If config.json exists, load / merge keys
    if os.path.isfile(config_path):
        keystore.load_from_config(config_path)

    return keystore


def _parse_xml_keys(xml_content: str, keystore: ExtractedKeyStore):
    # Remove XML tags to get raw lines
    text_content = re.sub(r"<[^>]+>", "\n", xml_content)
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("sk-or-v1-"):
            if line not in keystore.openrouter_keys:
                keystore.openrouter_keys.append(line)
        elif line.startswith("nvapi-"):
            keystore.nvidia_key = line
            if line not in keystore.nvidia_keys:
                keystore.nvidia_keys.append(line)
        elif line.startswith("csk-"):
            if line not in keystore.cerebras_keys:
                keystore.cerebras_keys.append(line)
        elif line.startswith("gsk_"):
            if line not in keystore.groq_keys:
                keystore.groq_keys.append(line)
        elif line.startswith("bytez_"):
            if line not in keystore.bytez_keys:
                keystore.bytez_keys.append(line)
        elif line.startswith("AIzaSy"):
            if line not in keystore.gemini_keys:
                keystore.gemini_keys.append(line)
        elif re.match(r"^AQ\.Ab8RN[a-zA-Z0-9_-]+", line):
            if line not in keystore.gemini_keys:
                keystore.gemini_keys.append(line)
        elif line.startswith("sk-agnes-"):
            if line not in keystore.agnes_keys:
                keystore.agnes_keys.append(line)
        elif line.startswith("sk-") and not line.startswith("sk-or-v1-") and not line.startswith("sk-ant-"):
            # Standalone sk- key
            if line not in keystore.agnes_keys and len(line) >= 20:
                keystore.agnes_keys.append(line)

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if "agnes" in lower_line:
            for j in range(i + 1, min(i + 6, len(lines))):
                cand = lines[j]
                if (cand.startswith("sk-") or re.match(r"^[a-zA-Z0-9_-]{20,80}$", cand)) and not cand.startswith("gsk_"):
                    if cand not in keystore.agnes_keys:
                        keystore.agnes_keys.append(cand)
        elif "mistral api key" in lower_line:
            for j in range(i + 1, min(i + 6, len(lines))):
                candidate = lines[j]
                if re.match(r"^[a-zA-Z0-9]{32}$", candidate) and not candidate.startswith("gsk_"):
                    if candidate not in keystore.mistral_keys:
                        keystore.mistral_keys.append(candidate)
        elif "explorium api key" in lower_line:
            for j in range(i + 1, min(i + 6, len(lines))):
                candidate = lines[j]
                if re.match(r"^[a-f0-9-]{32,36}$", candidate) and not candidate.startswith("sk-"):
                    if candidate not in keystore.explorium_keys:
                        keystore.explorium_keys.append(candidate)
        elif "bytez api key" in lower_line or "bytez" in lower_line:
            adjacent_hex = []
            for j in range(i + 1, min(i + 5, len(lines))):
                cand = lines[j]
                if re.match(r"^[a-f0-9]{32,64}$", cand, re.I):
                    adjacent_hex.append(cand)
                    if cand not in keystore.bytez_keys:
                        keystore.bytez_keys.append(cand)
            if len(adjacent_hex) >= 2 and len(adjacent_hex[0]) == 32 and len(adjacent_hex[1]) == 32:
                joined = adjacent_hex[0] + adjacent_hex[1]
                if joined not in keystore.bytez_keys:
                    keystore.bytez_keys.insert(0, joined)


def extract_gemini_api_key(target_filename: str = "untitled1.odt") -> str:
    """
    Backwards-compatible helper that returns a non-exhausted Gemini API key.
    Raises ValueError if only the exhausted key is available.
    """
    keystore = extract_all_keys()
    key = keystore.get_gemini_key()
    if not key:
        raise ValueError("No active, non-exhausted Gemini API key found.")
    return key
