"""
TTS Benchmark Studio — Streamlit App
=====================================
Run with:  streamlit run tts_streamlit_app.py

Features:
  1. Single-model generation
  2. Multi-model comparison
  3. Leaderboard system
  4. JSON-based model registry (./models/*.json)
  5. History browser with audio replay
  6. JSON import + visualization
  7. Model registration via JSON paste + auto-download
  8. LOCAL FILE REGISTRATION — scan local folders and register models
  9. MODEL INSPECTOR — view full JSON config, metadata, args, and
     per-sentence benchmark metrics for every registered model
 10. ENHANCED LOCAL FILES TAB — guided workflow, file verification,
     download support, auto-scan after extraction
"""

import os
import re
import json
import time
import subprocess
import tarfile
import zipfile
import threading

import psutil
import librosa
import jiwer
import soundfile as sf
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from faster_whisper import WhisperModel

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ─────────────────────────── PAGE CONFIG ──────────────────────
st.set_page_config(
    page_title="TTS Benchmark Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── PATHS ────────────────────────────
BASE_DIR       = Path(__file__).parent
ASR_MODEL_PATH = str(BASE_DIR / "faster-whisper-small")
OUTPUT_DIR     = BASE_DIR / "gradio_wav_outputs"
JSON_DIR       = BASE_DIR / "gradio_json_outputs"
MODELS_DIR     = BASE_DIR / "models"
ASSETS_DIR     = BASE_DIR / "downloaded_assets"
LOCAL_SCAN_DIR = BASE_DIR
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# ─────────────────────────── SCHEMA CONSTANTS ─────────────────
VALID_TYPES     = ["cli", "supertonic", "pocket", "kokoro", "mms", "indic_parler", "dynamic_sherpa"]
VALID_LANGS     = ["en", "hi", "mar"]
REQUIRED_FIELDS = ["display_name", "type", "langs"]
REQUIRED_META   = ["provider", "parameters", "compute_type", "model_size", "license", "best_use_case"]

LANG_LABEL    = {"en": "🇬🇧 English", "hi": "🇮🇳 Hindi", "mar": "🇮🇳 Marathi"}
WER_TRANSFORM = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip()]
)

CHUNK_SIZE               = 65536
DOWNLOAD_TIMEOUT_CONNECT = 15
DOWNLOAD_TIMEOUT_READ    = 180
# ─────────────────────────── LOCAL MODEL SCANNER ──────────────

# ONNX files that are sub-components, NOT standalone TTS models
_SUBCOMPONENT_KEYWORDS = [
    "vocos", "vocoder",
    "duration_predictor", "duration",
    "text_encoder",
    "vector_estimator",
    "lm_flow", "lm_main",
    "text_conditioner",
    "discriminator",
    "embedding",
]

# Must match at least one of these to be a TTS model
_ONNX_EXCLUDE_PATTERNS = [
    "vocos", "vocoder", "encoder", "decoder", "duration",
    "text_encoder", "vector_estimator", "lm_flow", "lm_main",
    "text_conditioner", "embedding", "discriminator",
]

_ONNX_TTS_PATTERNS = [
    "vits", "kokoro", "kitten", "matcha", "model",
    "tts", "piper", "amy", "lessac", "ryan", "alan",
    "jenny", "pratham", "priya", "rohan", "ljspeech",
    "hi_in", "en_us", "en_gb", "mar", "hindi",
]


def _is_tts_model_onnx(onnx_path: Path) -> bool:
    """
    Returns True only if this ONNX file looks like a top-level
    TTS model (not a vocoder or sub-component).
    """
    stem = onnx_path.stem.lower()
    for pat in _ONNX_EXCLUDE_PATTERNS:
        if pat in stem:
            return False
    for pat in _ONNX_TTS_PATTERNS:
        if pat in stem:
            return True
    parent_name = onnx_path.parent.name.lower()
    if re.match(r"^model", stem):
        for pat in _ONNX_TTS_PATTERNS:
            if pat in parent_name:
                return True
    return False


def _model_is_multispeaker(onnx_path: Path) -> bool:
    """
    Returns True if the model supports multiple speakers.
    Defaults to False (single-speaker) to avoid the
    'n_speakers does not exist' error.
    """
    parent = onnx_path.parent
    stem   = onnx_path.stem.lower()

    # Check companion JSON configs
    for search_root in [parent, parent.parent]:
        for jf in search_root.glob("*.json"):
            try:
                if jf.stat().st_size > 500_000:
                    continue
                data  = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
                n_spk = (data.get("n_speakers")
                         or data.get("num_speakers")
                         or data.get("speakers"))
                if isinstance(n_spk, int) and n_spk > 1:
                    return True
                if isinstance(n_spk, (list, dict)) and len(n_spk) > 1:
                    return True
                if "speaker_id_map" in data and len(data["speaker_id_map"]) > 1:
                    return True
                if "speaker2id" in data and len(data["speaker2id"]) > 1:
                    return True
            except Exception:
                continue

    # voices.bin = strong multi-speaker indicator
    for search_root in [parent, parent.parent]:
        if list(search_root.rglob("voices.bin")):
            return True

    # lexicon.txt = vits_lexicon = multi-speaker
    for search_root in [parent, parent.parent]:
        if list(search_root.rglob("lexicon.txt")):
            return True

    # Filename/folder heuristics
    parent_name  = parent.name.lower()
    always_multi = ["kokoro", "kitten"]
    multi_hints  = ["vctk", "multi", "multispeaker", "multi_speaker"]
    for hint in always_multi + multi_hints:
        if hint in stem or hint in parent_name:
            return True

    return False  # safe default



# ─────────────────────────── LOCAL MODEL SCANNER ──────────────
def _find_companion_files(model_onnx: Path) -> dict:
    parent     = model_onnx.parent
    companions = {}
    for search_root in [parent, parent.parent]:
        found = list(search_root.rglob("tokens.txt"))
        if found:
            companions["tokens"] = str(found[0])
            break
    for search_root in [parent, parent.parent]:
        found = list(search_root.rglob("lexicon.txt"))
        if found:
            companions["lexicon"] = str(found[0])
            break
    for search_root in [parent, parent.parent, parent.parent.parent]:
        found = list(search_root.rglob("espeak-ng-data"))
        if found and found[0].is_dir():
            companions["data_dir"] = str(found[0])
            break
    for search_root in [parent, parent.parent]:
        found = list(search_root.rglob("voices.bin"))
        if found:
            companions["voices"] = str(found[0])
            break
    return companions


def _build_cli_args_from_scan(
    model_onnx: Path,
    model_type: str,
    sid: int = 0,
    force_multispeaker: bool = None,
) -> list:
    """
    Build sherpa-onnx CLI args. Only adds --sid when the model
    is confirmed multi-speaker, preventing 'n_speakers' errors.
    """
    companions = _find_companion_files(model_onnx)
    args       = []
    model_str  = str(model_onnx)
    is_multi   = (force_multispeaker if force_multispeaker is not None
                  else _model_is_multispeaker(model_onnx))

    if model_type == "vits":
        args += ["--vits-model", model_str]
        if "tokens"   in companions: args += ["--vits-tokens",   companions["tokens"]]
        if "data_dir" in companions: args += ["--vits-data-dir", companions["data_dir"]]
        if is_multi:
            args += ["--sid", str(sid)]

    elif model_type == "vits_lexicon":
        args += ["--vits-model", model_str]
        if "tokens"  in companions: args += ["--vits-tokens",  companions["tokens"]]
        if "lexicon" in companions: args += ["--vits-lexicon", companions["lexicon"]]
        args += ["--sid", str(sid)]   # always multi-speaker

    elif model_type == "kokoro_onnx":
        args += ["--kokoro-model", model_str]
        if "voices"   in companions: args += ["--kokoro-voices",   companions["voices"]]
        if "tokens"   in companions: args += ["--kokoro-tokens",   companions["tokens"]]
        if "data_dir" in companions: args += ["--kokoro-data-dir", companions["data_dir"]]
        args += ["--sid", str(sid)]   # always multi-speaker

    elif model_type == "kitten":
        args += ["--kitten-model", model_str]
        if "voices"   in companions: args += ["--kitten-voices",   companions["voices"]]
        if "tokens"   in companions: args += ["--kitten-tokens",   companions["tokens"]]
        if "data_dir" in companions: args += ["--kitten-data-dir", companions["data_dir"]]
        args += ["--sid", str(sid)]   # always multi-speaker

    elif model_type == "matcha":
        args += ["--matcha-acoustic-model", model_str]
        vocos = list(BASE_DIR.rglob("vocos*.onnx"))
        if vocos: args += ["--matcha-vocoder", str(vocos[0])]
        if "tokens"   in companions: args += ["--matcha-tokens",   companions["tokens"]]
        if "data_dir" in companions: args += ["--matcha-data-dir", companions["data_dir"]]
        # Matcha is single-speaker — no --sid

    return args

def scan_local_models(scan_root: Path) -> list:
    """
    Scan for ONNX TTS model files. Excludes vocoders and
    sub-components that cause 'n_speakers'/'sample_rate' errors.
    """
    candidates = []
    seen_onnx  = set()
    skip_dirs  = {
        "faster-whisper-small", "gradio_wav_outputs", "gradio_json_outputs",
        "__pycache__", ".git", "downloaded_assets", "node_modules",
        ".venv", "venv",
    }

    for onnx_path in sorted(scan_root.rglob("*.onnx")):
        if set(onnx_path.parts) & skip_dirs:
            continue
        try:
            if onnx_path.stat().st_size < 1_000_000:
                continue
        except OSError:
            continue

        # Skip vocoders and sub-components
        if not _is_tts_model_onnx(onnx_path):
            continue

        onnx_str = str(onnx_path)
        if onnx_str in seen_onnx:
            continue
        seen_onnx.add(onnx_str)

        fname      = onnx_path.name.lower()
        stem       = onnx_path.stem.lower()
        parent_str = str(onnx_path.parent).lower()
        model_type = "vits"
        lang       = "en"
        provider   = "UNKNOWN"

        if "kokoro" in stem or "kokoro" in parent_str:
            model_type, provider = "kokoro_onnx", "K2-FSA/SHERPA-ONNX"
        elif "kitten" in stem or "kitten" in parent_str:
            model_type, provider = "kitten", "CUSTOM"
        elif "matcha" in stem or "matcha" in parent_str:
            model_type, provider = "matcha", "ICEFALL"
        elif "vctk" in stem or "lexicon" in parent_str:
            model_type, provider = "vits_lexicon", "K2-FSA"

        if any(x in stem or x in parent_str
               for x in ["hi_in", "hindi", "pratham", "priya", "rohan"]):
            lang, provider = "hi", "K2-FSA/PIPER"
        elif "mar" in stem or "marathi" in parent_str:
            lang, provider = "mar", "K2-FSA/PIPER"
        elif any(x in stem for x in ["en_us","en_gb","amy","lessac","ryan","alan","jenny"]):
            lang = "en"
            if provider == "UNKNOWN":
                provider = "COQUI/PIPER"

        if provider == "UNKNOWN":
            provider = "SHERPA-ONNX"

        dir_name     = onnx_path.parent.name
        display_name = (
            dir_name.replace("-", " ").replace("_", " ").title()
            + f" ({onnx_path.stem})"
        )
        if len(display_name) > 60:
            display_name = display_name[:57] + "…"

        args = _build_cli_args_from_scan(onnx_path, model_type, sid=0)

        try:
            size_mb  = onnx_path.stat().st_size / (1024 * 1024)
            size_str = f"~{size_mb:.0f}MB"
        except OSError:
            size_str = "unknown"

        candidates.append({
            "display_name":  display_name,
            "type":          "cli",
            "langs":         [lang],
            "args":          args,
            "installed":     True,
            "downloads":     [],
            "sherpa_config": {},
            "meta": {
                "provider":      provider,
                "parameters":    "unknown",
                "compute_type":  "ONNX",
                "model_size":    size_str,
                "license":       "unknown",
                "best_use_case": f"{lang.upper()} TTS",
            },
            "_onnx_path": onnx_str,
        })

    return candidates

def extract_local_archive(archive_path: Path, dest_dir: Path, status_ph) -> bool:
    """Extract a local archive file to a destination directory with safety checks."""
    fname = archive_path.name.lower()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        status_ph.info(f"📦 Extracting `{archive_path.name}` → `{dest_dir}`…")
        if fname.endswith((".tar.gz", ".tgz")):
            with tarfile.open(str(archive_path), "r:gz") as tar:
                safe = [m for m in tar.getmembers() if not re.search(r"(^/|\.\.)", m.name)]
                tar.extractall(str(dest_dir), members=safe)
        elif fname.endswith(".tar.bz2"):
            with tarfile.open(str(archive_path), "r:bz2") as tar:
                safe = [m for m in tar.getmembers() if not re.search(r"(^/|\.\.)", m.name)]
                tar.extractall(str(dest_dir), members=safe)
        elif fname.endswith(".tar.xz"):
            with tarfile.open(str(archive_path), "r:xz") as tar:
                safe = [m for m in tar.getmembers() if not re.search(r"(^/|\.\.)", m.name)]
                tar.extractall(str(dest_dir), members=safe)
        elif fname.endswith(".tar"):
            with tarfile.open(str(archive_path), "r:") as tar:
                safe = [m for m in tar.getmembers() if not re.search(r"(^/|\.\.)", m.name)]
                tar.extractall(str(dest_dir), members=safe)
        elif fname.endswith(".zip"):
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(str(dest_dir))
        else:
            status_ph.warning(f"⚠️ Unknown archive format: `{archive_path.name}`")
            return False
        status_ph.success(f"✅ Extracted `{archive_path.name}` successfully!")
        return True
    except Exception as e:
        status_ph.error(f"❌ Extraction failed: {e}")
        return False


# ─────────────────────────── DOWNLOAD HELPER ──────────────────
def download_file(url: str, dest_path: Path, progress_ph=None) -> bool:
    """
    Download a file from a URL with progress reporting.
    Returns True on success, False on failure.
    """
    if not REQUESTS_AVAILABLE:
        if progress_ph:
            progress_ph.error("❌ `requests` library not available. Cannot download.")
        return False
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(
            url,
            stream=True,
            timeout=(DOWNLOAD_TIMEOUT_CONNECT, DOWNLOAD_TIMEOUT_READ),
        )
        resp.raise_for_status()
        total    = int(resp.headers.get("content-length", 0))
        received = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
                    if progress_ph and total > 0:
                        pct  = min(received / total, 1.0)
                        done = received / (1024 * 1024)
                        tot  = total    / (1024 * 1024)
                        progress_ph.progress(pct, text=f"⬇️ {done:.1f} / {tot:.1f} MB")
        if progress_ph:
            progress_ph.success(f"✅ Downloaded `{dest_path.name}` ({received/(1024*1024):.1f} MB)")
        return True
    except Exception as e:
        if progress_ph:
            progress_ph.error(f"❌ Download failed: {e}")
        return False


# ─────────────────────────── HARDCODED REGISTRY ───────────────
def _hardcoded_registry() -> dict:
    return {
        "VITS (Amy - English)": {
            "type": "cli", "langs": ["en"],
            "args": ["--vits-model", "./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
                     "--vits-tokens", "./vits-piper-en_US-amy-low/tokens.txt",
                     "--vits-data-dir", "./vits-piper-en_US-amy-low/espeak-ng-data"],
            "meta": {"provider": "COQUI/PIPER", "parameters": "~40M",
                     "compute_type": "FP32/ONNX", "model_size": "100MB",
                     "license": "MIT", "best_use_case": "GENERAL TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "Matcha-TTS (LJSpeech)": {
            "type": "cli", "langs": ["en"],
            "args": ["--matcha-acoustic-model", "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
                     "--matcha-vocoder", "./vocos-22khz-univ.onnx",
                     "--matcha-tokens", "./matcha-icefall-en_US-ljspeech/tokens.txt",
                     "--matcha-data-dir", "./matcha-icefall-en_US-ljspeech/espeak-ng-data"],
            "meta": {"provider": "ICEFALL", "parameters": "~60M",
                     "compute_type": "ONNX", "model_size": "150MB",
                     "license": "Apache 2.0", "best_use_case": "FAST TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "Kitten (Nano)": {
            "type": "cli", "langs": ["en"],
            "args": ["--kitten-model", "./kitten-nano-en-v0_1-fp16/model.fp16.onnx",
                     "--kitten-voices", "./kitten-nano-en-v0_1-fp16/voices.bin",
                     "--kitten-tokens", "./kitten-nano-en-v0_1-fp16/tokens.txt",
                     "--kitten-data-dir", "./kitten-nano-en-v0_1-fp16/espeak-ng-data",
                     "--sid", "0"],
            "meta": {"provider": "CUSTOM", "parameters": "~20M",
                     "compute_type": "FP16", "model_size": "50MB",
                     "license": "MIT", "best_use_case": "EDGE DEVICES"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "VITS-VCTK Speaker 0 (Female British)": {
            "type": "cli", "langs": ["en"],
            "args": ["--vits-model", "./vits-vctk/vits-vctk.int8.onnx",
                     "--vits-tokens", "./vits-vctk/tokens.txt",
                     "--vits-lexicon", "./vits-vctk/lexicon.txt", "--sid", "0"],
            "meta": {"provider": "K2-FSA", "parameters": "~40M",
                     "compute_type": "INT8/ONNX", "model_size": "37MB",
                     "license": "MIT", "best_use_case": "FEMALE BRITISH TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "VITS-VCTK Speaker 10 (Male Scottish)": {
            "type": "cli", "langs": ["en"],
            "args": ["--vits-model", "./vits-vctk/vits-vctk.int8.onnx",
                     "--vits-tokens", "./vits-vctk/tokens.txt",
                     "--vits-lexicon", "./vits-vctk/lexicon.txt", "--sid", "10"],
            "meta": {"provider": "K2-FSA", "parameters": "~40M",
                     "compute_type": "INT8/ONNX", "model_size": "37MB",
                     "license": "MIT", "best_use_case": "MALE SCOTTISH TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "VITS-VCTK Speaker 60 (Male American)": {
            "type": "cli", "langs": ["en"],
            "args": ["--vits-model", "./vits-vctk/vits-vctk.int8.onnx",
                     "--vits-tokens", "./vits-vctk/tokens.txt",
                     "--vits-lexicon", "./vits-vctk/lexicon.txt", "--sid", "60"],
            "meta": {"provider": "K2-FSA", "parameters": "~40M",
                     "compute_type": "INT8/ONNX", "model_size": "37MB",
                     "license": "MIT", "best_use_case": "MALE AMERICAN TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "Kokoro v0.19 (American Female)": {
            "type": "cli", "langs": ["en"],
            "args": ["--kokoro-model", "./kokoro-en-v0_19/model.onnx",
                     "--kokoro-voices", "./kokoro-en-v0_19/voices.bin",
                     "--kokoro-tokens", "./kokoro-en-v0_19/tokens.txt",
                     "--kokoro-data-dir", "./kokoro-en-v0_19/espeak-ng-data", "--sid", "2"],
            "meta": {"provider": "K2-FSA/SHERPA-ONNX", "parameters": "~82M",
                     "compute_type": "FP32/ONNX", "model_size": "330MB",
                     "license": "Apache 2.0", "best_use_case": "AMERICAN FEMALE TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "Kokoro v0.19 (British Male)": {
            "type": "cli", "langs": ["en"],
            "args": ["--kokoro-model", "./kokoro-en-v0_19/model.onnx",
                     "--kokoro-voices", "./kokoro-en-v0_19/voices.bin",
                     "--kokoro-tokens", "./kokoro-en-v0_19/tokens.txt",
                     "--kokoro-data-dir", "./kokoro-en-v0_19/espeak-ng-data", "--sid", "10"],
            "meta": {"provider": "K2-FSA/SHERPA-ONNX", "parameters": "~82M",
                     "compute_type": "FP32/ONNX", "model_size": "330MB",
                     "license": "Apache 2.0", "best_use_case": "BRITISH MALE TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "Supertonic (English)": {
            "type": "supertonic", "langs": ["en"], "args": [],
            "meta": {"provider": "SHERPA-ONNX", "parameters": "~80M",
                     "compute_type": "INT8", "model_size": "200MB",
                     "license": "Apache 2.0", "best_use_case": "PRODUCTION TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "PocketTTS (Voice Cloning)": {
            "type": "pocket", "langs": ["en"], "args": [],
            "meta": {"provider": "SHERPA-ONNX", "parameters": "~100M",
                     "compute_type": "INT8", "model_size": "300MB",
                     "license": "Apache 2.0", "best_use_case": "VOICE CLONING"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "Kokoro (PyTorch - Emotional)": {
            "type": "kokoro", "langs": ["en"], "args": [],
            "meta": {"provider": "HEXAGRAD", "parameters": "82M",
                     "compute_type": "FP16", "model_size": "300MB",
                     "license": "Apache 2.0", "best_use_case": "EMOTIONAL TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "VITS Hindi (Generic)": {
            "type": "cli", "langs": ["hi"],
            "args": ["--vits-model", "./vits-hindi/model.onnx",
                     "--vits-tokens", "./vits-hindi/tokens.txt",
                     "--vits-data-dir", "./vits-hindi/espeak-ng-data"],
            "meta": {"provider": "MMS", "parameters": "~50M",
                     "compute_type": "FP32", "model_size": "120MB",
                     "license": "CC-BY-NC 4.0", "best_use_case": "HINDI TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "VITS Piper Hindi - Pratham (Male)": {
            "type": "cli", "langs": ["hi"],
            "args": ["--vits-model", "./vits-piper-hi_IN-pratham-medium/hi_IN-pratham-medium.onnx",
                     "--vits-tokens", "./vits-piper-hi_IN-pratham-medium/tokens.txt",
                     "--vits-data-dir", "./vits-piper-hi_IN-pratham-medium/espeak-ng-data"],
            "meta": {"provider": "K2-FSA/PIPER", "parameters": "~40M",
                     "compute_type": "FP32/ONNX", "model_size": "~100MB",
                     "license": "MIT", "best_use_case": "MALE HINDI TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-pratham-medium.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "VITS Piper Hindi - Priyamvada (Female)": {
            "type": "cli", "langs": ["hi"],
            "args": ["--vits-model", "./vits-piper-hi_IN-priyamvada-medium/hi_IN-priyamvada-medium.onnx",
                     "--vits-tokens", "./vits-piper-hi_IN-priyamvada-medium/tokens.txt",
                     "--vits-data-dir", "./vits-piper-hi_IN-priyamvada-medium/espeak-ng-data"],
            "meta": {"provider": "K2-FSA/PIPER", "parameters": "~40M",
                     "compute_type": "FP32/ONNX", "model_size": "~100MB",
                     "license": "MIT", "best_use_case": "FEMALE HINDI TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "VITS Piper Hindi - Rohan (Male)": {
            "type": "cli", "langs": ["hi"],
            "args": ["--vits-model", "./vits-piper-hi_IN-rohan-medium/hi_IN-rohan-medium.onnx",
                     "--vits-tokens", "./vits-piper-hi_IN-rohan-medium/tokens.txt",
                     "--vits-data-dir", "./vits-piper-hi_IN-rohan-medium/espeak-ng-data"],
            "meta": {"provider": "K2-FSA/PIPER", "parameters": "~40M",
                     "compute_type": "FP32/ONNX", "model_size": "~100MB",
                     "license": "MIT", "best_use_case": "MALE HINDI TTS"},
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2"
            ],
            "installed": True, "sherpa_config": {},
        },
        "MMS (English + Hindi)": {
            "type": "mms", "langs": ["en", "hi"], "args": [],
            "meta": {"provider": "META", "parameters": "~50M",
                     "compute_type": "FP16", "model_size": "150MB",
                     "license": "CC-BY-NC 4.0", "best_use_case": "MULTILINGUAL TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "MMS Marathi": {
            "type": "mms", "langs": ["mar"], "args": [],
            "meta": {"provider": "META", "parameters": "~50M",
                     "compute_type": "FP32", "model_size": "~150MB",
                     "license": "CC-BY-NC 4.0", "best_use_case": "MARATHI TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
        "Indic Parler TTS (Marathi)": {
            "type": "indic_parler", "langs": ["mar", "hi", "en"], "args": [],
            "meta": {"provider": "AI4BHARAT", "parameters": "~400M",
                     "compute_type": "FP32", "model_size": "3.75GB",
                     "license": "Apache 2.0", "best_use_case": "EXPRESSIVE INDIC TTS"},
            "downloads": [], "installed": True, "sherpa_config": {},
        },
    }


def _is_bad_model(name: str, cfg: dict) -> tuple[bool, str]:
    """
    Returns (is_bad: bool, reason: str).
    Called during registry load to exclude unusable models
    before they ever reach the UI.
    """
    BAD_KW = [
        "decoder", "encoder", "lm_flow", "lm_main",
        "text_conditioner", "text_encoder", "vector_estimator",
        "duration_predictor", "duration", "vocos", "vocoder",
        "discriminator", "embedding",
    ]
    UNKNOWN_V = {"", "unknown", "UNKNOWN", "—", "?", "none", "None"}

    name_lower = name.lower()
    args_lower = " ".join(str(a) for a in cfg.get("args", [])).lower()

    # Rule 1: sub-component keyword in name or args path
    for kw in BAD_KW:
        if kw in name_lower or kw in args_lower:
            return True, f"sub-component keyword '{kw}'"

    # Rule 2: 2+ critical metadata fields are unknown
    meta     = cfg.get("meta", {})
    critical = ["provider", "compute_type", "license"]
    unknown_critical = [
        f for f in critical
        if str(meta.get(f, "")).strip().lower() in UNKNOWN_V
        or meta.get(f) is None
    ]
    if len(unknown_critical) >= 2:
        return True, f"unknown critical metadata: {', '.join(unknown_critical)}"

    return False, ""


def load_model_registry() -> dict:
    """
    Load model registry from hardcoded defaults + JSON files.
    Bad models (sub-components, unknown metadata) are excluded
    at load time and never reach the UI.
    """
    registry = _hardcoded_registry()

    for jf in sorted(MODELS_DIR.glob("*.json")):
        try:
            raw     = json.loads(jf.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else [raw]
            for e in entries:
                name = e.get("display_name", jf.stem)
                cfg  = {
                    "type":          e.get("type",          "cli"),
                    "langs":         e.get("langs",         ["en"]),
                    "args":          e.get("args",          []),
                    "meta":          e.get("meta",          {}),
                    "downloads":     e.get("downloads",     []),
                    "installed":     e.get("installed",     False),
                    "sherpa_config": e.get("sherpa_config", {}),
                }

                # ── Filter bad models at load time ─────────────
                bad, reason = _is_bad_model(name, cfg)
                if bad:
                    # Also delete the JSON file so it never comes back
                    try:
                        jf.unlink()
                    except Exception:
                        pass
                    continue  # skip — do not add to registry

                registry[name] = cfg

        except Exception as ex:
            st.warning(f"⚠️ Could not load {jf.name}: {ex}")

    return registry


# ─────────────────────────── SESSION STATE ────────────────────
if "model_registry" not in st.session_state:
    st.session_state.model_registry = load_model_registry()
if "dynamic_loaders" not in st.session_state:
    st.session_state.dynamic_loaders = {}
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
# Tracks which models have been verified this session (name -> bool: all files present)
if "file_verification_cache" not in st.session_state:
    st.session_state.file_verification_cache = {}
# Flag to auto-trigger a scan after extraction
if "trigger_auto_scan" not in st.session_state:
    st.session_state.trigger_auto_scan = False

MODEL_REGISTRY: dict = st.session_state.model_registry


# ─────────────────────────── STARTUP CLEANUP ──────────────────
# ─────────────────────────── STARTUP CLEANUP ──────────────────
_SUBCOMPONENT_KEYWORDS = [
    "vocos", "vocoder",
    "duration_predictor", "duration",
    "text_encoder",
    "vector_estimator",
    "lm_flow", "lm_main",
    "text_conditioner",
    "discriminator",
    "embedding",
    "decoder",        # ← ADD THIS — catches decoder.int8
    "encoder",        # ← ADD THIS — catches encoder.onnx
]

_UNKNOWN_VALUES = frozenset([
    "", "unknown", "UNKNOWN", "—", "?", "none", "None",
])

def _is_model_unusable(name: str, cfg: dict) -> tuple[bool, str]:
    """
    Returns (is_unusable: bool, reason: str).
    A model is unusable if:
      1. Its name or args contain sub-component keywords
      2. It has 2+ critical metadata fields that are unknown
      3. It is a CLI model with ALL of provider AND compute_type unknown
    """
    name_lower = name.lower()
    args_lower = " ".join(str(a) for a in cfg.get("args", [])).lower()
    full_str   = f"{name_lower} {args_lower}"

    # ── Rule 1: sub-component keyword in name or args ──────────
    for kw in _SUBCOMPONENT_KEYWORDS:
        # Match whole word or word boundary to avoid false positives
        # e.g. "decoder" should not block "decoder_ring_model"
        # but should block "decoder.int8" or "pocket_decoder"
        if kw in name_lower or kw in args_lower:
            return True, f"Sub-component file detected (keyword: '{kw}')"

    # ── Rule 2: too many unknown critical metadata fields ───────
    meta = cfg.get("meta", {})
    critical = ["provider", "compute_type", "license"]
    unknown_critical = [
        f for f in critical
        if str(meta.get(f, "")).strip().lower() in _UNKNOWN_VALUES
        or meta.get(f) is None
    ]
    if len(unknown_critical) >= 2:
        return True, (
            f"Too many unknown critical metadata fields: "
            f"{', '.join(unknown_critical)}"
        )

    return False, ""


def _purge_bad_models_from_registry():
    """
    Remove sub-component ONNX files and models with critically
    incomplete metadata from the registry. Also deletes their
    JSON files from disk. Runs once per session.
    """
    to_remove = []

    for name, cfg in list(st.session_state.model_registry.items()):
        unusable, reason = _is_model_unusable(name, cfg)
        if unusable:
            to_remove.append((name, reason))

    for name, reason in to_remove:
        st.session_state.model_registry.pop(name, None)
        if "file_verification_cache" in st.session_state:
            st.session_state.file_verification_cache.pop(name, None)

        # Delete by safe filename pattern
        safe = (name.replace(" ", "_").replace("/", "_")
                    .replace("(", "").replace(")", "")
                    .replace(".", "").replace("…", "")[:60])
        for candidate in [
            MODELS_DIR / f"{safe}.json",
            MODELS_DIR / f"{safe.lower()}.json",
        ]:
            if candidate.exists():
                candidate.unlink()

        # Deep search: scan ALL json files for this display_name
        for jf in list(MODELS_DIR.glob("*.json")):
            try:
                raw     = json.loads(jf.read_text(encoding="utf-8"))
                entries = raw if isinstance(raw, list) else [raw]
                if any(e.get("display_name") == name for e in entries):
                    jf.unlink()
                    break
            except Exception:
                pass

    if to_remove:
        names = [n for n, _ in to_remove]
        st.toast(
            f"🧹 Removed {len(to_remove)} unusable model(s): "
            + ", ".join(names[:3])
            + ("…" if len(to_remove) > 3 else ""),
            icon="🗑️",
        )

    return to_remove

if "exclusion_reasons" not in st.session_state:
    st.session_state.exclusion_reasons = {}

if "startup_cleanup_done" not in st.session_state:
    _purge_bad_models_from_registry()
    st.session_state.startup_cleanup_done = True
# ─────────────────────────── MODEL USABILITY FILTER ───────────
_UNKNOWN_VALUES = {
    None, "", "unknown", "UNKNOWN", "—", "?",
    "unknown\n", "UNKNOWN\n",
}


def _meta_field_is_unknown(value) -> bool:
    """Return True if a metadata field is effectively unknown/empty."""
    if value is None:
        return True
    return str(value).strip() in _UNKNOWN_VALUES


def get_model_usability(model_name: str) -> dict:
    """
    Assess whether a model is safe to use for generation.
    Returns a dict with:
      - 'usable'          : bool  — True if model can be used
      - 'unknown_meta'    : list  — metadata fields that are unknown
      - 'missing_files'   : list  — (flag, path) pairs for missing files
      - 'is_subcomponent' : bool  — True if model is a sub-component
      - 'reason'          : str   — human-readable reason if not usable
      - 'severity'        : str   — 'error' | 'warning' | 'ok'
    """
    cfg  = MODEL_REGISTRY.get(model_name, {})
    meta = cfg.get("meta", {})
    mtype = cfg.get("type", "cli")

    # ── Check 1: sub-component ────────────────────────────────
    name_lower = model_name.lower()
    args_str   = " ".join(cfg.get("args", [])).lower()
    search_str = f"{name_lower} {args_str}"
    is_subcomp = any(kw in search_str for kw in _SUBCOMPONENT_KEYWORDS)

    # ── Check 2: unknown metadata fields ─────────────────────
    critical_fields = ["provider", "compute_type"]
    all_meta_fields = [
        "provider", "parameters", "compute_type",
        "model_size", "license", "best_use_case",
    ]
    unknown_meta     = [f for f in all_meta_fields
                        if _meta_field_is_unknown(meta.get(f))]
    critical_unknown = [f for f in critical_fields
                        if _meta_field_is_unknown(meta.get(f))]

    # ── Check 3: missing files (CLI models only) ──────────────
    missing_files = []
    if mtype == "cli" and cfg.get("args"):
        missing_files = check_required_files(cfg["args"])

    # ── Determine usability ───────────────────────────────────
    if is_subcomp:
        return {
            "usable":          False,
            "unknown_meta":    unknown_meta,
            "missing_files":   missing_files,
            "is_subcomponent": True,
            "reason": (
                "This is a sub-component ONNX file (e.g. vocoder, duration "
                "predictor, encoder) — not a standalone TTS model. "
                "Delete it and register the correct parent model."
            ),
            "severity": "error",
        }

    if missing_files and mtype == "cli":
        return {
            "usable":          False,
            "unknown_meta":    unknown_meta,
            "missing_files":   missing_files,
            "is_subcomponent": False,
            "reason": (
                f"{len(missing_files)} required file(s) missing. "
                "Go to 📁 Local Files tab to fix."
            ),
            "severity": "error",
        }

    if len(unknown_meta) >= 4:
        # More than half the meta fields unknown — very likely a bad registration
        return {
            "usable":          False,
            "unknown_meta":    unknown_meta,
            "missing_files":   missing_files,
            "is_subcomponent": False,
            "reason": (
                f"{len(unknown_meta)} metadata fields are unknown "
                f"({', '.join(unknown_meta)}). "
                "This model was likely auto-scanned incorrectly. "
                "Fix metadata in 🔎 Model Inspector or delete and re-register."
            ),
            "severity": "error",
        }

    if critical_unknown:
        # Critical fields unknown — model may fail to load
        return {
            "usable":          False,
            "unknown_meta":    unknown_meta,
            "missing_files":   missing_files,
            "is_subcomponent": False,
            "reason": (
                f"Critical metadata fields are unknown: "
                f"{', '.join(critical_unknown)}. "
                "Go to 🔎 Model Inspector → Fix Metadata to resolve."
            ),
            "severity": "error",
        }

    if unknown_meta:
        # Some non-critical fields unknown — warn but allow
        return {
            "usable":          True,
            "unknown_meta":    unknown_meta,
            "missing_files":   missing_files,
            "is_subcomponent": False,
            "reason": (
                f"Non-critical metadata fields are unknown: "
                f"{', '.join(unknown_meta)}. "
                "Model may still work."
            ),
            "severity": "warning",
        }

    return {
        "usable":          True,
        "unknown_meta":    [],
        "missing_files":   [],
        "is_subcomponent": False,
        "reason":          "",
        "severity":        "ok",
    }


def get_usable_models(model_dict: dict) -> tuple[dict, dict]:
    """
    Split a model dict into (usable, excluded).
    Returns two dicts with the same structure as MODEL_REGISTRY.
    Also stores exclusion reasons in session state for UI display.
    """
    usable   = {}
    excluded = {}

    if "exclusion_reasons" not in st.session_state:
        st.session_state.exclusion_reasons = {}

    for name, cfg in model_dict.items():
        result = get_model_usability(name)
        if result["usable"]:
            usable[name] = cfg
        else:
            excluded[name] = cfg
            st.session_state.exclusion_reasons[name] = result

    return usable, excluded

# ─────────────────────────── HELPERS ──────────────────────────
def validate_model_json(raw: dict):
    errors, warnings = [], []
    for f in REQUIRED_FIELDS:
        if f not in raw:
            errors.append(f"Missing required field: `{f}`")
    mtype = raw.get("type", "")
    if mtype and mtype not in VALID_TYPES:
        errors.append(f"Invalid `type`: `{mtype}`. Must be one of: {VALID_TYPES}")
    langs = raw.get("langs", [])
    if not isinstance(langs, list) or not langs:
        errors.append("`langs` must be a non-empty list.")
    else:
        bad = [l for l in langs if l not in VALID_LANGS]
        if bad:
            errors.append(f"Unknown language code(s): {bad}")
    if mtype == "cli" and not isinstance(raw.get("args", []), list):
        errors.append("`args` must be a list for CLI models.")
    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        errors.append("`meta` must be a dict.")
    name = raw.get("display_name", "")
    if name and name in st.session_state.get("model_registry", {}):
        warnings.append(f"`{name}` already exists — saving will overwrite it.")
    return len(errors) == 0, errors, warnings


def save_model_json_to_disk(raw: dict, mark_installed: bool = True) -> Path:
    payload = dict(raw)
    payload.pop("_onnx_path", None)
    if mark_installed:
        payload["installed"] = True
    safe = (payload["display_name"]
            .replace(" ", "_").replace("/", "_")
            .replace("(", "").replace(")", "")
            .replace(".", "").replace("…", "")[:60])
    dest = MODELS_DIR / f"{safe}.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def register_model_in_session(cfg: dict):
    name = cfg["display_name"]
    st.session_state.model_registry[name] = {
        "type":          cfg.get("type",          "cli"),
        "langs":         cfg.get("langs",         ["en"]),
        "args":          cfg.get("args",          []),
        "meta":          cfg.get("meta",          {}),
        "downloads":     cfg.get("downloads",     []),
        "installed":     cfg.get("installed",     True),
        "sherpa_config": cfg.get("sherpa_config", {}),
    }
    # Invalidate verification cache for this model
    st.session_state.file_verification_cache.pop(name, None)


def check_required_files(args: list) -> list:
    """
    Returns a list of (flag, path) tuples for files that are declared
    in CLI args but do not exist on disk.
    """
    all_flags = {
        "--vits-model", "--vits-tokens", "--vits-lexicon",
        "--matcha-acoustic-model", "--matcha-vocoder", "--matcha-tokens",
        "--kokoro-model", "--kokoro-voices", "--kokoro-tokens",
        "--kitten-model", "--kitten-voices", "--kitten-tokens",
        "--vits-data-dir", "--matcha-data-dir", "--kokoro-data-dir", "--kitten-data-dir",
    }
    missing = []
    i = 0
    while i < len(args):
        if args[i] in all_flags and i + 1 < len(args):
            if not Path(args[i + 1]).exists():
                missing.append((args[i], args[i + 1]))
            i += 2
        else:
            i += 1
    return missing


def get_model_file_status(model_name: str) -> tuple[bool, list]:
    """
    Returns (all_present: bool, missing_files: list of (flag, path)).
    Uses session-state cache to avoid repeated disk checks.
    """
    cfg  = MODEL_REGISTRY.get(model_name, {})
    args = cfg.get("args", [])
    # Non-CLI models have no file args to check — treat as installed
    if cfg.get("type", "cli") != "cli" or not args:
        return True, []
    missing = check_required_files(args)
    all_ok  = len(missing) == 0
    # Update session registry installed flag if changed
    if MODEL_REGISTRY.get(model_name, {}).get("installed") != all_ok:
        st.session_state.model_registry[model_name]["installed"] = all_ok
    return all_ok, missing


def refresh_all_file_verification():
    """Re-verify all registered models and update installed flags."""
    st.session_state.file_verification_cache.clear()
    for name in list(st.session_state.model_registry.keys()):
        ok, _ = get_model_file_status(name)
        st.session_state.file_verification_cache[name] = ok

# ─────────────────────────── METADATA INFERENCE ───────────────
_KNOWN_MODEL_METADATA = {
    "supertonic": {
        "type": "supertonic", "provider": "SHERPA-ONNX / K2-FSA",
        "parameters": "~80M", "compute_type": "INT8/ONNX",
        "model_size": "~200MB", "license": "Apache 2.0",
        "best_use_case": "HIGH-QUALITY ENGLISH TTS",
        "langs": ["en"], "args": [], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2"
        ], "sherpa_config": {},
    },
    "pocket": {
        "type": "pocket", "provider": "SHERPA-ONNX / K2-FSA",
        "parameters": "~100M", "compute_type": "INT8/ONNX",
        "model_size": "~300MB", "license": "Apache 2.0",
        "best_use_case": "VOICE CLONING / ZERO-SHOT TTS",
        "langs": ["en"], "args": [], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2"
        ], "sherpa_config": {},
    },
    "kokoro": {
        "type": "cli", "provider": "K2-FSA / SHERPA-ONNX",
        "parameters": "~82M", "compute_type": "FP32/ONNX",
        "model_size": "~330MB", "license": "Apache 2.0",
        "best_use_case": "EXPRESSIVE MULTI-SPEAKER ENGLISH TTS",
        "langs": ["en"], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/kokoro-en-v0_19.tar.bz2"
        ], "sherpa_config": {},
    },
    "kitten": {
        "type": "cli", "provider": "CUSTOM / SHERPA-ONNX",
        "parameters": "~20M", "compute_type": "FP16/ONNX",
        "model_size": "~50MB", "license": "MIT",
        "best_use_case": "EDGE DEVICE / LOW LATENCY TTS",
        "langs": ["en"], "downloads": [], "sherpa_config": {},
    },
    "matcha": {
        "type": "cli", "provider": "ICEFALL / K2-FSA",
        "parameters": "~60M", "compute_type": "ONNX",
        "model_size": "~150MB", "license": "Apache 2.0",
        "best_use_case": "FAST HIGH-QUALITY ENGLISH TTS",
        "langs": ["en"], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
        ], "sherpa_config": {},
    },
    "vctk": {
        "type": "cli", "provider": "K2-FSA / PIPER",
        "parameters": "~40M", "compute_type": "INT8/ONNX",
        "model_size": "~37MB", "license": "MIT",
        "best_use_case": "MULTI-SPEAKER BRITISH/AMERICAN ENGLISH TTS",
        "langs": ["en"], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-vctk.tar.bz2"
        ], "sherpa_config": {},
    },
    "amy": {
        "type": "cli", "provider": "COQUI / PIPER",
        "parameters": "~40M", "compute_type": "FP32/ONNX",
        "model_size": "~100MB", "license": "MIT",
        "best_use_case": "GENERAL ENGLISH TTS (FEMALE)",
        "langs": ["en"], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-en_US-amy-low.tar.bz2"
        ], "sherpa_config": {},
    },
    "lessac": {
        "type": "cli", "provider": "COQUI / PIPER",
        "parameters": "~40M", "compute_type": "FP32/ONNX",
        "model_size": "~100MB", "license": "MIT",
        "best_use_case": "GENERAL ENGLISH TTS (MALE)",
        "langs": ["en"], "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-en_US-lessac-medium.tar.bz2"
        ], "sherpa_config": {},
    },
    "pratham": {
        "type": "cli", "provider": "K2-FSA / PIPER",
        "parameters": "~40M", "compute_type": "FP32/ONNX",
        "model_size": "~100MB", "license": "MIT",
        "best_use_case": "MALE HINDI TTS", "langs": ["hi"],
        "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-hi_IN-pratham-medium.tar.bz2"
        ], "sherpa_config": {},
    },
    "priyamvada": {
        "type": "cli", "provider": "K2-FSA / PIPER",
        "parameters": "~40M", "compute_type": "FP32/ONNX",
        "model_size": "~100MB", "license": "MIT",
        "best_use_case": "FEMALE HINDI TTS", "langs": ["hi"],
        "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2"
        ], "sherpa_config": {},
    },
    "rohan": {
        "type": "cli", "provider": "K2-FSA / PIPER",
        "parameters": "~40M", "compute_type": "FP32/ONNX",
        "model_size": "~100MB", "license": "MIT",
        "best_use_case": "MALE HINDI TTS", "langs": ["hi"],
        "downloads": [
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2"
        ], "sherpa_config": {},
    },
    "mms": {
        "type": "mms", "provider": "META AI",
        "parameters": "~50M", "compute_type": "FP16/PyTorch",
        "model_size": "~150MB", "license": "CC-BY-NC 4.0",
        "best_use_case": "MULTILINGUAL TTS (1000+ LANGUAGES)",
        "langs": ["en", "hi", "mar"], "downloads": [], "sherpa_config": {},
    },
    "parler": {
        "type": "indic_parler", "provider": "AI4BHARAT",
        "parameters": "~400M", "compute_type": "FP32/PyTorch",
        "model_size": "~3.75GB", "license": "Apache 2.0",
        "best_use_case": "EXPRESSIVE INDIC TTS (HI/MAR/EN)",
        "langs": ["hi", "mar", "en"], "downloads": [], "sherpa_config": {},
    },
}


def infer_model_metadata(model_name: str, cfg: dict) -> dict:
    """
    Try to infer missing/unknown metadata by matching the model
    name and args against _KNOWN_MODEL_METADATA patterns.
    Returns inference results including a ready-to-paste JSON.
    """
    name_lower = model_name.lower()
    args_str   = " ".join(cfg.get("args", [])).lower()
    search_str = f"{name_lower} {args_str}"

    current_meta = cfg.get("meta", {})

    # Check if this looks like a sub-component
    is_subcomponent = any(kw in search_str for kw in _SUBCOMPONENT_KEYWORDS)

    # Try to match a known model pattern
    matched_key  = None
    matched_data = {}
    for pattern_key, pattern_data in _KNOWN_MODEL_METADATA.items():
        if pattern_key in search_str:
            matched_key  = pattern_key
            matched_data = pattern_data
            break

    def _is_unknown(v):
        return v in (None, "", "unknown", "UNKNOWN", "—", "?")

    # Merge: keep existing good values, fill in gaps from known data
    inferred_meta = dict(current_meta)
    for field in ["provider", "parameters", "compute_type",
                  "model_size", "license", "best_use_case"]:
        if _is_unknown(inferred_meta.get(field)) and field in matched_data:
            inferred_meta[field] = matched_data[field]

    # Which fields are still unknown after inference?
    missing_fields = [
        f for f in ["provider", "parameters", "compute_type",
                    "model_size", "license", "best_use_case"]
        if _is_unknown(inferred_meta.get(f))
    ]

    # Confidence
    if matched_key and not missing_fields:
        confidence = "high"
    elif matched_key and len(missing_fields) <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Build the suggested complete JSON for the Add Model tab
    suggested = {
        "display_name":  model_name,
        "type":          matched_data.get("type",  cfg.get("type",  "cli")),
        "langs":         matched_data.get("langs", cfg.get("langs", ["en"])),
        "args":          cfg.get("args", []),
        "installed":     cfg.get("installed", False),
        "downloads":     matched_data.get("downloads", cfg.get("downloads", [])),
        "sherpa_config": matched_data.get("sherpa_config", cfg.get("sherpa_config", {})),
        "meta": {
            "provider":      inferred_meta.get("provider",      "UNKNOWN"),
            "parameters":    inferred_meta.get("parameters",    "unknown"),
            "compute_type":  inferred_meta.get("compute_type",  "ONNX"),
            "model_size":    inferred_meta.get("model_size",    "unknown"),
            "license":       inferred_meta.get("license",       "unknown"),
            "best_use_case": inferred_meta.get("best_use_case", "TTS"),
        },
    }

    return {
        "inferred_meta":   inferred_meta,
        "missing_fields":  missing_fields,
        "confidence":      confidence,
        "matched_key":     matched_key,
        "suggested_json":  suggested,
        "is_subcomponent": is_subcomponent,
    }


# ─────────────────────────── INSPECTOR HELPERS ────────────────
# (get_model_benchmark_history and _full_model_config follow here — unchanged)
# ─────────────────────────── INSPECTOR HELPERS ────────────────
def get_model_benchmark_history(model_name: str) -> pd.DataFrame:
    records = []
    for jf in sorted(JSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("model_name", "") == model_name:
                records.append({
                    "timestamp":         d.get("timestamp",          ""),
                    "input_text":        d.get("input_text",         ""),
                    "lang":              d.get("lang",               ""),
                    "wer":               d.get("wer",                None),
                    "cer":               d.get("cer",                None),
                    "mos_proxy":         d.get("mos_proxy",          None),
                    "rtf":               d.get("rtf",                None),
                    "generation_time_s": d.get("generation_time_s",  None),
                    "audio_duration_s":  d.get("audio_duration_s",   None),
                    "cpu_model_pct":     d.get("cpu_model_pct",      None),
                    "memory_mb":         d.get("memory_mb",          None),
                    "asr_transcript":    d.get("asr_transcript",     ""),
                    "output_wav":        d.get("output_wav",         ""),
                    "_jf":               str(jf),
                })
        except Exception:
            pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["wer", "cer", "mos_proxy", "rtf", "generation_time_s",
                "audio_duration_s", "cpu_model_pct", "memory_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _full_model_config(model_name: str) -> dict:
    cfg = dict(MODEL_REGISTRY.get(model_name, {}))
    cfg["display_name"] = model_name
    safe = (model_name
            .replace(" ", "_").replace("/", "_")
            .replace("(", "").replace(")", "")
            .replace(".", "").replace("…", "")[:60])
    candidate = MODELS_DIR / f"{safe}.json"
    if candidate.exists():
        cfg["_source_file"] = str(candidate)
    else:
        for jf in MODELS_DIR.glob("*.json"):
            try:
                raw = json.loads(jf.read_text(encoding="utf-8"))
                entries = raw if isinstance(raw, list) else [raw]
                for e in entries:
                    if e.get("display_name") == model_name:
                        cfg["_source_file"] = str(jf)
                        break
            except Exception:
                pass
    return cfg


# ─────────────────────────── CACHED RESOURCES ─────────────────
@st.cache_resource
def load_asr():
    return WhisperModel(ASR_MODEL_PATH, device="cpu", compute_type="int8", num_workers=2)

@st.cache_resource
def load_supertonic():
    import sherpa_onnx
    base = "./sherpa-onnx-supertonic-tts-int8-2026-03-06"
    cfg  = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=f"{base}/duration_predictor.int8.onnx",
                text_encoder      =f"{base}/text_encoder.int8.onnx",
                vector_estimator  =f"{base}/vector_estimator.int8.onnx",
                vocoder           =f"{base}/vocoder.int8.onnx",
                tts_json          =f"{base}/tts.json",
                unicode_indexer   =f"{base}/unicode_indexer.bin",
                voice_style       =f"{base}/voice.bin",
            )))
    return sherpa_onnx.OfflineTts(cfg)

@st.cache_resource
def load_pocket():
    import sherpa_onnx
    base = "./sherpa-onnx-pocket-tts-int8-2026-01-26"
    cfg  = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow          =f"{base}/lm_flow.int8.onnx",
                lm_main          =f"{base}/lm_main.int8.onnx",
                encoder          =f"{base}/encoder.onnx",
                decoder          =f"{base}/decoder.int8.onnx",
                text_conditioner =f"{base}/text_conditioner.onnx",
                vocab_json       =f"{base}/vocab.json",
                token_scores_json=f"{base}/token_scores.json",
            )))
    tts = sherpa_onnx.OfflineTts(cfg)
    ref, sr = librosa.load(f"{base}/test_wavs/bria.wav", sr=tts.sample_rate)
    return tts, ref, sr

@st.cache_resource
def load_kokoro():
    from kokoro import KModel, KPipeline
    MODEL_DIR = str(Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/main")
    model    = KModel(repo_id=None, config=f"{MODEL_DIR}/config.json",
                      model=f"{MODEL_DIR}/kokoro-v1_0.pth").eval()
    pipeline = KPipeline(lang_code="a", model=False)
    voice    = pipeline.load_voice("./voicess/af_heart.pt")
    return model, pipeline, voice

@st.cache_resource
def load_mms(lang: str):
    from transformers import VitsModel, VitsTokenizer
    import torch
    folder_map = {"en": "mms-eng", "hi": "mms-hi", "mar": "mms-mar"}
    model_dir  = str(Path(folder_map.get(lang, f"mms-{lang}")).resolve())
    tok = VitsTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = VitsModel.from_pretrained(model_dir, local_files_only=True)
    sr  = mdl.config.sampling_rate
    def _infer(text):
        inp = tok(text, return_tensors="pt")
        with torch.no_grad():
            audio = mdl(**inp).waveform.squeeze().numpy()
        return {"audio": audio, "sampling_rate": sr}
    return _infer

@st.cache_resource
def load_parler():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    model_dir = str(Path("indic-parler-tts").resolve())
    mdl = ParlerTTSForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    try:
        desc_tok = AutoTokenizer.from_pretrained(mdl.config.text_encoder._name_or_path)
    except Exception:
        desc_tok = tok
    return mdl, tok, desc_tok


# ─────────────────────────── LANGUAGE DETECTION ───────────────
def detect_language(text: str) -> str:
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    if dev == 0:
        return "en"
    mar_hints = ["आहे", "करा", "चाचणी", "सादर", "वेळेवर"]
    hi_hints  = ["है",  "करें", "जमा",  "कृपया", "भविष्य"]
    return ("mar" if sum(1 for w in mar_hints if w in text)
            >= sum(1 for w in hi_hints if w in text) else "hi")


# ─────────────────────────── GENERATION ───────────────────────
def generate(model_name: str, text: str):
    cfg      = MODEL_REGISTRY[model_name]
    mtype    = cfg["type"]
    langs    = cfg["langs"]
    det_lang = detect_language(text)
    if det_lang not in langs:
        det_lang = langs[0]
    proc     = psutil.Process(os.getpid())
    safe     = (model_name.replace(" ", "_").replace("/", "_")
                .replace("(", "").replace(")", ""))
    wav_path = str(OUTPUT_DIR / f"{safe[:40]}_{int(time.time())}.wav")
    t0, cpu0 = time.perf_counter(), proc.cpu_times()

    if mtype == "cli":
        cmd = (["python", "kitten_tts.py", "--output-filename", wav_path]
               + cfg["args"] + [text])
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore")
        if p.returncode != 0:
            raise RuntimeError(p.stderr[:500] or p.stdout[:500])
        audio, sr = librosa.load(wav_path, sr=None)
    elif mtype == "supertonic":
        tts = load_supertonic()
        out = tts.generate(text)
        audio, sr = np.array(out.samples), out.sample_rate
        sf.write(wav_path, audio, sr)
    elif mtype == "pocket":
        import sherpa_onnx
        tts, ref, ref_sr = load_pocket()
        gcfg = sherpa_onnx.GenerationConfig()
        gcfg.reference_audio       = ref
        gcfg.reference_sample_rate = ref_sr
        out  = tts.generate(text, gcfg)
        audio, sr = np.array(out.samples), tts.sample_rate
        sf.write(wav_path, audio, sr)
    elif mtype == "kokoro":
        mdl, pipeline, voice = load_kokoro()
        res = list(pipeline(text, "af_heart"))
        _, ps, _ = res[0]
        audio = mdl(ps, voice[len(ps) - 1], 1.0)
        sr    = 24000
        if not isinstance(audio, np.ndarray):
            audio = audio.numpy()
        sf.write(wav_path, audio, sr)
    elif mtype == "mms":
        infer = load_mms(det_lang)
        out   = infer(text)
        audio, sr = out["audio"], out["sampling_rate"]
        sf.write(wav_path, audio, sr)
    elif mtype == "indic_parler":
        import torch
        mdl, tok, desc_tok = load_parler()
        desc = ("A clear, neutral speaker delivers the speech at a moderate pace "
                "with very high recording quality and no background noise.")
        desc_in = desc_tok(desc, return_tensors="pt")
        text_in = tok(text, return_tensors="pt")
        with torch.no_grad():
            gen = mdl.generate(
                input_ids=desc_in.input_ids,
                attention_mask=desc_in.attention_mask,
                prompt_input_ids=text_in.input_ids,
                prompt_attention_mask=text_in.attention_mask,
            )
        audio = gen.cpu().numpy().squeeze()
        sr    = mdl.config.sampling_rate
        sf.write(wav_path, audio, sr)
    elif mtype == "dynamic_sherpa":
        infer = st.session_state.dynamic_loaders.get(model_name)
        if infer is None:
            raise RuntimeError(f"No dynamic loader found for: {model_name}")
        audio, sr = infer(text, wav_path)
    else:
        raise RuntimeError(f"Unknown model type: {mtype}")

    t1, cpu1  = time.perf_counter(), proc.cpu_times()
    gen_time  = t1 - t0
    duration  = len(audio) / sr if sr > 0 else 0
    rtf       = gen_time / duration if duration > 0 else -1
    cpu_used  = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    cpu_pct   = (cpu_used / (gen_time * psutil.cpu_count())) * 100 if gen_time > 0 else 0
    mem_mb    = proc.memory_info().rss / (1024 * 1024)
    return wav_path, audio, sr, gen_time, duration, rtf, cpu_pct, mem_mb, det_lang


def compute_metrics(wav_path, ref_text, gen_time, duration, rtf, cpu_pct, mem_mb):
    asr = load_asr()
    try:
        segs, _ = asr.transcribe(wav_path, beam_size=5, temperature=0.0, best_of=1)
        hyp = " ".join(s.text for s in segs).strip()
        wer = round(jiwer.wer(WER_TRANSFORM(ref_text), WER_TRANSFORM(hyp)), 4)
        cer = round(jiwer.cer(WER_TRANSFORM(ref_text), WER_TRANSFORM(hyp)), 4)
        mos = round(max(0, 1 - wer), 4)
    except Exception as e:
        hyp, wer, cer, mos = f"ASR Error: {e}", -1, -1, -1
    return dict(asr_transcript=hyp, wer=wer, cer=cer, mos_proxy=mos,
                generation_time_s=round(gen_time, 4), audio_duration_s=round(duration, 4),
                rtf=round(rtf, 4), cpu_model_pct=round(cpu_pct, 2), memory_mb=round(mem_mb, 2))


def save_result(model_name, text, lang, wav_path, metrics):
    meta   = MODEL_REGISTRY.get(model_name, {}).get("meta", {})
    record = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "model_name": model_name, "lang": lang,
              "input_text": text, "output_wav": wav_path, **meta, **metrics}
    safe  = (model_name.replace(" ", "_").replace("/", "_")
             .replace("(", "").replace(")", ""))
    jpath = JSON_DIR / f"{safe[:30]}_{int(time.time())}.json"
    jpath.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record, str(jpath)


def compute_leaderboard() -> pd.DataFrame:
    records = []
    for jf in JSON_DIR.glob("*.json"):
        try:
            records.append(json.loads(jf.read_text(encoding="utf-8")))
        except Exception:
            pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    mc = ["wer", "cer", "mos_proxy", "rtf", "generation_time_s", "cpu_model_pct", "memory_mb"]
    ex = [c for c in mc if c in df.columns]
    for c in ex:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg   = df.groupby("model_name")[ex].mean().reset_index()
    wer_s = agg["wer"]       if "wer"       in agg.columns else pd.Series(1, index=agg.index)
    rtf_s = agg["rtf"]       if "rtf"       in agg.columns else pd.Series(1, index=agg.index)
    mos_s = agg["mos_proxy"] if "mos_proxy" in agg.columns else pd.Series(0, index=agg.index)
    agg["score"] = (0.5*(1-wer_s.fillna(1)) + 0.3*(1/(1+rtf_s.fillna(1)))
                    + 0.2*mos_s.fillna(0)).round(4)
    agg["runs"]  = df.groupby("model_name").size().values
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg)+1))
    return agg


def load_all_json_results() -> list:
    results = []
    for jf in sorted(JSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            d["_filepath"] = str(jf)
            results.append(d)
        except Exception:
            pass
    return results


# ─────────────────────────── UI HELPERS ───────────────────────
def rtf_class(v):
    if v < 0: return ""
    return "metric-good" if v < 0.5 else "metric-warn" if v < 1.0 else "metric-bad"

def wer_class(v):
    if v < 0: return ""
    return "metric-good" if v < 0.2 else "metric-warn" if v < 0.5 else "metric-bad"

def metric_card(label, value, css_class="", sub=""):
    sub_html = f"<div style='color:#636678;font-size:0.72rem'>{sub}</div>" if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {css_class}">{value}</div>{sub_html}</div>')

def render_metric_rows(m: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("RTF",       f"{m.get('rtf',0):.3f}",
                             rtf_class(m.get("rtf",0)), "lower = faster"), unsafe_allow_html=True)
    c2.markdown(metric_card("WER",       f"{m.get('wer',0):.3f}",
                             wer_class(m.get("wer",0)), "lower = better"), unsafe_allow_html=True)
    c3.markdown(metric_card("CER",       f"{m.get('cer',0):.3f}",
                             wer_class(m.get("cer",0)), "lower = better"), unsafe_allow_html=True)
    c4.markdown(metric_card("MOS Proxy", f"{m.get('mos_proxy',0):.3f}",
                             "metric-good", "higher = better"), unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(metric_card("Gen Time",  f"{m.get('generation_time_s',0):.2f}s"), unsafe_allow_html=True)
    c6.markdown(metric_card("Audio Dur", f"{m.get('audio_duration_s',0):.2f}s"),  unsafe_allow_html=True)
    c7.markdown(metric_card("CPU Usage", f"{m.get('cpu_model_pct',0):.1f}%"),     unsafe_allow_html=True)
    c8.markdown(metric_card("Memory",    f"{m.get('memory_mb',0):.0f} MB"),       unsafe_allow_html=True)


# ─────────────────────────── CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{ font-family:'Sora',sans-serif !important; }
.main{ background:#0d0f14; }
.metric-card{ background:#161920;border:1px solid #252830;border-radius:12px;padding:16px 20px;text-align:center;margin-bottom:12px; }
.metric-label{ font-size:0.72rem;color:#636678;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:4px; }
.metric-value{ font-size:1.6rem;font-weight:700;font-family:'DM Mono',monospace;color:#e8eaf0; }
.metric-good{ color:#5ef08a !important; }
.metric-warn{ color:#fbbf24 !important; }
.metric-bad { color:#f87171 !important; }
.stream-box{ background:#161920;border:1px solid #252830;border-left:3px solid #7c6af7;border-radius:12px;padding:16px 20px;font-family:'DM Mono',monospace;font-size:1rem;line-height:1.8;color:#5ef08a;min-height:80px; }
.model-info-card{ background:#161920;border:1px solid #252830;border-radius:12px;padding:16px 20px;margin-bottom:16px;font-size:0.88rem;color:#9ca3af;line-height:1.7; }
.model-info-card.missing-files{ border-left:4px solid #f87171; }
.app-title{ font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#7c6af7 0%,#f76a6a 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0; }
.section-header{ font-size:0.75rem;color:#636678;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;margin:20px 0 8px;border-bottom:1px solid #252830;padding-bottom:6px; }
.compare-header{ background:#1a1d26;border:1px solid #2d3142;border-radius:10px;padding:12px 16px;margin-bottom:8px;font-weight:600;color:#c4c8e0;font-size:0.9rem; }
.reg-box  { background:#0f1a12;border:1px solid #1e3a22;border-left:3px solid #5ef08a;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:0.88rem; }
.reg-error{ background:#1a0f0f;border:1px solid #3a1e1e;border-left:3px solid #f87171;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:0.88rem; }
.reg-warn { background:#1a1600;border:1px solid #3a3000;border-left:3px solid #fbbf24;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:0.88rem; }
.file-ok  { background:#0f1a12;border:1px solid #1e3a22;border-radius:8px;padding:10px 14px;margin:4px 0;font-family:'DM Mono',monospace;font-size:0.82rem;color:#5ef08a; }
.file-miss{ background:#1a1010;border:1px solid #3a2020;border-radius:8px;padding:10px 14px;margin:4px 0;font-family:'DM Mono',monospace;font-size:0.82rem;color:#f87171; }
.scan-card{ background:#161920;border:1px solid #252830;border-left:3px solid #7c6af7;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:0.85rem;color:#c4c8e0; }
.scan-card-ok{ background:#0f1a12;border:1px solid #1e3a22;border-left:3px solid #5ef08a;border-radius:10px;padding:14px 18px;margin:8px 0;font-size:0.85rem;color:#c4c8e0; }
.offline-box{ background:#0d1a2e;border:1px solid #1a3a5c;border-left:3px solid #38bdf8;border-radius:10px;padding:16px 20px;margin:10px 0;font-size:0.9rem;color:#93c5fd; }
.step-badge{ display:inline-block;background:#7c6af7;color:#fff;border-radius:50%;width:22px;height:22px;text-align:center;font-size:0.75rem;font-weight:700;line-height:22px;margin-right:6px; }
.archive-card{ background:#161920;border:1px solid #252830;border-left:3px solid #f59e0b;border-radius:10px;padding:14px 18px;margin:6px 0;font-size:0.85rem;color:#fbbf24; }

/* ── Verify section styles ── */
.verify-card-ok  { background:#0f1a12;border:1px solid #1e3a22;border-left:4px solid #5ef08a;border-radius:10px;padding:14px 18px;margin:6px 0; }
.verify-card-miss{ background:#1a0f12;border:1px solid #3a1e1e;border-left:4px solid #f87171;border-radius:10px;padding:14px 18px;margin:6px 0; }
.verify-card-na  { background:#161920;border:1px solid #252830;border-left:4px solid #636678;border-radius:10px;padding:14px 18px;margin:6px 0; }
.verify-model-name{ font-weight:700;color:#e8eaf0;font-size:0.9rem;margin-bottom:6px; }
.verify-status-ok  { color:#5ef08a;font-size:0.82rem;font-weight:600; }
.verify-status-miss{ color:#f87171;font-size:0.82rem;font-weight:600; }
.verify-status-na  { color:#636678;font-size:0.82rem;font-weight:600; }
.dl-url-chip{ display:inline-block;background:#0d1a2e;border:1px solid #1a3a5c;border-radius:6px;padding:4px 10px;margin:3px 0;font-family:'DM Mono',monospace;font-size:0.75rem;color:#38bdf8;word-break:break-all; }

/* ── Inspector styles ── */
.insp-header{ background:linear-gradient(135deg,#1a1d26 0%,#0f1a12 100%);border:1px solid #2d3142;border-radius:14px;padding:20px 24px;margin-bottom:20px; }
.insp-kv{ display:flex;gap:10px;margin:4px 0;font-size:0.86rem; }
.insp-key{ color:#636678;font-weight:600;min-width:130px;flex-shrink:0; }
.insp-val{ color:#e8eaf0;font-family:'DM Mono',monospace; }
.insp-section{ background:#161920;border:1px solid #252830;border-radius:10px;padding:16px 20px;margin:10px 0; }
.insp-arg-chip{ display:inline-block;background:#1a1d26;border:1px solid #2d3142;border-radius:6px;padding:3px 10px;margin:3px;font-family:'DM Mono',monospace;font-size:0.78rem;color:#c4c8e0; }
.insp-flag-chip{ display:inline-block;background:#0f1a2e;border:1px solid #1a3a5c;border-radius:6px;padding:3px 10px;margin:3px;font-family:'DM Mono',monospace;font-size:0.78rem;color:#38bdf8; }
.insp-metric-row{ display:flex;gap:12px;flex-wrap:wrap;margin:8px 0; }
.insp-mini-card{ background:#1a1d26;border:1px solid #2d3142;border-radius:8px;padding:10px 14px;text-align:center;min-width:100px;flex:1; }
.insp-mini-label{ font-size:0.65rem;color:#636678;text-transform:uppercase;letter-spacing:0.08em; }
.insp-mini-val{ font-size:1.1rem;font-weight:700;font-family:'DM Mono',monospace;color:#e8eaf0; }
.bench-row{ background:#161920;border:1px solid #252830;border-radius:8px;padding:12px 16px;margin:6px 0;font-size:0.83rem;color:#c4c8e0;border-left:3px solid #7c6af7; }
.bench-text{ font-style:italic;color:#9ca3af;font-size:0.8rem;margin-top:4px; }
.no-data-box{ background:#161920;border:1px dashed #252830;border-radius:10px;padding:30px;text-align:center;color:#636678;font-size:0.9rem; }

/* ── Missing files banner in Generate tab ── */
.missing-banner{ background:#1a0f0f;border:1px solid #3a1e1e;border-left:4px solid #f87171;border-radius:10px;padding:16px 20px;margin:12px 0;font-size:0.9rem;color:#fca5a5; }
.missing-banner b{ color:#f87171; }

/* ── Workflow guide ── */
.workflow-step{ display:flex;align-items:flex-start;gap:12px;margin:10px 0;padding:12px 16px;background:#161920;border:1px solid #252830;border-radius:8px; }
.workflow-step-num{ background:#7c6af7;color:#fff;border-radius:50%;width:28px;height:28px;min-width:28px;text-align:center;font-size:0.85rem;font-weight:700;line-height:28px; }
.workflow-step-text{ color:#c4c8e0;font-size:0.88rem;line-height:1.6; }
            
/* ── Metadata inference indicators ── */
.meta-inferred { color:#fbbf24 !important; font-style:italic; }
.meta-unknown  { color:#f87171 !important; }
.meta-ok       { color:#e8eaf0 !important; }

/* ── Sub-component error box ── */
.subcomp-error {
    background:#1a0505;
    border:2px solid #f87171;
    border-radius:10px;
    padding:16px 20px;
    margin:10px 0;
    color:#fca5a5;
    font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── BATCH GENERATE HELPER ───────────
def generate_batch_filename(index: int, model_name: str, lang: str, text: str) -> str:
    """
    Generate output filename for batch jobs.
    Format: {index}_{model_slug}_{lang}.wav
    """
    model_slug = (model_name.lower()
                  .replace(" ", "_").replace("/", "_")
                  .replace("(", "").replace(")", "")
                  .replace(".", "").replace("—", "")
                  .replace("…", "")[:30])
    return f"{index}_{model_slug}_{lang}.wav"


def run_batch_generation(
    sentences: list[str],
    model_name: str,
    output_dir: Path,
    progress_callback=None,
) -> list[dict]:
    """
    Run TTS generation for a list of sentences using one model.
    Returns list of result dicts with keys:
      index, input_text, lang, wav_path, status, error, metrics
    """
    results = []
    for i, text in enumerate(sentences, start=1):
        text = text.strip()
        if not text:
            continue
        det_lang = detect_language(text)
        cfg = MODEL_REGISTRY.get(model_name, {})
        if det_lang not in cfg.get("langs", ["en"]):
            det_lang = cfg.get("langs", ["en"])[0]

        out_filename = generate_batch_filename(i, model_name, det_lang, text)
        out_path     = output_dir / out_filename

        if progress_callback:
            progress_callback(i, len(sentences), text)

        try:
            (wav_path, audio, sr, gen_time, duration,
             rtf, cpu_pct, mem_mb, det_lang) = generate(model_name, text)
            # Copy to batch output path with clean name
            import shutil
            shutil.copy2(wav_path, str(out_path))
            metrics = compute_metrics(
                str(out_path), text, gen_time, duration, rtf, cpu_pct, mem_mb
            )
            results.append({
                "index":      i,
                "input_text": text,
                "lang":       det_lang,
                "wav_path":   str(out_path),
                "status":     "success",
                "error":      "",
                **metrics,
            })
        except Exception as e:
            results.append({
                "index":      i,
                "input_text": text,
                "lang":       det_lang,
                "wav_path":   "",
                "status":     "error",
                "error":      str(e),
                "wer": -1, "cer": -1, "mos_proxy": -1, "rtf": -1,
                "generation_time_s": -1, "audio_duration_s": -1,
                "cpu_model_pct": -1, "memory_mb": -1,
                "asr_transcript": "",
            })
    return results


# ─────────────────────────── SIDEBAR ──────────────────────────
with st.sidebar:
    st.markdown('<p class="app-title">⚡ TTS Studio</p>', unsafe_allow_html=True)
    st.markdown("<p style='color:#636678;font-size:0.85rem;margin-top:-8px;'>"
                "Benchmark · Compare · Analyse</p>", unsafe_allow_html=True)
    n_json = len(list(MODELS_DIR.glob("*.json")))
    st.caption(f"📦 {len(MODEL_REGISTRY)} models loaded "
               f"({'JSON (' + str(n_json) + ') + ' if n_json else ''}hardcoded defaults)")
    st.divider()

    st.markdown('<p class="section-header">Language Filter</p>', unsafe_allow_html=True)
    lang_filter = st.radio("Filter", ["All","English","Hindi","Marathi","Multilingual"],
                           horizontal=True, label_visibility="collapsed")

    def lang_matches(cfg):
        if lang_filter == "All":          return True
        if lang_filter == "English":      return cfg["langs"] == ["en"]
        if lang_filter == "Hindi":        return "hi" in cfg["langs"] and "mar" not in cfg["langs"]
        if lang_filter == "Marathi":      return "mar" in cfg["langs"]
        if lang_filter == "Multilingual": return len(cfg["langs"]) > 1
        return True

        # Language filter
        # Language filter + usability filter combined
    # Bad models are excluded from ALL tabs at this point
    filtered        = {}
    _truly_excluded = {}

    for _name, _cfg in MODEL_REGISTRY.items():
        if not lang_matches(_cfg):
            continue
        _bad, _why = _is_bad_model(_name, _cfg)
        if _bad:
            _truly_excluded[_name] = {"cfg": _cfg, "reason": _why}
        else:
            filtered[_name] = _cfg

    # Store for display in tabs
    st.session_state.exclusion_reasons = {
        n: d["reason"] for n, d in _truly_excluded.items()
    }

    

    st.divider()
    st.markdown('<p class="section-header">Quick Examples</p>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    use_en  = ec1.button("🇬🇧 EN")
    use_hi  = ec2.button("🇮🇳 HI")
    use_mar = ec3.button("🇮🇳 MAR")
    st.divider()

    st.markdown('<p class="section-header">📁 Local Models</p>', unsafe_allow_html=True)
    if st.button("🔍 Scan & Register Local Files", key="sidebar_scan",
                 use_container_width=True, type="primary"):
        with st.spinner("Scanning…"):
            candidates = scan_local_models(BASE_DIR)
        new_count = 0
        for c in candidates:
            if c["display_name"] not in st.session_state.model_registry:
                register_model_in_session(c)
                save_model_json_to_disk(c, mark_installed=True)
                new_count += 1
        if new_count:
            st.success(f"✅ Registered {new_count} new local model(s)!")
        else:
            st.info(f"Found {len(candidates)} model(s) — all already registered.")

    st.divider()
    st.markdown('<p class="section-header">Recent Results</p>', unsafe_allow_html=True)
    for jf in sorted(JSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            with st.expander(f"{d.get('model_name','?')[:28]}"):
                st.json({k: d[k] for k in ["model_name","lang","wer","cer","rtf",
                                             "mos_proxy","asr_transcript"] if k in d})
        except Exception:
            pass


# ─────────────────────────── TABS ─────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎙 Generate",
    "⚖️ Compare",
    "🏆 Leaderboard",
    "🗂 History",
    "📊 Import & Visualize",
    "🔎 Model Inspector",
    "📁 Local Files",
    "➕ Add Model (JSON)",
    
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════
with tab1:
    gen_mode = st.radio(
        "Mode",
        ["🎙 Single Generate", "🗃 Batch Generate"],
        horizontal=True,
        key="gen_mode_toggle",
    )

    # ══════════════════════════════════════════════════════════
    # SINGLE GENERATE MODE
    # ══════════════════════════════════════════════════════════
    if gen_mode == "🎙 Single Generate":

        usable_filtered   = filtered
        excluded_filtered = _truly_excluded

        if excluded_filtered:
            with st.expander(
                f"⚠️ {len(excluded_filtered)} model(s) hidden "
                f"(unusable — incomplete metadata or sub-components)",
                expanded=False,
            ):
                st.caption(
                    "Fix these in **🔎 Model Inspector** → Fix Metadata, "
                    "or delete them via **📁 Local Files** → Section F."
                )
                for exc_name, exc_data in excluded_filtered.items():
                    reason_text = exc_data.get("reason", "Unknown reason.")
                    st.markdown(
                        f'<div class="reg-error" style="margin:4px 0;">'
                        f'🚫 <b>{exc_name}</b><br>'
                        f'<span style="font-size:0.82rem;color:#fca5a5">'
                        f'{reason_text}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('<p class="section-header">Model</p>', unsafe_allow_html=True)

        if not usable_filtered:
            st.error(
                "❌ No usable models available for the current language filter. "
                "All models are either missing files or have incomplete metadata.\n\n"
                "**To fix:**\n"
                "- Go to **🔎 Model Inspector** → select a model → use "
                "**Fix Metadata** or **Quick Metadata Edit**\n"
                "- Go to **📁 Local Files** → extract archives → re-scan\n"
                "- Go to **➕ Add Model (JSON)** → paste a corrected config"
            )
        else:
            model_name = st.selectbox(
                "Model",
                list(usable_filtered.keys()),
                label_visibility="collapsed",
                key="gen_model",
            )

            all_files_ok  = True
            missing_files = []

            if model_name and model_name in MODEL_REGISTRY:
                cfg       = MODEL_REGISTRY[model_name]
                meta      = cfg.get("meta", {})
                usability = get_model_usability(model_name)
                langs_str = " · ".join(LANG_LABEL.get(l, l) for l in cfg["langs"])

                all_files_ok, missing_files = get_model_file_status(model_name)

                if usability["severity"] == "warning":
                    badge = (
                        " &nbsp;·&nbsp; <span style='color:#fbbf24;font-size:0.8rem'>"
                        f"⚠ {len(usability['unknown_meta'])} meta field(s) incomplete</span>"
                    )
                    card_class = "model-info-card"
                elif all_files_ok:
                    badge      = (" &nbsp;·&nbsp; <span style='color:#5ef08a;font-size:0.8rem'>"
                                  "✓ Ready</span>")
                    card_class = "model-info-card"
                else:
                    badge      = (" &nbsp;·&nbsp; <span style='color:#f87171;font-size:0.8rem'>"
                                  f"⚠ {len(missing_files)} file(s) missing</span>")
                    card_class = "model-info-card missing-files"

                st.markdown(
                    f"""<div class="{card_class}">
                    <b style="color:#e8eaf0">{model_name}</b>{badge}<br>
                    Provider: <code>{meta.get('provider','—')}</code>
                    &nbsp;|&nbsp; Size: <code>{meta.get('model_size','—')}</code>
                    &nbsp;|&nbsp; Compute: <code>{meta.get('compute_type','—')}</code><br>
                    Languages: {langs_str}
                    &nbsp;|&nbsp; License: <code>{meta.get('license','—')}</code><br>
                    Best use: <i>{meta.get('best_use_case','—')}</i>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if usability["severity"] == "warning" and usability["unknown_meta"]:
                    unknown_list = ", ".join(f"`{f}`" for f in usability["unknown_meta"])
                    st.warning(
                        f"⚠️ Some metadata fields are incomplete: {unknown_list}. "
                        f"The model may still work. Fix in **🔎 Model Inspector**."
                    )

                if not all_files_ok and cfg.get("type") == "cli":
                    st.markdown(
                        f'<div class="missing-banner">'
                        f'<b>⚠️ {len(missing_files)} missing file(s) — '
                        f'generation disabled.</b><br>'
                        f'Go to <b>📁 Local Files</b> tab to fix.<br><br>'
                        + "".join(
                            f'<div style="margin:3px 0;font-family:monospace;font-size:0.8rem;">'
                            f'✗ <b>{flag}</b> → {path}</div>'
                            for flag, path in missing_files
                        )
                        + '</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<p class="section-header">Input Text</p>', unsafe_allow_html=True)

            default_text = ""
            if use_en:  default_text = "Revenue increased by 17.3% to $1.25 million this quarter."
            if use_hi:  default_text = "कृपया रिपोर्ट समय पर जमा करें।"
            if use_mar: default_text = "कृपया अहवाल वेळेवर सादर करा।"

            input_text = st.text_area(
                "Text",
                value=default_text,
                height=120,
                placeholder="Type in English, Hindi (हिंदी), or Marathi (मराठी)…",
                label_visibility="collapsed",
                key="gen_text",
            )
            if input_text.strip():
                det = detect_language(input_text)
                st.caption(f"🔍 Detected: **{LANG_LABEL.get(det, det)}**")

            col_g, col_c = st.columns([3, 1])

            _gen_disabled = (
                model_name in MODEL_REGISTRY
                and MODEL_REGISTRY[model_name].get("type") == "cli"
                and not all_files_ok
            ) if model_name in MODEL_REGISTRY else False

            gen_btn   = col_g.button(
                "▶ Generate Audio",
                type="primary",
                use_container_width=True,
                key="gen_btn",
                disabled=_gen_disabled,
            )
            clear_btn = col_c.button("✕ Clear", use_container_width=True, key="gen_clr")

            if _gen_disabled:
                st.caption("⚠️ Generation disabled — fix missing files in **📁 Local Files** tab.")

            if clear_btn:
                st.rerun()

            if gen_btn:
                if not input_text.strip():
                    st.warning("Please enter some text first.")
                else:
                    det_lang = detect_language(input_text)
                    cfg      = MODEL_REGISTRY[model_name]
                    if det_lang not in cfg["langs"]:
                        det_lang = cfg["langs"][0]
                        st.info(f"Using **{LANG_LABEL.get(det_lang, det_lang)}** (model default).")
                    with st.spinner(f"Generating with **{model_name}**…"):
                        try:
                            (wav_path, audio, sr, gen_time, duration,
                             rtf, cpu_pct, mem_mb, det_lang) = generate(model_name, input_text)
                            st.success(f"✅ Generated in **{gen_time:.2f}s**")
                        except Exception as e:
                            st.error(f"❌ Generation failed: {e}")
                            st.stop()

                    st.markdown('<p class="section-header">Audio</p>', unsafe_allow_html=True)
                    st.audio(wav_path, format="audio/wav", autoplay=True)

                    st.markdown(
                        '<p class="section-header">Streaming Transcript</p>',
                        unsafe_allow_html=True,
                    )
                    ph = st.empty()
                    words, streamed = input_text.split(), ""
                    for word in words:
                        streamed += word + " "
                        ph.markdown(
                            f'<div class="stream-box">{streamed.strip()}</div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(min(duration / max(len(words), 1), 0.25))

                    st.markdown(
                        '<p class="section-header">Performance Metrics</p>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Computing WER / CER…"):
                        metrics = compute_metrics(
                            wav_path, input_text, gen_time, duration, rtf, cpu_pct, mem_mb
                        )
                    render_metric_rows(metrics)

                    st.markdown(
                        '<p class="section-header">ASR Transcript</p>',
                        unsafe_allow_html=True,
                    )
                    st.info(f"**Whisper heard:** _{metrics['asr_transcript']}_")

                    record, jpath = save_result(
                        model_name, input_text, det_lang, wav_path, metrics
                    )
                    st.caption(f"💾 Saved → `{Path(jpath).name}`")
                    with st.expander("View full JSON"):
                        st.json(record)

    # ══════════════════════════════════════════════════════════
    # BATCH GENERATE MODE
    # ══════════════════════════════════════════════════════════
    elif gen_mode == "🗃 Batch Generate":

        BATCH_OUTPUT_DIR = BASE_DIR / "batch_outputs"
        BATCH_OUTPUT_DIR.mkdir(exist_ok=True)

        st.markdown('<p class="section-header">Step 1 — Load Input Sentences</p>',
                    unsafe_allow_html=True)

        input_method = st.radio(
            "Input method",
            ["📄 Upload CSV", "✏️ Paste Text"],
            horizontal=True,
            key="batch_input_method",
        )

        batch_sentences: list = []

        if input_method == "📄 Upload CSV":
            uploaded_csv = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                label_visibility="collapsed",
                key="batch_csv_upload",
            )
            if uploaded_csv:
                try:
                    df_csv      = pd.read_csv(uploaded_csv, header=None, dtype=str)
                    col_options = list(df_csv.columns)
                    if len(col_options) > 1:
                        chosen_col = st.selectbox(
                            "Which column contains the sentences?",
                            col_options,
                            key="batch_csv_col",
                        )
                    else:
                        chosen_col = col_options[0]

                    raw_sentences = df_csv[chosen_col].dropna().tolist()
                    cleaned = []
                    for s in raw_sentences:
                        s = str(s).strip()
                        s = re.sub(r"^[\*\-\•\d]+[\.\):\s]+", "", s).strip()
                        if s:
                            cleaned.append(s)
                    batch_sentences = cleaned
                    st.session_state["_batch_sentences_ready"] = batch_sentences

                    st.success(f"✅ Loaded **{len(batch_sentences)}** sentence(s) from CSV.")
                    with st.expander("Preview sentences", expanded=False):
                        for idx, s in enumerate(batch_sentences[:20], 1):
                            st.markdown(f"`{idx}.` {s}")
                        if len(batch_sentences) > 20:
                            st.caption(f"… and {len(batch_sentences)-20} more.")
                except Exception as e:
                    st.error(f"❌ Failed to read CSV: {e}")

        else:
            pasted = st.text_area(
                "Paste sentences (one per line, or bullet/asterisk list)",
                height=200,
                placeholder="* Play Senorita\n* Navigate to home\n* Turn on AC\n...",
                label_visibility="collapsed",
                key="batch_paste_input",
            )
            if pasted.strip():
                lines   = pasted.strip().splitlines()
                cleaned = []
                for line in lines:
                    line = line.strip()
                    line = re.sub(r"^[\*\-\•\d]+[\.\):\s]+", "", line).strip()
                    if line:
                        cleaned.append(line)
                batch_sentences = cleaned
                st.session_state["_batch_sentences_ready"] = batch_sentences
                st.caption(f"Parsed **{len(batch_sentences)}** sentence(s).")

        # ── Model Selection ───────────────────────────────────
        st.markdown('<p class="section-header">Step 2 — Select Model</p>',
                    unsafe_allow_html=True)

        usable_batch = filtered

        if not usable_batch:
            st.error("❌ No usable models available.")
        else:
            batch_model = st.selectbox(
                "Model for batch generation",
                list(usable_batch.keys()),
                label_visibility="collapsed",
                key="batch_model_select",
            )

            if batch_model:
                bcfg        = MODEL_REGISTRY[batch_model]
                bmeta       = bcfg.get("meta", {})
                langs_str_b = " · ".join(LANG_LABEL.get(l, l) for l in bcfg["langs"])
                st.markdown(
                    f'<div class="model-info-card">'
                    f'<b style="color:#e8eaf0">{batch_model}</b><br>'
                    f'Provider: <code>{bmeta.get("provider","—")}</code> &nbsp;|&nbsp; '
                    f'Compute: <code>{bmeta.get("compute_type","—")}</code><br>'
                    f'Languages: {langs_str_b} &nbsp;|&nbsp; '
                    f'License: <code>{bmeta.get("license","—")}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                _ready_sentences = st.session_state.get("_batch_sentences_ready", [])
                if _ready_sentences:
                    detected_langs = set(detect_language(s) for s in _ready_sentences[:50])
                    model_langs    = set(bcfg.get("langs", ["en"]))
                    unsupported    = detected_langs - model_langs
                    if unsupported:
                        st.warning(
                            f"⚠️ Some sentences appear to be in "
                            f"{[LANG_LABEL.get(l,l) for l in unsupported]} "
                            f"which this model doesn't natively support. "
                            f"The model's default language will be used as fallback."
                        )

            # ── Options ───────────────────────────────────────
            st.markdown('<p class="section-header">Step 3 — Options</p>',
                        unsafe_allow_html=True)

            opt_col1, opt_col2, opt_col3 = st.columns(3)
            compute_wer_batch = opt_col1.checkbox(
                "Compute WER/CER metrics",
                value=True,
                key="batch_compute_wer",
                help="Slower — runs Whisper ASR on each output",
            )
            skip_errors = opt_col2.checkbox(
                "Skip errors & continue",
                value=True,
                key="batch_skip_errors",
            )
            save_batch_json = opt_col3.checkbox(
                "Save results JSON",
                value=True,
                key="batch_save_json",
            )
            st.caption(f"📁 Outputs will be saved to: `{BATCH_OUTPUT_DIR}`")

            # ── Run Batch ─────────────────────────────────────
            st.markdown('<p class="section-header">Step 4 — Generate</p>',
                        unsafe_allow_html=True)

            _ready_sentences                = st.session_state.get("_batch_sentences_ready", [])
            _batch_files_ok, _batch_missing = get_model_file_status(batch_model)

            if _batch_missing:
                st.markdown(
                    f'<div class="missing-banner">⚠️ Model has {len(_batch_missing)} '
                    f'missing file(s). Fix in <b>📁 Local Files</b> tab.</div>',
                    unsafe_allow_html=True,
                )

            _batch_disabled = not _batch_files_ok or not _ready_sentences

            batch_run_btn = st.button(
                f"▶ Generate {len(_ready_sentences)} Audio File(s)"
                if _ready_sentences else "▶ Generate (load sentences first)",
                type="primary",
                disabled=_batch_disabled,
                key="batch_run_btn",
                use_container_width=True,
            )

            if not _ready_sentences:
                st.caption("Load sentences above to enable generation.")

            if batch_run_btn and _ready_sentences and batch_model:
                batch_sentences = _ready_sentences
                total           = len(batch_sentences)
                progress_bar    = st.progress(0, text="Starting batch generation…")
                status_text     = st.empty()
                live_table_ph   = st.empty()
                results_so_far  = []

                BATCH_OUTPUT_DIR = BASE_DIR / "batch_outputs"
                BATCH_OUTPUT_DIR.mkdir(exist_ok=True)

                for i, sentence in enumerate(batch_sentences, start=1):
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    progress_bar.progress(
                        (i - 1) / total,
                        text=f"Generating {i}/{total}: {sentence[:60]}…",
                    )
                    status_text.info(f"🎙 [{i}/{total}] `{sentence[:80]}`")

                    det_lang = detect_language(sentence)
                    bcfg_run = MODEL_REGISTRY[batch_model]
                    if det_lang not in bcfg_run.get("langs", ["en"]):
                        det_lang = bcfg_run.get("langs", ["en"])[0]

                    model_slug = batch_model.lower().replace(" ","_").replace("/","_").replace("(","").replace(")","")[:30]
                    out_filename = f"{i}_{model_slug}_{det_lang}.wav"
                    out_path = BATCH_OUTPUT_DIR / out_filename

                    try:
                        (wav_path, audio, sr, gen_time, duration,
                         rtf, cpu_pct, mem_mb, det_lang_out) = generate(batch_model, sentence)

                        import shutil
                        shutil.copy2(wav_path, str(out_path))

                        if compute_wer_batch:
                            metrics = compute_metrics(
                                str(out_path), sentence,
                                gen_time, duration, rtf, cpu_pct, mem_mb,
                            )
                        else:
                            metrics = {
                                "asr_transcript": "", "wer": -1, "cer": -1,
                                "mos_proxy": -1,
                                "rtf":               round(rtf,      4),
                                "generation_time_s": round(gen_time, 4),
                                "audio_duration_s":  round(duration, 4),
                                "cpu_model_pct":     round(cpu_pct,  2),
                                "memory_mb":         round(mem_mb,   2),
                            }

                        results_so_far.append({
                            "index":      i,
                            "input_text": sentence,
                            "lang":       det_lang_out,
                            "wav_path":   str(out_path),
                            "filename":   out_filename,
                            "status":     "✅ OK",
                            "error":      "",
                            **metrics,
                        })

                    except Exception as e:
                        err_msg = str(e)[:120]
                        results_so_far.append({
                            "index":             i,
                            "input_text":        sentence,
                            "lang":              det_lang,
                            "wav_path":          "",
                            "filename":          out_filename,
                            "status":            "❌ Error",
                            "error":             err_msg,
                            "wer": -1, "cer": -1, "mos_proxy": -1, "rtf": -1,
                            "generation_time_s": -1, "audio_duration_s": -1,
                            "cpu_model_pct": -1, "memory_mb": -1,
                            "asr_transcript": "",
                        })
                        if not skip_errors:
                            progress_bar.progress(i / total, text="Stopped due to error.")
                            status_text.error(f"❌ Error on sentence {i}: {err_msg}")
                            break

                    if i % 5 == 0 or i == total:
                        live_df = pd.DataFrame(results_so_far)[[
                            "index", "filename", "status",
                            "generation_time_s", "rtf", "wer",
                        ]]
                        live_table_ph.dataframe(live_df, use_container_width=True, hide_index=True)

                progress_bar.progress(1.0, text=f"Done! {total} sentence(s) processed.")
                status_text.empty()

                ok_count  = sum(1 for r in results_so_far if r["status"] == "✅ OK")
                err_count = sum(1 for r in results_so_far if r["status"] == "❌ Error")
                st.success(f"✅ Batch complete — **{ok_count}** succeeded, **{err_count}** failed.")

                final_df = pd.DataFrame(results_so_far)
                st.session_state["batch_results_df"]   = final_df
                st.session_state["batch_results_list"] = results_so_far

                if save_batch_json:
                    model_slug_json = batch_model.lower().replace(" ","_").replace("/","_").replace("(","").replace(")","")[:30]
                    batch_json_path = (
                        BATCH_OUTPUT_DIR
                        / f"batch_{model_slug_json}_{int(time.time())}.json"
                    )
                    batch_json_path.write_text(
                        json.dumps(results_so_far, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    st.caption(f"💾 Results saved to `{batch_json_path.name}`")

            # ── Results Display ───────────────────────────────
            if "batch_results_df" in st.session_state:
                final_df    = st.session_state["batch_results_df"]
                results_all = st.session_state["batch_results_list"]

                st.divider()
                st.markdown('<p class="section-header">📊 Results</p>', unsafe_allow_html=True)

                ok_rows = [r for r in results_all if r["status"] == "✅ OK"]
                if ok_rows:
                    num_fields = ["generation_time_s", "rtf", "wer", "cer", "mos_proxy"]
                    s_col = st.columns(len(num_fields))
                    for col, field in zip(s_col, num_fields):
                        vals = [r[field] for r in ok_rows if r.get(field, -1) >= 0]
                        avg  = sum(vals) / len(vals) if vals else None
                        col.metric(
                            field.replace("_", " ").upper(),
                            f"{avg:.3f}" if avg is not None else "—",
                        )

                display_cols = ["index", "filename", "lang", "status",
                                "generation_time_s", "rtf", "wer", "error"]
                show_cols    = [c for c in display_cols if c in final_df.columns]
                st.dataframe(final_df[show_cols], use_container_width=True, hide_index=True)

                st.markdown('<p class="section-header">🔊 Audio Playback</p>',
                            unsafe_allow_html=True)
                show_players = st.checkbox(
                    "Show audio players for all outputs",
                    value=False,
                    key="batch_show_players",
                )

                for r in results_all:
                    if r["status"] != "✅ OK":
                        continue
                    label = f"`{r['index']}.` {r['input_text'][:70]}"
                    if show_players:
                        col_l, col_r = st.columns([3, 2])
                        col_l.markdown(label)
                        wav_f = r.get("wav_path", "")
                        if wav_f and Path(wav_f).exists():
                            col_r.audio(wav_f, format="audio/wav")
                        else:
                            col_r.caption("⚠️ File not found")
                    else:
                        with st.expander(label, expanded=False):
                            wav_f = r.get("wav_path", "")
                            if wav_f and Path(wav_f).exists():
                                st.audio(wav_f, format="audio/wav")
                            else:
                                st.caption("⚠️ File not found on disk.")
                            st.caption(
                                f"RTF: {r.get('rtf',-1):.3f} · "
                                f"WER: {r.get('wer',-1):.3f} · "
                                f"Gen: {r.get('generation_time_s',-1):.2f}s"
                            )

                st.divider()
                st.markdown('<p class="section-header">⬇️ Download Results</p>',
                            unsafe_allow_html=True)

                dl_col1, dl_col2 = st.columns(2)
                csv_bytes = final_df.drop(columns=["wav_path"], errors="ignore")\
                                     .to_csv(index=False).encode("utf-8")
                dl_col1.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name=f"batch_results_{int(time.time())}.csv",
                    mime="text/csv",
                    key="batch_dl_csv",
                )
                json_bytes = json.dumps(results_all, indent=2,
                                        ensure_ascii=False).encode("utf-8")
                dl_col2.download_button(
                    "⬇️ Download Results JSON",
                    data=json_bytes,
                    file_name=f"batch_results_{int(time.time())}.json",
                    mime="application/json",
                    key="batch_dl_json",
                )

                st.markdown("**Download all audio files as ZIP:**")
                ok_wavs = [r["wav_path"] for r in results_all
                           if r["status"] == "✅ OK" and Path(r["wav_path"]).exists()]
                if ok_wavs:
                    import io
                    import zipfile as _zf
                    zip_buf = io.BytesIO()
                    with _zf.ZipFile(zip_buf, "w", _zf.ZIP_DEFLATED) as zf:
                        for wav_file in ok_wavs:
                            zf.write(wav_file, Path(wav_file).name)
                    zip_buf.seek(0)
                    model_slug_zip = batch_model.lower().replace(" ","_").replace("/","_").replace("(","").replace(")","")[:30]
                    st.download_button(
                        f"⬇️ Download All {len(ok_wavs)} WAV Files (ZIP)",
                        data=zip_buf.getvalue(),
                        file_name=f"batch_audio_{model_slug_zip}_{int(time.time())}.zip",
                        mime="application/zip",
                        key="batch_dl_zip",
                    )
                else:
                    st.caption("No audio files available to ZIP.")

                if st.button("🗑 Clear Batch Results", key="batch_clear_results_btn"):
                    st.session_state.pop("batch_results_df", None)
                    st.session_state.pop("batch_results_list", None)
                    st.session_state.pop("_batch_sentences_ready", None)
                    st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════
with tab2:

    

    # 'filtered' is already cleaned in the sidebar — only usable models
    usable_filtered_cmp  = filtered
    excluded_filtered_cmp = _truly_excluded  # from sidebar scope

    # ── Excluded models notice ─────────────────────────────────
    if excluded_filtered_cmp:
        with st.expander(
            f"⚠️ {len(excluded_filtered_cmp)} model(s) hidden — "
            f"incomplete metadata or missing files",
            expanded=False,
        ):
            st.caption(
                "Fix these in **🔎 Model Inspector** or **📁 Local Files** tab."
            )
            for exc_name in excluded_filtered_cmp:
                reason_data = st.session_state.exclusion_reasons.get(exc_name, {})
                st.markdown(
                    f'<div class="reg-error" style="margin:4px 0;">'
                    f'🚫 <b>{exc_name}</b> — '
                    f'<span style="font-size:0.82rem;color:#fca5a5">'
                    f'{reason_data.get("reason","Unknown reason.")}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
                
    st.markdown('<p class="section-header">Select Models</p>', unsafe_allow_html=True)

    if not usable_filtered_cmp:
        st.error(
            "❌ No usable models available. "
            "Fix metadata in **🔎 Model Inspector** or add models via **📁 Local Files**."
        )
        st.stop()

    if len(usable_filtered_cmp) < 2:
        st.warning(
            f"⚠️ Only **{len(usable_filtered_cmp)}** usable model(s) available "
            f"for the current language filter. "
            f"Need at least 2 to compare. "
            f"Fix excluded models or change the language filter."
        )

    selected_models = st.multiselect(
        "Models",
        list(usable_filtered_cmp.keys()),
        default=list(usable_filtered_cmp.keys())[:min(2, len(usable_filtered_cmp))],
        label_visibility="collapsed",
        key="cmp_models",
    )

    st.markdown('<p class="section-header">Input Text</p>', unsafe_allow_html=True)
    cmp_text = st.text_area(
        "Text",
        height=100,
        placeholder="Enter text to run through all selected models…",
        label_visibility="collapsed",
        key="cmp_text",
    )
    if cmp_text.strip():
        st.caption(
            f"🔍 Detected: **{LANG_LABEL.get(detect_language(cmp_text), '?')}**"
        )

    # Show usability summary for selected models
    if selected_models:
        any_warn = False
        for sm in selected_models:
            u = get_model_usability(sm)
            if u["severity"] == "warning":
                any_warn = True
                st.caption(
                    f"⚠️ **{sm}** — some metadata incomplete "
                    f"({', '.join(u['unknown_meta'])}), but will attempt generation."
                )
        if any_warn:
            st.caption(
                "Fix incomplete metadata in **🔎 Model Inspector** for best results."
            )

    cmp_disabled = len(selected_models) < 2
    cmp_btn = st.button(
        "⚖️ Run Comparison",
        type="primary",
        key="cmp_btn",
        disabled=cmp_disabled,
    )
    if cmp_disabled:
        st.caption("Select at least 2 models to compare.")

    if cmp_btn and cmp_text.strip() and not cmp_disabled:
        comparison_results = []
        progress = st.progress(0, text="Starting…")

        for i, mname in enumerate(selected_models):
            progress.progress(i / len(selected_models), text=f"Running {mname}…")
            st.markdown(
                f'<div class="compare-header">🔄 {mname}</div>',
                unsafe_allow_html=True,
            )
            det_lang = detect_language(cmp_text)
            cfg      = MODEL_REGISTRY[mname]
            if det_lang not in cfg["langs"]:
                det_lang = cfg["langs"][0]

            try:
                (wav_path, audio, sr, gen_time, duration,
                 rtf, cpu_pct, mem_mb, det_lang) = generate(mname, cmp_text)
                metrics = compute_metrics(
                    wav_path, cmp_text, gen_time, duration, rtf, cpu_pct, mem_mb
                )
                st.audio(wav_path, format="audio/wav")
                render_metric_rows(metrics)
                save_result(mname, cmp_text, det_lang, wav_path, metrics)
                comparison_results.append({"Model": mname, **metrics})
            except Exception as e:
                st.error(f"❌ {mname}: {e}")
                comparison_results.append({"Model": mname, "error": str(e)})

            st.divider()

        progress.progress(1.0, text="Done!")

        if comparison_results:
            st.markdown(
                '<p class="section-header">Comparison Summary</p>',
                unsafe_allow_html=True,
            )
            cmp_df    = pd.DataFrame(
                [{k: v for k, v in r.items() if k != "asr_transcript"}
                 for r in comparison_results]
            )
            num_cols  = ["rtf", "wer", "cer", "mos_proxy", "generation_time_s"]
            avail_num = [c for c in num_cols if c in cmp_df.columns]

            def highlight_best(col):
                lower_better = col.name in ["rtf", "wer", "cer", "generation_time_s"]
                best = col.min() if lower_better else col.max()
                return [
                    "background-color:#1a3a1a;color:#5ef08a;font-weight:700"
                    if v == best else ""
                    for v in col
                ]

            st.dataframe(
                cmp_df.style.apply(highlight_best, subset=avail_num),
                use_container_width=True,
            )

            ch1, ch2 = st.columns(2)
            if "wer" in cmp_df.columns:
                ch1.markdown("**WER by Model**")
                ch1.bar_chart(cmp_df.set_index("Model")["wer"])
            if "rtf" in cmp_df.columns:
                ch2.markdown("**RTF by Model**")
                ch2.bar_chart(cmp_df.set_index("Model")["rtf"])


# ══════════════════════════════════════════════════════════════
# TAB 3 — LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Model Leaderboard</p>', unsafe_allow_html=True)
    st.caption("Score = 0.5×(1−WER) + 0.3×(1/(1+RTF)) + 0.2×MOS")
    if st.button("🔄 Refresh", key="lb_ref"):
        st.rerun()
    lb_df = compute_leaderboard()
    if lb_df.empty:
        st.info("No results yet. Generate audio first.")
    else:
        for _, row in lb_df.iterrows():
            rank  = int(row["rank"])
            medal = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
            wer_v = f"{row['wer']:.4f}"       if "wer"       in row and pd.notna(row["wer"])       else "—"
            rtf_v = f"{row['rtf']:.4f}"       if "rtf"       in row and pd.notna(row["rtf"])       else "—"
            mos_v = f"{row['mos_proxy']:.4f}" if "mos_proxy" in row and pd.notna(row["mos_proxy"]) else "—"
            st.markdown(
                f'<div class="compare-header">{medal} &nbsp; <b>{row["model_name"]}</b>'
                f' &nbsp;·&nbsp; Score: <code>{row["score"]:.4f}</code>'
                f' &nbsp;·&nbsp; WER: <code>{wer_v}</code>'
                f' &nbsp;·&nbsp; RTF: <code>{rtf_v}</code>'
                f' &nbsp;·&nbsp; MOS: <code>{mos_v}</code>'
                f' &nbsp;·&nbsp; Runs: <code>{int(row.get("runs",0))}</code></div>',
                unsafe_allow_html=True)
        st.divider()
        disp = ["rank","model_name","score","wer","cer","mos_proxy",
                "rtf","generation_time_s","cpu_model_pct","memory_mb","runs"]
        st.dataframe(lb_df[[c for c in disp if c in lb_df.columns]],
                     use_container_width=True, hide_index=True)
        st.bar_chart(lb_df.set_index("model_name")["score"])


# ══════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Past Runs</p>', unsafe_allow_html=True)
    all_results = load_all_json_results()
    if not all_results:
        st.info("No saved runs found.")
    else:
        hc1, hc2 = st.columns(2)
        all_m    = sorted(set(r.get("model_name","") for r in all_results))
        all_l    = sorted(set(r.get("lang","")       for r in all_results))
        filt_m   = hc1.multiselect("Filter by model",    all_m, default=all_m, key="hist_m")
        filt_l   = hc2.multiselect("Filter by language", all_l, default=all_l, key="hist_l")
        shown    = [r for r in all_results
                    if r.get("model_name","") in filt_m and r.get("lang","") in filt_l]
        st.caption(f"Showing {len(shown)} of {len(all_results)} runs")
        for r in shown:
            label = (f"**{r.get('model_name','?')}** | "
                     f"{LANG_LABEL.get(r.get('lang',''),r.get('lang',''))} | "
                     f"WER: {r.get('wer','?')} | RTF: {r.get('rtf','?')} | "
                     f"{r.get('timestamp','')[:16]}")
            with st.expander(label):
                st.write(f"**Input:** {r.get('input_text','')}")
                st.write(f"**Transcript:** _{r.get('asr_transcript','')}_")
                wav_f = r.get("output_wav","")
                if wav_f and Path(wav_f).exists():
                    st.audio(wav_f, format="audio/wav")
                else:
                    st.caption("⚠️ Audio file not found.")
                render_metric_rows({k: r.get(k,0) for k in
                                    ["wer","cer","mos_proxy","rtf","generation_time_s",
                                     "audio_duration_s","cpu_model_pct","memory_mb"]})
                with st.expander("Raw JSON"):
                    st.json({k:v for k,v in r.items() if not k.startswith("_")})


# ══════════════════════════════════════════════════════════════
# TAB 5 — IMPORT & VISUALIZE
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Import JSON Result File</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload JSON", type=["json"],
                                 label_visibility="collapsed", key="imp_upload")
    if uploaded:
        try:
            raw  = json.load(uploaded)
            data = raw if isinstance(raw, list) else [raw]
            df   = pd.DataFrame(data)
            st.success(f"Loaded {len(df)} record(s).")
            prev_cols = ["model_name","lang","wer","cer","mos_proxy","rtf",
                         "generation_time_s","asr_transcript"]
            st.dataframe(df[[c for c in prev_cols if c in df.columns]],
                         use_container_width=True)
            mkeys = ["wer","cer","mos_proxy","rtf","generation_time_s","cpu_model_pct","memory_mb"]
            avail = [k for k in mkeys if k in df.columns]
            for c in avail:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if avail:
                pick = st.selectbox("Metric to chart", avail, key="imp_metric")
                if "model_name" in df.columns:
                    st.bar_chart(df[["model_name",pick]].dropna().set_index("model_name")[pick])
                if len(avail) >= 2:
                    sc1, sc2 = st.columns(2)
                    xa = sc1.selectbox("X axis", avail, index=0, key="sc_x")
                    ya = sc2.selectbox("Y axis", avail, index=min(1,len(avail)-1), key="sc_y")
                    st.scatter_chart(df[["model_name",xa,ya]].dropna().set_index("model_name"))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download as CSV", csv,
                                file_name="tts_export.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
    else:
        st.info("Upload a `.json` file to visualize.")
        if st.button("📊 Load All Saved Results", key="load_all"):
            all_r = load_all_json_results()
            if not all_r:
                st.warning("No saved results found.")
            else:
                df_all = pd.DataFrame(all_r)
                keys   = ["wer","cer","mos_proxy","rtf","generation_time_s"]
                for k in keys:
                    if k in df_all.columns:
                        df_all[k] = pd.to_numeric(df_all[k], errors="coerce")
                avail = [k for k in keys if k in df_all.columns]
                if avail and "model_name" in df_all.columns:
                    pick = st.selectbox("Metric", avail, key="all_m")
                    st.bar_chart(df_all.groupby("model_name")[pick].mean().dropna())
                show_cols = [c for c in ["model_name","lang","wer","rtf","mos_proxy","timestamp"]
                             if c in df_all.columns]
                st.dataframe(df_all[show_cols], use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — MODEL INSPECTOR
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-header">🔎 Model Inspector</p>', unsafe_allow_html=True)
    st.caption(
        "Select any registered model to inspect its full configuration, "
        "metadata, CLI args, file status, and complete benchmark history "
        "across all input sentences."
    )

    all_model_names = sorted(MODEL_REGISTRY.keys())
    insp_model = st.selectbox(
        "Select model to inspect",
        all_model_names,
        label_visibility="collapsed",
        key="insp_model_select",
    )

    if insp_model:
        cfg  = MODEL_REGISTRY[insp_model]
        meta = cfg.get("meta", {})

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:

            langs_str = " · ".join(LANG_LABEL.get(l, l) for l in cfg.get("langs", []))

            # ── Run metadata inference ─────────────────────────────
            inference  = infer_model_metadata(insp_model, cfg)
            inf_meta   = inference["inferred_meta"]
            missing_f  = inference["missing_fields"]
            confidence = inference["confidence"]
            matched_k  = inference["matched_key"]
            is_subcomp = inference["is_subcomponent"]
            suggested  = inference["suggested_json"]

            # ── Sub-component warning ──────────────────────────────
            if is_subcomp:
                st.markdown(
                    '<div class="subcomp-error">'
                    '🚫 <b>This is a sub-component, not a standalone TTS model.</b><br>'
                    'Files like <code>duration_predictor</code>, <code>text_encoder</code>, '
                    '<code>vocos</code>, <code>lm_flow</code> are internal parts of '
                    'multi-file architectures. They cannot be used directly for TTS.<br><br>'
                    '👉 Delete this entry and register the correct parent model '
                    '(e.g. use <code>type: supertonic</code> for Supertonic).'
                    '</div>',
                    unsafe_allow_html=True,
                )

            installed  = cfg.get("installed", False)
            inst_badge = (
                "<span style='color:#5ef08a;font-weight:600'>✓ Installed</span>"
                if installed else
                "<span style='color:#fbbf24;font-weight:600'>⚠ Not fully installed</span>"
            )
            st.markdown(
                f'<div class="insp-header">'
                f'<div style="font-size:1.3rem;font-weight:700;color:#e8eaf0;margin-bottom:12px">'
                f'{insp_model}</div>'
                f'<div class="insp-kv"><span class="insp-key">Status</span>'
                f'<span class="insp-val">{inst_badge}</span></div>'
                f'<div class="insp-kv"><span class="insp-key">Type</span>'
                f'<span class="insp-val"><code>{cfg.get("type","—")}</code></span></div>'
                f'<div class="insp-kv"><span class="insp-key">Languages</span>'
                f'<span class="insp-val">{langs_str}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Metadata with inference indicators ────────────────
            st.markdown('<p class="section-header">📋 Metadata</p>', unsafe_allow_html=True)

            conf_color = {"high": "#5ef08a", "medium": "#fbbf24", "low": "#f87171"}
            conf_label = {
                "high":   "✓ Fully auto-filled",
                "medium": "~ Partially matched",
                "low":    "✗ No pattern matched",
            }
            if matched_k:
                st.caption(
                    f"Inference: <span style='color:{conf_color[confidence]};font-weight:600'>"
                    f"{conf_label[confidence]}</span> "
                    f"· matched pattern: <code>{matched_k}</code>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "<span style='color:#f87171'>⚠ No known model pattern matched — "
                    "metadata fields may be incomplete or wrong.</span>",
                    unsafe_allow_html=True,
                )

            orig_meta   = cfg.get("meta", {})
            meta_fields = [
                ("Provider",     "provider"),
                ("Parameters",   "parameters"),
                ("Compute Type", "compute_type"),
                ("Model Size",   "model_size"),
                ("License",      "license"),
                ("Best Use",     "best_use_case"),
            ]
            meta_html = '<div class="insp-section">'
            for label, key in meta_fields:
                orig_val    = orig_meta.get(key, "—")
                inf_val     = inf_meta.get(key, "—")
                was_unknown = orig_val in (None, "", "unknown", "UNKNOWN", "—", "?")
                was_filled  = was_unknown and inf_val != orig_val

                display_val = inf_val if was_filled else orig_val
                if was_filled:
                    val_html = (
                        f'<span style="color:#fbbf24">{display_val} '
                        f'<span style="font-size:0.65rem;opacity:0.8">(inferred)</span></span>'
                    )
                elif was_unknown:
                    val_html = (
                        f'<span style="color:#f87171">{orig_val} '
                        f'<span style="font-size:0.65rem">⚠ needs update</span></span>'
                    )
                else:
                    val_html = f'<span style="color:#e8eaf0">{display_val}</span>'

                meta_html += (
                    f'<div class="insp-kv">'
                    f'<span class="insp-key">{label}</span>'
                    f'<div class="insp-val">{val_html}</div></div>'
                )
            meta_html += "</div>"
            st.markdown(meta_html, unsafe_allow_html=True)

            # ── Missing metadata + Fix panel ──────────────────────
            if missing_f or is_subcomp:
                st.markdown(
                    '<p class="section-header">⚠️ Metadata Issues</p>',
                    unsafe_allow_html=True,
                )
                if missing_f:
                    items = "".join(f"<li><code>{f}</code></li>" for f in missing_f)
                    st.markdown(
                        f'<div class="reg-warn">Still unknown after inference:'
                        f'<ul>{items}</ul>'
                        f'Use the Fix panel below to correct them.</div>',
                        unsafe_allow_html=True,
                    )

                suggested_json_str = json.dumps(suggested, indent=2, ensure_ascii=False)

                with st.expander(
                    "🔧 Fix Metadata — Pre-filled JSON",
                    expanded=bool(missing_f),
                ):
                    st.caption(
                        "Auto-filled with known values. Review, then paste into "
                        "**➕ Add Model (JSON)** tab and click **Validate & Register**."
                    )
                    st.code(suggested_json_str, language="json")

                    fix_col1, fix_col2 = st.columns(2)
                    if fix_col1.button(
                        "📋 Send to Add Model Tab",
                        key=f"send_json_{insp_model[:20]}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["reg_json_input"] = suggested_json_str
                        st.success(
                            "✅ Copied! Switch to **➕ Add Model (JSON)** tab "
                            "and click **Validate & Register**."
                        )
                    fix_col2.download_button(
                        "⬇️ Download JSON",
                        data=suggested_json_str.encode("utf-8"),
                        file_name=f"{insp_model[:30].replace(' ','_')}_fixed.json",
                        mime="application/json",
                        key=f"dl_fix_{insp_model[:20]}",
                    )

                # Field-by-field guidance
                if missing_f:
                    field_guide = {
                        "provider":
                            "The org that published this model. "
                            "E.g. `K2-FSA`, `COQUI`, `META`, `AI4BHARAT`, `ICEFALL`.",
                        "parameters":
                            "Number of parameters. E.g. `~40M`, `~82M`, `~400M`. "
                            "Check the model's HuggingFace page.",
                        "compute_type":
                            "Precision & runtime. E.g. `FP32/ONNX`, `INT8/ONNX`, "
                            "`FP16/ONNX`. Hint: `int8` in filename → `INT8/ONNX`.",
                        "model_size":
                            "Total disk size of all model files. "
                            "E.g. `~37MB`, `~330MB`, `~3.75GB`.",
                        "license":
                            "Usage license. E.g. `MIT`, `Apache 2.0`, `CC-BY-NC 4.0`. "
                            "Check the model repo.",
                        "best_use_case":
                            "Short description. E.g. `GENERAL ENGLISH TTS`, "
                            "`FEMALE HINDI TTS`, `EDGE DEVICES`.",
                    }
                    with st.expander("📖 What should each unknown field contain?"):
                        for f in missing_f:
                            st.markdown(
                                f'<div class="insp-section" style="margin:6px 0;">'
                                f'<div style="color:#fbbf24;font-weight:600;'
                                f'font-size:0.85rem;margin-bottom:4px">'
                                f'<code>{f}</code></div>'
                                f'<div style="color:#c4c8e0;font-size:0.82rem">'
                                f'{field_guide.get(f,"No guidance available.")}'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )

            # ── Quick inline metadata editor ──────────────────────
            st.markdown(
                '<p class="section-header">✏️ Quick Metadata Edit</p>',
                unsafe_allow_html=True,
            )
            with st.expander("Edit metadata fields directly", expanded=False):
                st.caption("Edit and click Apply & Save — changes persist to disk.")
                qe_provider = st.text_input("Provider",
                    value=inf_meta.get("provider",""),
                    key=f"qe_prov_{insp_model[:20]}")
                qe_params   = st.text_input("Parameters",
                    value=inf_meta.get("parameters",""),
                    key=f"qe_param_{insp_model[:20]}")
                qe_compute  = st.text_input("Compute Type",
                    value=inf_meta.get("compute_type",""),
                    key=f"qe_comp_{insp_model[:20]}")
                qe_size     = st.text_input("Model Size",
                    value=inf_meta.get("model_size",""),
                    key=f"qe_size_{insp_model[:20]}")
                qe_license  = st.text_input("License",
                    value=inf_meta.get("license",""),
                    key=f"qe_lic_{insp_model[:20]}")
                qe_use      = st.text_input("Best Use Case",
                    value=inf_meta.get("best_use_case",""),
                    key=f"qe_use_{insp_model[:20]}")

                if st.button("💾 Apply & Save Metadata",
                             key=f"qe_save_{insp_model[:20]}",
                             type="primary", use_container_width=True):
                    new_meta = {
                        "provider":      qe_provider,
                        "parameters":    qe_params,
                        "compute_type":  qe_compute,
                        "model_size":    qe_size,
                        "license":       qe_license,
                        "best_use_case": qe_use,
                    }
                    st.session_state.model_registry[insp_model]["meta"] = new_meta
                    updated_cfg                  = dict(cfg)
                    updated_cfg["meta"]          = new_meta
                    updated_cfg["display_name"]  = insp_model
                    save_model_json_to_disk(updated_cfg, mark_installed=installed)
                    st.success(f"✅ Metadata saved for **{insp_model}**!")
                    st.rerun()

            args = cfg.get("args", [])
            st.markdown('<p class="section-header">⚙️ CLI Arguments</p>', unsafe_allow_html=True)
            if args:
                flag_html = '<div class="insp-section">'
                i = 0
                while i < len(args):
                    if args[i].startswith("--") and i + 1 < len(args) and not args[i+1].startswith("--"):
                        flag_html += (
                            f'<div class="insp-kv" style="margin:6px 0;">'
                            f'<span class="insp-flag-chip">{args[i]}</span>'
                            f'<span class="insp-val" style="font-size:0.78rem;word-break:break-all">'
                            f'{args[i+1]}</span></div>'
                        )
                        i += 2
                    else:
                        flag_html += (f'<span class="insp-arg-chip">{args[i]}</span>')
                        i += 1
                flag_html += "</div>"
                st.markdown(flag_html, unsafe_allow_html=True)
                with st.expander("📋 Copy raw args"):
                    st.code(" ".join(args), language="bash")
            else:
                st.caption("No CLI args (non-CLI model type).")

            st.markdown('<p class="section-header">📂 File Status</p>', unsafe_allow_html=True)
            if args:
                all_flags = {
                    "--vits-model","--vits-tokens","--vits-lexicon",
                    "--matcha-acoustic-model","--matcha-vocoder","--matcha-tokens",
                    "--kokoro-model","--kokoro-voices","--kokoro-tokens",
                    "--kitten-model","--kitten-voices","--kitten-tokens",
                    "--vits-data-dir","--matcha-data-dir","--kokoro-data-dir","--kitten-data-dir",
                }
                i = 0
                any_file_shown = False
                while i < len(args):
                    if args[i] in all_flags and i + 1 < len(args):
                        p      = Path(args[i + 1])
                        exists = p.exists()
                        css    = "file-ok" if exists else "file-miss"
                        icon   = "✓" if exists else "✗"
                        size   = ""
                        if exists and p.is_file():
                            try:
                                size = f" ({p.stat().st_size / 1e6:.1f} MB)"
                            except OSError:
                                pass
                        st.markdown(
                            f'<div class="{css}">{icon} <b>{args[i]}</b>'
                            f'<br><span style="font-size:0.75rem;opacity:0.8">'
                            f'{args[i+1]}{size}</span></div>',
                            unsafe_allow_html=True,
                        )
                        any_file_shown = True
                        i += 2
                    else:
                        i += 1
                if not any_file_shown:
                    st.caption("No file-path args to check.")
            else:
                st.caption("No file paths to verify for this model type.")

            st.markdown('<p class="section-header">📄 Full JSON Config</p>',
                        unsafe_allow_html=True)
            full_cfg = _full_model_config(insp_model)
            source   = full_cfg.pop("_source_file", None)
            if source:
                st.caption(f"Source file: `{source}`")
            else:
                st.caption("Source: hardcoded default (not in `./models/`)")

            with st.expander("View / Copy Full Config JSON", expanded=False):
                st.json(full_cfg)
                json_bytes = json.dumps(full_cfg, indent=2,
                                        ensure_ascii=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download config JSON",
                    data=json_bytes,
                    file_name=f"{insp_model[:40].replace(' ','_')}_config.json",
                    mime="application/json",
                    key=f"dl_cfg_{insp_model[:30]}",
                )

        with right_col:
            st.markdown('<p class="section-header">📊 Benchmark History</p>',
                        unsafe_allow_html=True)

            bench_df = get_model_benchmark_history(insp_model)

            if bench_df.empty:
                st.markdown(
                    '<div class="no-data-box">'
                    '📭 No benchmark runs found for this model yet.<br><br>'
                    'Generate some audio with this model in the '
                    '<b>🎙 Generate</b> or <b>⚖️ Compare</b> tab '
                    'to populate metrics here.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                n_runs = len(bench_df)
                st.caption(f"**{n_runs}** benchmark run(s) found")

                st.markdown('<p class="section-header">Aggregate Averages</p>',
                            unsafe_allow_html=True)

                def safe_mean(col):
                    vals = bench_df[col].dropna()
                    vals = vals[vals >= 0] if col != "mos_proxy" else vals
                    return vals.mean() if len(vals) > 0 else None

                avg_wer  = safe_mean("wer")
                avg_cer  = safe_mean("cer")
                avg_mos  = safe_mean("mos_proxy")
                avg_rtf  = safe_mean("rtf")
                avg_gen  = safe_mean("generation_time_s")
                avg_dur  = safe_mean("audio_duration_s")
                avg_cpu  = safe_mean("cpu_model_pct")
                avg_mem  = safe_mean("memory_mb")

                def fmt(v, decimals=3):
                    return f"{v:.{decimals}f}" if v is not None else "—"

                summary_html = '<div class="insp-metric-row">'
                for label, val, decs in [
                    ("Avg WER",      avg_wer,  3),
                    ("Avg CER",      avg_cer,  3),
                    ("Avg MOS",      avg_mos,  3),
                    ("Avg RTF",      avg_rtf,  3),
                    ("Avg Gen (s)",  avg_gen,  2),
                    ("Avg Dur (s)",  avg_dur,  2),
                    ("Avg CPU %",    avg_cpu,  1),
                    ("Avg RAM (MB)", avg_mem,  0),
                ]:
                    summary_html += (
                        f'<div class="insp-mini-card">'
                        f'<div class="insp-mini-label">{label}</div>'
                        f'<div class="insp-mini-val">{fmt(val, decs)}</div>'
                        f'</div>'
                    )
                summary_html += "</div>"
                st.markdown(summary_html, unsafe_allow_html=True)

                st.markdown('<p class="section-header">Metric Trends Over Runs</p>',
                            unsafe_allow_html=True)

                chart_col1, chart_col2 = st.columns(2)

                wer_data = bench_df[["wer"]].copy().reset_index(drop=True)
                wer_data.index = wer_data.index + 1
                if wer_data["wer"].notna().any():
                    chart_col1.markdown("**WER per run**")
                    chart_col1.line_chart(wer_data["wer"].dropna())

                rtf_data = bench_df[["rtf"]].copy().reset_index(drop=True)
                rtf_data.index = rtf_data.index + 1
                if rtf_data["rtf"].notna().any():
                    chart_col2.markdown("**RTF per run**")
                    chart_col2.line_chart(rtf_data["rtf"].dropna())

                mos_data = bench_df[["mos_proxy"]].copy().reset_index(drop=True)
                mos_data.index = mos_data.index + 1
                if mos_data["mos_proxy"].notna().any():
                    chart_col1.markdown("**MOS Proxy per run**")
                    chart_col1.line_chart(mos_data["mos_proxy"].dropna())

                gen_data = bench_df[["generation_time_s"]].copy().reset_index(drop=True)
                gen_data.index = gen_data.index + 1
                if gen_data["generation_time_s"].notna().any():
                    chart_col2.markdown("**Generation time (s)**")
                    chart_col2.line_chart(gen_data["generation_time_s"].dropna())

                st.markdown('<p class="section-header">Per-Sentence Results</p>',
                            unsafe_allow_html=True)

                display_cols = ["timestamp", "lang", "wer", "cer", "mos_proxy",
                                "rtf", "generation_time_s", "audio_duration_s",
                                "cpu_model_pct", "memory_mb"]
                show_df = bench_df[[c for c in display_cols if c in bench_df.columns]].copy()
                show_df = show_df.rename(columns={
                    "generation_time_s": "gen_s",
                    "audio_duration_s":  "dur_s",
                    "cpu_model_pct":     "cpu%",
                    "memory_mb":         "ram_mb",
                    "mos_proxy":         "mos",
                })

                def color_wer(val):
                    if pd.isna(val) or val < 0: return ""
                    if val < 0.2: return "color:#5ef08a"
                    if val < 0.5: return "color:#fbbf24"
                    return "color:#f87171"

                def color_rtf(val):
                    if pd.isna(val) or val < 0: return ""
                    if val < 0.5: return "color:#5ef08a"
                    if val < 1.0: return "color:#fbbf24"
                    return "color:#f87171"

                styled = show_df.style
                if "wer" in show_df.columns:
                    styled = styled.applymap(color_wer, subset=["wer"])
                if "rtf" in show_df.columns:
                    styled = styled.applymap(color_rtf, subset=["rtf"])

                for num_col in ["wer","cer","mos","rtf","gen_s","dur_s","cpu%","ram_mb"]:
                    if num_col in show_df.columns:
                        styled = styled.format({num_col: lambda x: f"{x:.3f}" if pd.notna(x) else "—"})

                st.dataframe(styled, use_container_width=True, hide_index=True)

                st.markdown('<p class="section-header">Run Details (with audio)</p>',
                            unsafe_allow_html=True)

                show_audio = st.checkbox("Show audio players for each run",
                                          value=False, key="insp_audio_toggle")

                for idx, row in bench_df.iterrows():
                    wer_v  = f"{row['wer']:.4f}"  if pd.notna(row.get("wer"))  else "—"
                    rtf_v  = f"{row['rtf']:.4f}"  if pd.notna(row.get("rtf"))  else "—"
                    mos_v  = f"{row['mos_proxy']:.4f}" if pd.notna(row.get("mos_proxy")) else "—"
                    ts     = row.get("timestamp", "")[:16]
                    lang_v = LANG_LABEL.get(row.get("lang",""), row.get("lang",""))
                    text_v = str(row.get("input_text",""))
                    label  = (f"Run {idx+1} — {ts} | {lang_v} | "
                              f"WER: {wer_v} | RTF: {rtf_v} | MOS: {mos_v}")

                    with st.expander(label, expanded=False):
                        st.markdown(
                            f'<div class="bench-row">'
                            f'<b>Input text</b>'
                            f'<div class="bench-text">{text_v}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        asr_v = str(row.get("asr_transcript", ""))
                        if asr_v:
                            st.markdown(
                                f'<div class="bench-row">'
                                f'<b>Whisper transcript</b>'
                                f'<div class="bench-text">{asr_v}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        mini_html = '<div class="insp-metric-row">'
                        for lbl, key, decs in [
                            ("WER",     "wer",              3),
                            ("CER",     "cer",              3),
                            ("MOS",     "mos_proxy",        3),
                            ("RTF",     "rtf",              3),
                            ("Gen (s)", "generation_time_s",2),
                            ("Dur (s)", "audio_duration_s", 2),
                            ("CPU %",   "cpu_model_pct",    1),
                            ("RAM MB",  "memory_mb",        0),
                        ]:
                            v = row.get(key)
                            mini_html += (
                                f'<div class="insp-mini-card">'
                                f'<div class="insp-mini-label">{lbl}</div>'
                                f'<div class="insp-mini-val">'
                                f'{f"{v:.{decs}f}" if pd.notna(v) and v is not None else "—"}'
                                f'</div></div>'
                            )
                        mini_html += "</div>"
                        st.markdown(mini_html, unsafe_allow_html=True)

                        if show_audio:
                            wav_f = row.get("output_wav","")
                            if wav_f and Path(wav_f).exists():
                                st.audio(wav_f, format="audio/wav")
                            else:
                                st.caption("⚠️ Audio file not found on disk.")

                        with st.expander("Raw JSON for this run"):
                            raw_row = {k: v for k, v in row.items()
                                       if not k.startswith("_")}
                            st.json(raw_row)

                st.divider()
                export_col1, export_col2 = st.columns(2)
                csv_bytes = bench_df.drop(columns=["_jf"], errors="ignore")\
                                    .to_csv(index=False).encode("utf-8")
                export_col1.download_button(
                    "⬇️ Export all runs as CSV",
                    data=csv_bytes,
                    file_name=f"{insp_model[:30].replace(' ','_')}_benchmarks.csv",
                    mime="text/csv",
                    key=f"dl_bench_csv_{insp_model[:20]}",
                )
                json_export = bench_df.drop(columns=["_jf"], errors="ignore")\
                                      .to_json(orient="records", indent=2).encode("utf-8")
                export_col2.download_button(
                    "⬇️ Export all runs as JSON",
                    data=json_export,
                    file_name=f"{insp_model[:30].replace(' ','_')}_benchmarks.json",
                    mime="application/json",
                    key=f"dl_bench_json_{insp_model[:20]}",
                )


# ══════════════════════════════════════════════════════════════
# TAB 7 — LOCAL FILE REGISTRATION  (ENHANCED)
# ══════════════════════════════════════════════════════════════
with tab7:

    # ── Auto-scan trigger after extraction ────────────────────
    # If extraction happened in a previous interaction, automatically run scan
    if st.session_state.get("trigger_auto_scan", False):
        st.session_state.trigger_auto_scan = False
        with st.spinner("🔍 Auto-scanning for new model files…"):
            auto_candidates = scan_local_models(BASE_DIR)
        st.session_state.scan_results = auto_candidates
        st.session_state.scan_done    = True

    # ══════════════════════════════════════════════════════════
    # SECTION 0 — Introduction & Workflow Guide
    # ══════════════════════════════════════════════════════════
    st.markdown('<p class="section-header">📁 Local File Registration</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="offline-box">'
        '🔒 <b>Offline / Air-Gapped / Corporate Network Mode</b><br>'
        'Use this tab to register TTS models from local files — '
        'no internet connection required for extraction and registration. '
        'Downloads are attempted only when you explicitly click "Download Missing Files".'
        '</div>',
        unsafe_allow_html=True,
    )

    # Workflow steps as styled cards
    st.markdown('<p class="section-header">📋 Typical Workflow</p>', unsafe_allow_html=True)
    steps = [
        ("1", "Download model archives on a personal device or via VPN "
              "(see download links below)."),
        ("2", f"Copy <code>.tar.bz2</code> / <code>.zip</code> archives into: "
              f"<code>{BASE_DIR}</code>"),
        ("3", "Use <b>Section A</b> to extract archives, or <b>Section B</b> to upload directly."),
        ("4", "Click <b>🔍 Scan for Model Files</b> in <b>Section C</b> — "
              "new models are auto-detected and offered for registration."),
        ("5", "Use <b>Section E — Verify & Download</b> to check which models "
              "have all required files, and download missing ones if available."),
    ]
    for num, text in steps:
        st.markdown(
            f'<div class="workflow-step">'
            f'<div class="workflow-step-num">{num}</div>'
            f'<div class="workflow-step-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📋 Download links for personal device"):
        st.code("""# Copy any URL into a browser on a non-corporate machine
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-pratham-medium.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-priyamvada-medium.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-rohan-medium.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2""",
                language="text")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION A — Extract Pre-Existing Archives
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">A</span> Extract Archive</p>',
        unsafe_allow_html=True,
    )

    archive_extensions = {".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip", ".tar"}
    found_archives     = sorted(
        [f for f in BASE_DIR.iterdir()
         if f.is_file() and any(f.name.lower().endswith(e) for e in archive_extensions)],
        key=lambda f: f.stat().st_size,
    )

    if found_archives:
        st.caption(f"📦 Found **{len(found_archives)}** archive(s) in `{BASE_DIR}`")

        # Show archive cards with size info
        for arch in found_archives:
            try:
                size_mb = arch.stat().st_size / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"
            except OSError:
                size_str = "?"
            st.markdown(
                f'<div class="archive-card">📦 <b>{arch.name}</b> · {size_str}</div>',
                unsafe_allow_html=True,
            )

        chosen_arc = st.selectbox(
            "Select archive to extract",
            [a.name for a in found_archives],
            key="arc_select",
        )
        arc_dest   = st.text_input(
            "Extract to directory",
            value=str(BASE_DIR),
            help="Destination folder. Leave as-is to extract into the app directory.",
            key="arc_dest",
        )

        arc_status = st.empty()

        if st.button("📦 Extract & Auto-Scan", key="do_extract",
                     type="primary", use_container_width=True):
            ok = extract_local_archive(BASE_DIR / chosen_arc, Path(arc_dest), arc_status)
            if ok:
                # Auto-scan immediately after successful extraction
                with st.spinner("🔍 Scanning for new model files after extraction…"):
                    new_candidates = scan_local_models(BASE_DIR)
                st.session_state.scan_results = new_candidates
                st.session_state.scan_done    = True
                # Invalidate file verification cache so re-verify picks up new files
                st.session_state.file_verification_cache.clear()
                arc_status.success(
                    f"✅ Extracted successfully! "
                    f"Found **{len(new_candidates)}** model(s) — see Section C below."
                )
    else:
        st.markdown(
            f'<div class="reg-warn">⚠️ No archives found in <code>{BASE_DIR}</code>. '
            f'Copy a <code>.tar.bz2</code> or <code>.zip</code> file there, '
            f'then refresh this page.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION B — Upload Archive via Browser
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">B</span> Upload Archive via Browser</p>',
        unsafe_allow_html=True,
    )
    st.caption("Max ~200 MB per upload. For larger files, copy directly into the app folder.")

    uploaded_arc = st.file_uploader(
        "Upload model archive",
        type=["gz", "bz2", "zip", "xz", "tar"],
        label_visibility="collapsed",
        key="arc_upload",
    )

    if uploaded_arc is not None:
        save_path     = BASE_DIR / uploaded_arc.name
        upload_status = st.empty()

        if not save_path.exists():
            upload_status.info(f"💾 Saving `{uploaded_arc.name}` to disk…")
            with open(save_path, "wb") as f:
                f.write(uploaded_arc.getbuffer())
            upload_status.success(
                f"✅ Saved `{uploaded_arc.name}` ({save_path.stat().st_size/1e6:.1f} MB)"
            )

        if st.button("📦 Extract Uploaded Archive & Scan", key="extract_uploaded",
                     type="primary", use_container_width=True):
            upload_extract_status = st.empty()
            ok = extract_local_archive(save_path, BASE_DIR, upload_extract_status)
            if ok:
                with st.spinner("🔍 Scanning for new model files…"):
                    new_candidates = scan_local_models(BASE_DIR)
                st.session_state.scan_results = new_candidates
                st.session_state.scan_done    = True
                st.session_state.file_verification_cache.clear()
                upload_extract_status.success(
                    f"✅ Extracted & scanned! Found **{len(new_candidates)}** model(s) — "
                    f"see Section C to register."
                )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION C — Scan & Register
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">C</span> Scan & Register Models</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Scans the app directory for `.onnx` model files ≥ 1 MB, "
        "auto-detects architecture and language, and offers registration."
    )

    sc1, sc2 = st.columns(2)
    do_scan = sc1.button(
        "🔍 Scan for Model Files",
        key="do_scan",
        type="primary",
        use_container_width=True,
    )
    if sc2.button("🗑 Clear Scan Results", key="clear_scan", use_container_width=True):
        st.session_state.scan_results = []
        st.session_state.scan_done    = False
        st.rerun()

    if do_scan:
        with st.spinner(f"🔍 Scanning `{BASE_DIR}` for ONNX model files…"):
            candidates = scan_local_models(BASE_DIR)
        st.session_state.scan_results = candidates
        st.session_state.scan_done    = True

    # ── Display scan results ───────────────────────────────────
    if st.session_state.scan_done:
        candidates = st.session_state.scan_results

        if not candidates:
            st.warning(
                "⚠️ No ONNX model files found (≥ 1 MB). "
                "Extract an archive first using Section A or B above."
            )
        else:
            # Partition into already-registered vs new
            already    = [c for c in candidates
                          if c["display_name"] in st.session_state.model_registry]
            new_cands  = [c for c in candidates
                          if c["display_name"] not in st.session_state.model_registry]

            # Summary banner
            st.markdown(
                f'<div class="reg-box">'
                f'✅ Scan complete — found <b>{len(candidates)}</b> model(s): '
                f'<b style="color:#5ef08a">{len(new_cands)} new</b> · '
                f'<b style="color:#636678">{len(already)} already registered</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Already-registered models (collapsed list) ─────
            if already:
                with st.expander(
                    f"📋 {len(already)} already-registered model(s) — no action needed",
                    expanded=False,
                ):
                    for c in already:
                        lang_str = " · ".join(LANG_LABEL.get(l, l) for l in c["langs"])
                        st.markdown(
                            f'<div class="scan-card-ok">'
                            f'✓ <b>{c["display_name"]}</b> — {lang_str}'
                            f'<br><span style="font-size:0.78rem;color:#636678;">'
                            f'{c.get("_onnx_path","")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── New models ─────────────────────────────────────
            if new_cands:
                # Batch register button
                if st.button(
                    f"⚡ Register All {len(new_cands)} New Model(s)",
                    key="register_all",
                    type="primary",
                    use_container_width=True,
                ):
                    for c in new_cands:
                        register_model_in_session(c)
                        save_model_json_to_disk(c, mark_installed=True)
                    st.success(f"✅ Registered {len(new_cands)} model(s)!")
                    st.balloons()
                    st.session_state.scan_results = []
                    st.session_state.scan_done    = False
                    st.session_state.file_verification_cache.clear()
                    st.rerun()

                st.caption(
                    "Or review and customise each model individually before registering:"
                )

                # Individual model cards
                for idx, c in enumerate(new_cands):
                    name        = c["display_name"]
                    lang_str    = " · ".join(LANG_LABEL.get(l, l) for l in c["langs"])
                    args        = c.get("args", [])
                    onnx_path   = c.get("_onnx_path", "")
                    size_str    = c.get("meta", {}).get("model_size", "?")

                    with st.expander(
                        f"🆕 {name} · {lang_str} · {size_str}",
                        expanded=False,
                    ):
                        # Show auto-detected info
                        st.markdown(
                            f'<div class="scan-card">'
                            f'<b>Auto-detected info</b><br>'
                            f'Architecture: <code>{c.get("meta",{}).get("compute_type","?")}</code> · '
                            f'Provider: <code>{c.get("meta",{}).get("provider","?")}</code><br>'
                            f'ONNX path: <code style="font-size:0.75rem">{onnx_path}</code>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Show proposed args
                        if args:
                            st.code(" ".join(args), language="bash")

                        # Allow user to customise name and language
                        ind_col1, ind_col2 = st.columns(2)
                        new_name = ind_col1.text_input(
                            "Display name",
                            value=name,
                            key=f"ind_name_{idx}",
                        )
                        lang_opts = ["en", "hi", "mar"]
                        cur_lang  = c["langs"][0] if c["langs"] else "en"
                        lang_idx  = lang_opts.index(cur_lang) if cur_lang in lang_opts else 0
                        lang_ov   = ind_col2.selectbox(
                            "Language",
                            lang_opts,
                            index=lang_idx,
                            key=f"ind_lang_{idx}",
                        )

                        if st.button(
                            f"✅ Register This Model",
                            key=f"ind_reg_{idx}",
                            type="primary",
                            use_container_width=True,
                        ):
                            fc = dict(c)
                            fc["display_name"] = new_name
                            fc["langs"]        = [lang_ov]
                            register_model_in_session(fc)
                            save_model_json_to_disk(fc, mark_installed=True)
                            st.success(f"✅ **{new_name}** registered!")
                            # Remove from scan results so it won't show again
                            st.session_state.scan_results = [
                                x for x in st.session_state.scan_results
                                if x["display_name"] != name
                            ]
                            st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION D — Manual Path Registration
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">D</span> Manual Path Registration</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Point directly to an `.onnx` file if the auto-scanner didn't pick it up."
    )

    manual_onnx = st.text_input(
        "Full path to .onnx file",
        placeholder="./my-model/model.onnx",
        key="manual_onnx_path",
    )
    manual_name = st.text_input(
        "Display name",
        placeholder="My Custom Model",
        key="manual_model_name",
    )
    man_col1, man_col2, man_col3 = st.columns(3)
    manual_lang = man_col1.selectbox("Language", ["en", "hi", "mar"], key="manual_lang")
    manual_type = man_col2.selectbox(
        "Architecture",
        ["vits", "vits_lexicon", "kokoro_onnx", "kitten", "matcha"],
        key="manual_type",
    )
    manual_sid  = man_col3.number_input(
        "Speaker ID (--sid)", min_value=0, max_value=999, value=0, key="manual_sid"
    )

    prev_col1, prev_col2 = st.columns(2)
    if prev_col1.button("🔍 Preview Args", key="manual_preview", use_container_width=True):
        if manual_onnx and Path(manual_onnx).exists():
            built = _build_cli_args_from_scan(Path(manual_onnx), manual_type, sid=int(manual_sid))
            st.code(" ".join(built), language="bash")
            st.session_state["manual_built_args"] = built
        elif manual_onnx:
            st.error(f"❌ File not found: `{manual_onnx}`")
        else:
            st.warning("Enter a path first.")

    if prev_col2.button(
        "✅ Register Manual Model",
        key="manual_register",
        type="primary",
        use_container_width=True,
    ):
        if not manual_onnx or not manual_name:
            st.warning("Enter both path and display name.")
        elif not Path(manual_onnx).exists():
            st.error(f"❌ File not found: `{manual_onnx}`")
        else:
            onnx_p = Path(manual_onnx)
            built  = st.session_state.get(
                "manual_built_args",
                _build_cli_args_from_scan(onnx_p, manual_type, sid=int(manual_sid)),
            )
            # Verify files
            missing_manual = check_required_files(built)
            all_ok_manual  = len(missing_manual) == 0
            try:
                size_str = f"~{onnx_p.stat().st_size/(1024*1024):.0f}MB"
            except OSError:
                size_str = "unknown"
            mcfg = {
                "display_name": manual_name,
                "type":         "cli",
                "langs":        [manual_lang],
                "args":         built,
                "installed":    all_ok_manual,
                "downloads":    [],
                "sherpa_config":{},
                "meta": {
                    "provider":      "LOCAL",
                    "parameters":    "unknown",
                    "compute_type":  "ONNX",
                    "model_size":    size_str,
                    "license":       "unknown",
                    "best_use_case": f"{manual_lang.upper()} TTS",
                },
            }
            register_model_in_session(mcfg)
            save_model_json_to_disk(mcfg, mark_installed=all_ok_manual)

            if all_ok_manual:
                st.success(f"✅ **{manual_name}** registered — all files present!")
            else:
                st.warning(
                    f"⚠️ **{manual_name}** registered but {len(missing_manual)} "
                    f"file(s) are missing. Check Section E."
                )
                for flag, path in missing_manual:
                    st.markdown(
                        f'<div class="file-miss">✗ {flag} → {path}</div>',
                        unsafe_allow_html=True,
                    )
            st.balloons()

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION E — Verify & Download Model Files
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">E</span> Verify & Download Model Files</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Check which registered models have all required files on disk. "
        "Download missing files if URLs are available, or follow the "
        "manual placement instructions."
    )

    # Re-verify button
    verify_col1, verify_col2 = st.columns([3, 1])
    if verify_col1.button(
        "🔄 Re-verify All Model Files",
        key="reverify_all",
        use_container_width=True,
    ):
        refresh_all_file_verification()
        st.success("✅ File verification cache refreshed.")

    # Quick summary counters
    all_ok_count  = 0
    has_miss_count = 0
    na_count       = 0

    for mname, mcfg in MODEL_REGISTRY.items():
        mtype = mcfg.get("type", "cli")
        if mtype != "cli" or not mcfg.get("args"):
            na_count += 1
        else:
            ok, _ = get_model_file_status(mname)
            if ok:
                all_ok_count += 1
            else:
                has_miss_count += 1

    verify_col2.metric("Models OK", all_ok_count)
    vc3, vc4 = st.columns(2)
    vc3.metric("Missing Files", has_miss_count, delta=None)
    vc4.metric("Non-CLI (N/A)", na_count)

    st.divider()

    # Filter for the verify section
    verify_filter = st.radio(
        "Show models:",
        ["All", "✓ OK only", "⚠ Missing files only", "Non-CLI only"],
        horizontal=True,
        key="verify_filter",
    )

    # Iterate all registered models and show status cards
    for mname in sorted(MODEL_REGISTRY.keys()):
        mcfg  = MODEL_REGISTRY[mname]
        mtype = mcfg.get("type", "cli")
        args  = mcfg.get("args", [])
        meta  = mcfg.get("meta", {})
        dls   = mcfg.get("downloads", [])

        # Determine status
        if mtype != "cli" or not args:
            status_label = "N/A (non-CLI)"
            card_class   = "verify-card-na"
            status_class = "verify-status-na"
            all_ok_m     = True
            missing_m    = []
            if verify_filter in ["✓ OK only", "⚠ Missing files only"]:
                continue
        else:
            all_ok_m, missing_m = get_model_file_status(mname)
            if all_ok_m:
                status_label = "✓ All files present"
                card_class   = "verify-card-ok"
                status_class = "verify-status-ok"
                if verify_filter == "⚠ Missing files only":
                    continue
            else:
                status_label = f"⚠ {len(missing_m)} file(s) missing"
                card_class   = "verify-card-miss"
                status_class = "verify-status-miss"
                if verify_filter == "✓ OK only":
                    continue

        if verify_filter == "Non-CLI only" and mtype == "cli":
            continue

        # Render the card
        lang_str = " · ".join(LANG_LABEL.get(l, l) for l in mcfg.get("langs", []))
        with st.expander(
            f"{mname}  —  {status_label}",
            expanded=not all_ok_m and mtype == "cli" and bool(args),
        ):
            # Header row
            st.markdown(
                f'<div class="{card_class}">'
                f'<div class="verify-model-name">{mname}</div>'
                f'<div class="{status_class}">{status_label}</div>'
                f'<div style="font-size:0.78rem;color:#636678;margin-top:4px;">'
                f'Type: <code>{mtype}</code> · Lang: {lang_str} · '
                f'Size: {meta.get("model_size","—")} · '
                f'Provider: {meta.get("provider","—")}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Missing files list
            if missing_m:
                st.markdown("**Missing files:**")
                for flag, path in missing_m:
                    st.markdown(
                        f'<div class="file-miss">✗ <b>{flag}</b> → {path}</div>',
                        unsafe_allow_html=True,
                    )

            # Present files (collapsed summary)
            if mtype == "cli" and args and all_ok_m:
                all_flags = {
                    "--vits-model","--vits-tokens","--vits-lexicon",
                    "--matcha-acoustic-model","--matcha-vocoder","--matcha-tokens",
                    "--kokoro-model","--kokoro-voices","--kokoro-tokens",
                    "--kitten-model","--kitten-voices","--kitten-tokens",
                    "--vits-data-dir","--matcha-data-dir",
                    "--kokoro-data-dir","--kitten-data-dir",
                }
                file_args = []
                i = 0
                while i < len(args):
                    if args[i] in all_flags and i + 1 < len(args):
                        file_args.append((args[i], args[i+1]))
                        i += 2
                    else:
                        i += 1
                if file_args:
                    for flag, path in file_args:
                        p = Path(path)
                        size = ""
                        if p.is_file():
                            try:
                                size = f" · {p.stat().st_size/1e6:.1f} MB"
                            except OSError:
                                pass
                        st.markdown(
                            f'<div class="file-ok">✓ <b>{flag}</b> → {path}{size}</div>',
                            unsafe_allow_html=True,
                        )

            # ── Download section for missing files ────────────
            if not all_ok_m and mtype == "cli":
                if dls:
                    st.markdown("**Download URLs available:**")
                    for dl_url in dls:
                        st.markdown(
                            f'<div class="dl-url-chip">🔗 {dl_url}</div>',
                            unsafe_allow_html=True,
                        )

                    dl_key = f"dl_btn_{mname[:30].replace(' ','_')}"
                    if st.button(
                        f"⬇️ Download Missing Files for {mname[:35]}",
                        key=dl_key,
                        use_container_width=True,
                    ):
                        if not REQUESTS_AVAILABLE:
                            st.error(
                                "❌ `requests` library not available. "
                                "Install with `pip install requests`."
                            )
                        else:
                            for dl_url in dls:
                                fname    = dl_url.split("/")[-1]
                                dest     = ASSETS_DIR / fname
                                prog_ph  = st.empty()
                                prog_bar = st.progress(0, text=f"⬇️ Preparing {fname}…")

                                ok_dl = download_file(dl_url, dest, prog_bar)

                                if ok_dl and dest.exists():
                                    # Extract archive
                                    ex_status = st.empty()
                                    ex_ok = extract_local_archive(dest, BASE_DIR, ex_status)
                                    if ex_ok:
                                        # Re-verify
                                        st.session_state.file_verification_cache.pop(mname, None)
                                        new_ok, new_miss = get_model_file_status(mname)
                                        if new_ok:
                                            st.success(
                                                f"✅ **{mname}** — all files now present! "
                                                f"Ready to generate."
                                            )
                                        else:
                                            st.warning(
                                                f"⚠️ Downloaded & extracted, but "
                                                f"{len(new_miss)} file(s) still missing. "
                                                f"The archive may have different paths."
                                            )
                                        prog_ph.empty()
                                else:
                                    st.error(
                                        f"❌ Download failed for `{fname}`. "
                                        f"Try downloading manually and using Section A."
                                    )
                else:
                    # No download URLs — guide the user
                    st.markdown(
                        '<div class="reg-warn">'
                        '⚠️ No download URLs configured for this model.<br>'
                        'Please:<br>'
                        '1. Download the model archive manually (see links above).<br>'
                        '2. Copy it into the app directory.<br>'
                        '3. Use <b>Section A</b> to extract it.<br>'
                        '4. Click <b>🔄 Re-verify All Model Files</b> above.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            # Non-CLI info
            if mtype != "cli":
                st.caption(
                    f"ℹ️ This is a `{mtype}` model — file paths are managed internally. "
                    f"No individual file check performed."
                )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION F — JSON-Registered Models Manager
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">'
        '<span class="step-badge">F</span> Manage JSON-Registered Models</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Models saved as JSON files in `{MODELS_DIR}`. "
        "These persist across app restarts."
    )

    json_model_files = sorted(MODELS_DIR.glob("*.json"))
    if not json_model_files:
        st.caption("No JSON model files in `./models/` yet.")
    else:
        st.caption(f"**{len(json_model_files)}** file(s) in `./models/`")
        for jf in json_model_files:
            try:
                raw_jf  = json.loads(jf.read_text(encoding="utf-8"))
                entries = raw_jf if isinstance(raw_jf, list) else [raw_jf]
                for e in entries:
                    jname       = e.get("display_name", jf.stem)
                    jmeta       = e.get("meta", {})
                    jinstalled  = e.get("installed", False)
                    # Quick live file check
                    jok, jmiss  = get_model_file_status(jname)
                    status_icon = "✓" if jok else f"⚠ {len(jmiss)} missing"
                    status_col  = "#5ef08a" if jok else "#f87171"

                    dc1, dc2, dc3 = st.columns([4, 1, 1])
                    dc1.markdown(
                        f"**{jname}** — `{e.get('type','')}` · "
                        + " · ".join(LANG_LABEL.get(l, l) for l in e.get("langs", []))
                        + f" · <span style='color:{status_col};font-size:0.82rem'>"
                        + status_icon + "</span>",
                        unsafe_allow_html=True,
                    )
                    dc2.caption(jmeta.get("model_size", ""))
                    if dc3.button("🗑", key=f"del_{jf.stem}_{jname[:15]}",
                                  use_container_width=True):
                        jf.unlink()
                        st.session_state.model_registry.pop(jname, None)
                        # Restore hardcoded default if exists
                        hc = _hardcoded_registry()
                        if jname in hc:
                            st.session_state.model_registry[jname] = hc[jname]
                        st.session_state.file_verification_cache.pop(jname, None)
                        st.success(f"Deleted `{jf.name}`.")
                        st.rerun()
            except Exception as ex:
                st.warning(f"Could not read {jf.name}: {ex}")


# ══════════════════════════════════════════════════════════════
# TAB 8 — ADD MODEL VIA JSON
# ══════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<p class="section-header">Register via JSON Config</p>',
                unsafe_allow_html=True)
    st.caption(
        "Paste a full model config JSON for precise control over args and metadata. "
        "File existence is verified immediately after validation."
    )

    if st.button("📝 Load VCTK Template", key="tpl_vctk", use_container_width=False):
        st.session_state["reg_json_input"] = json.dumps({
            "display_name": "VITS-VCTK Speaker 5 (Custom)",
            "type": "cli", "langs": ["en"],
            "args": ["--vits-model","./vits-vctk/vits-vctk.int8.onnx",
                     "--vits-tokens","./vits-vctk/tokens.txt",
                     "--vits-lexicon","./vits-vctk/lexicon.txt","--sid","5"],
            "downloads": [
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2"
            ],
            "meta": {
                "provider":"K2-FSA","parameters":"~40M","compute_type":"INT8/ONNX",
                "model_size":"37MB","license":"MIT","best_use_case":"CUSTOM SPEAKER TTS",
            },
        }, indent=2)

    json_input = st.text_area(
        "Paste JSON",
        value=st.session_state.get("reg_json_input",""),
        height=320,
        placeholder=(
            '{\n  "display_name": "My Model",\n  "type": "cli",\n'
            '  "langs": ["en"],\n  "args": [...],\n  "downloads": [...],\n'
            '  "meta": {...}\n}'
        ),
        label_visibility="collapsed",
        key="reg_json_paste",
    )

    col_val, col_clr = st.columns([5, 1])
    validate_btn = col_val.button(
        "🔍 Validate & Register",
        key="reg_validate",
        use_container_width=True,
        type="primary",
    )
    clr_btn = col_clr.button("✕", key="reg_clr", use_container_width=True)

    if clr_btn:
        st.session_state.pop("reg_json_input", None)
        st.rerun()

    if validate_btn:
        if not json_input.strip():
            st.warning("Please paste a JSON config first.")
        else:
            try:
                parsed = json.loads(json_input)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON syntax: {e}")
                st.stop()

            is_valid, errors, warnings = validate_model_json(parsed)
            for err in errors:
                st.markdown(f'<div class="reg-error">❌ {err}</div>', unsafe_allow_html=True)
            for warn in warnings:
                st.markdown(f'<div class="reg-warn">⚠️ {warn}</div>', unsafe_allow_html=True)
            if not is_valid:
                st.stop()

            display_name = parsed["display_name"]

            # File check
            missing      = check_required_files(parsed.get("args", []))
            all_present  = len(missing) == 0

            if missing:
                st.warning(f"⚠️ {len(missing)} file(s) missing:")
                for flag, path in missing:
                    st.markdown(
                        f'<div class="file-miss">✗ {flag} → {path}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="reg-box">✅ All required files are present on disk.</div>',
                    unsafe_allow_html=True,
                )

            # Show download URLs if present
            dls = parsed.get("downloads", [])
            if dls and not all_present:
                st.markdown("**Download URLs configured:**")
                for u in dls:
                    st.markdown(
                        f'<div class="dl-url-chip">🔗 {u}</div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Go to **📁 Local Files → Section E** after registering "
                    "to download missing files."
                )

            if st.button(
                "💾 Save & Register",
                key="json_save_final",
                type="primary",
                use_container_width=True,
            ):
                register_model_in_session({**parsed, "installed": all_present})
                save_model_json_to_disk(parsed, mark_installed=all_present)
                st.success(
                    f"✅ **{display_name}** registered "
                    f"({'all files present' if all_present else 'some files missing — see Local Files tab'})!"
                )
                st.balloons()
                st.session_state.pop("reg_json_input", None)

    st.divider()
    st.markdown('<p class="section-header">JSON-Registered Models</p>', unsafe_allow_html=True)
    json_model_files_t8 = sorted(MODELS_DIR.glob("*.json"))
    if not json_model_files_t8:
        st.caption("No JSON model files in `./models/` yet.")
    else:
        st.caption(f"{len(json_model_files_t8)} file(s) in `./models/`")
        for jf in json_model_files_t8:
            try:
                raw     = json.loads(jf.read_text(encoding="utf-8"))
                entries = raw if isinstance(raw, list) else [raw]
                for e in entries:
                    name      = e.get("display_name", jf.stem)
                    meta      = e.get("meta", {})
                    ok_j, _   = get_model_file_status(name)
                    dc1, dc2, dc3 = st.columns([4, 1, 1])
                    dc1.markdown(
                        f"**{name}** — `{e.get('type','')}` · "
                        + " · ".join(LANG_LABEL.get(l, l) for l in e.get("langs", []))
                        + (" · <span style='color:#5ef08a'>✓</span>"
                           if ok_j else
                           " · <span style='color:#fbbf24'>⚠</span>"),
                        unsafe_allow_html=True,
                    )
                    dc2.caption(meta.get("model_size", ""))
                    if dc3.button("🗑", key=f"del8_{jf.stem}_{name[:15]}",
                                  use_container_width=True):
                        jf.unlink()
                        st.session_state.model_registry.pop(name, None)
                        hc = _hardcoded_registry()
                        if name in hc:
                            st.session_state.model_registry[name] = hc[name]
                        st.session_state.file_verification_cache.pop(name, None)
                        st.success(f"Deleted `{jf.name}`.")
                        st.rerun()
            except Exception as ex:
                st.warning(f"Could not read {jf.name}: {ex}")


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
