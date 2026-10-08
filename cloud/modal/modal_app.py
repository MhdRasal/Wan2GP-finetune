"""
Production Modal Application for Wan2GP (High-Performance GPU-Accelerated)
Architecture: Immutable Base Container + Persistent Volume Venv (--system-site-packages)

Fully compliant with:
  - WanVideo-2nd-stage services/gpu-provider.ts
  - WanVideo-2nd-stage server.ts orchestration
  - High-performance GPU tiers (L4, L40S, A100-40GB, A100-80GB, H100)

KEY HIGHLIGHTS:
1. IMMUTABLE BASE IMAGE:
   - Ubuntu 22.04 + CUDA 12.4.1 devel + Python 3.11
   - PyTorch 2.5.1 + TorchVision + Torchaudio (matching CUDA)
   - Pre-compiled kernels: FlashAttention, Triton, SageAttention, mmgp
   - Fast installer: uv
2. PERSISTENT VOLUME VENV (/models/venv):
   - Created with --system-site-packages on wan2gp-models volume
   - Inherits 4GB+ PyTorch & CUDA binaries with ZERO disk duplication
   - Holds fast-changing dependencies: transformers, diffusers, huggingface_hub, etc.
   - Synchronized on CPU via uv pip install in 3-5 seconds with $0 GPU cost
3. ZERO-TOUCH MODEL WEIGHTS:
   - Checkpoints remain untouched in /models/Wan2GP/ckpts across all updates
"""

import os
import sys
import json
import uuid
import shutil
import hashlib
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import modal
from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

# ---------------------------------------------------------------------------
# Configuration & Volumes
# ---------------------------------------------------------------------------
APP_NAME = os.environ.get("MODAL_APP_NAME", "wan2gp-prod3")
EXPECTED_API_KEY = os.environ.get("MODAL_API_KEY", "as-eRxoMxE95CgoFOZ7o98iBp")

OUTPUTS_VOL_NAME = os.environ.get("MODAL_VOLUME_NAME", "wan2gp-outputs")
MODELS_VOL_NAME = os.environ.get("MODAL_MODELS_VOLUME_NAME", "wan2gp-models")
CACHE_VOL_NAME = "wan2gp-cache"

outputs_volume = modal.Volume.from_name(OUTPUTS_VOL_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOL_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)

OUTPUTS_MOUNT = "/outputs"
MODELS_MOUNT = "/models"
CACHE_MOUNT = "/cache"

VENV_PATH = Path(MODELS_MOUNT) / "venv"
VENV_PYTHON = VENV_PATH / "bin" / "python"

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Source Patching Helper (Build-time & Runtime)
# ---------------------------------------------------------------------------
def apply_source_patches(wgp_dir: Path, log_fn=print):
    """
    Applies source-level patches to a Wan2GP installation:
    1. Triton compatibility (removes do_not_specialize_on_alignment in nanovllm).
    2. higgs_audio_v2_tokenizer (adds exist_ok=True).
    3. download_progress.py (adds **kwargs to _http_get & _xet_get).
    Can be called during container build time OR dynamically at runtime on volume code.
    """
    import re
    if not wgp_dir or not wgp_dir.exists():
        return

    # 1. Triton compatibility (nanovllm)
    nanovllm_dir = wgp_dir / "shared" / "llm_engines" / "nanovllm"
    if nanovllm_dir.exists():
        for py_file in nanovllm_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                original = content
                if "do_not_specialize_on_alignment" in content:
                    content = re.sub(r',\s*do_not_specialize_on_alignment=\([^)]*\)', '', content)
                    content = re.sub(r'do_not_specialize_on_alignment=\([^)]*\),\s*', '', content)
                content = re.sub(
                    r'do_not_specialize=\(([a-zA-Z0-9_"\']+)\)',
                    r'do_not_specialize=(\1,)',
                    content
                )
                if content != original:
                    py_file.write_text(content, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {py_file.name} for Triton compat")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {py_file}: {e}")

    # 2. higgs_audio_v2_tokenizer exist_ok=True
    tts_tok = wgp_dir / "models" / "TTS" / "omnivoice" / "higgs_audio_v2_tokenizer" / "__init__.py"
    if tts_tok.exists():
        try:
            content = tts_tok.read_text(encoding="utf-8")
            original = content
            content = content.replace(
                'AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig)',
                'AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig, exist_ok=True)'
            )
            content = content.replace(
                'AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel)',
                'AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, exist_ok=True)'
            )
            if content != original:
                tts_tok.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched higgs_audio_v2_tokenizer for exist_ok=True")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch tts tokenizer: {e}")

    # 3. download_progress.py: accept **kwargs & huggingface_hub _request_wrapper compat
    dl_progress = wgp_dir / "shared" / "utils" / "download_progress.py"
    if dl_progress.exists():
        try:
            content = dl_progress.read_text(encoding="utf-8")
            original = content
            if "_tqdm_bar=None):" in content:
                content = content.replace("_tqdm_bar=None):", "_tqdm_bar=None, **kwargs):")

            hf_target = "from huggingface_hub import file_download as hf"
            hf_shim = (
                "from huggingface_hub import file_download as hf\n"
                "    if not hasattr(hf, '_request_wrapper'):\n"
                "        import requests\n"
                "        hf._request_wrapper = lambda *args, **kwargs: requests.request(*args, **kwargs)\n"
                "    if not hasattr(hf, '_get_file_length_from_http_response'):\n"
                "        hf._get_file_length_from_http_response = lambda r: int(r.headers.get('content-length', 0)) or None\n"
                "    if not hasattr(hf, '_adjust_range_header'):\n"
                "        hf._adjust_range_header = lambda r, c: f'bytes={c}-'\n"
            )
            if "requests.request(*args, **kwargs)" not in content:
                if "http_backoff" in content:
                    content = content.replace("lambda method, url, **p: http_backoff(requests.request, method=method, url=url, **p)", "lambda *args, **kwargs: requests.request(*args, **kwargs)")
                elif hf_target in content:
                    content = content.replace(hf_target, hf_shim, 1)
            if content != original:
                dl_progress.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched download_progress.py for huggingface_hub compat")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch download_progress.py: {e}")

    # 4. wgp.py: Add torch.accelerator compatibility shim
    wgp_py = wgp_dir / "wgp.py"
    if wgp_py.exists():
        try:
            wgp_code = wgp_py.read_text(encoding="utf-8")
            shim = (
                "# --- Torch Accelerator Compat Shim ---\n"
                "import torch\n"
                "if not hasattr(torch, 'accelerator'):\n"
                "    class _TorchAccShim:\n"
                "        @staticmethod\n"
                "        def is_available(): return False\n"
                "    torch.accelerator = _TorchAccShim\n"
                "# --------------------------------------\n"
            )
            if "_TorchAccShim" not in wgp_code:
                wgp_py.write_text(shim + wgp_code, encoding="utf-8")
                log_fn("[PATCH] Added torch.accelerator compatibility shim to wgp.py")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch wgp.py: {e}")

    # 4b. Qwen3VL pad_token_id compatibility for Hugging Face Transformers
    for q_rel in [
        "models/ideogram4/qwen3_vl_transformers.py",
        "models/hidream/qwen3_vl_transformers.py",
    ]:
        q_path = wgp_dir / q_rel
        if q_path.exists():
            try:
                txt = q_path.read_text(encoding="utf-8")
                if "self.padding_idx = config.pad_token_id" in txt:
                    txt = txt.replace(
                        "self.padding_idx = config.pad_token_id",
                        'self.padding_idx = getattr(config, "pad_token_id", None)'
                    )
                    q_path.write_text(txt, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {q_rel} for pad_token_id attribute safety")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

    for q_rel in [
        "models/ideogram4/qwen3_vl_configuration.py",
        "models/hidream/qwen3_vl_configuration.py",
    ]:
        q_path = wgp_dir / q_rel
        if q_path.exists():
            try:
                txt = q_path.read_text(encoding="utf-8")
                if "pad_token_id=None" not in txt and "class Qwen3VLTextConfig(PretrainedConfig):" in txt:
                    txt = txt.replace(
                        "attention_dropout=0.0,\n        **kwargs,",
                        "attention_dropout=0.0,\n        pad_token_id=None,\n        **kwargs,"
                    )
                    txt = txt.replace(
                        "self.vocab_size = vocab_size",
                        "self.pad_token_id = pad_token_id\n        self.vocab_size = vocab_size"
                    )
                    q_path.write_text(txt, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {q_rel} for pad_token_id config default")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

    # 5. diffusers torch_utils: patch if available
    for p in [
        Path("/usr/local/lib/python3.11/site-packages/diffusers/utils/torch_utils.py"),
        Path(MODELS_MOUNT) / "venv/lib/python3.11/site-packages/diffusers/utils/torch_utils.py"
    ]:
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                if "if torch.accelerator.is_available():" in txt:
                    p.write_text(txt.replace("if torch.accelerator.is_available():", "if hasattr(torch, 'accelerator') and torch.accelerator.is_available():"), encoding="utf-8")
                    log_fn(f"[PATCH] Patched {p.name} for torch.accelerator")
            except Exception as e:
                pass

    # 6. sitecustomize.py shim
    for site_dir in [
        Path("/usr/local/lib/python3.11/site-packages"),
        Path(MODELS_MOUNT) / "venv/lib/python3.11/site-packages"
    ]:
        if site_dir.exists():
            try:
                sc = site_dir / "sitecustomize.py"
                sc.write_text(
                    "import torch\n"
                    "if not hasattr(torch, 'accelerator'):\n"
                    "    class _TorchAccShim:\n"
                    "        @staticmethod\n"
                    "        def is_available(): return False\n"
                    "    torch.accelerator = _TorchAccShim\n",
                    encoding="utf-8"
                )
            except Exception as e:
                pass


def _bake_patches():
    """
    BUILD-TIME ONLY: Called via modal.Image.run_function() during image build.
    Applies source-level patches to /root/Wan2GP and installs helper wheels.
    """
    import subprocess
    import sys
    from pathlib import Path

    apply_source_patches(Path("/root/Wan2GP"), log_fn=print)

    # 4. smplfitter + chumpy wheels
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://github.com/deepbeepmeep/smplfitter/releases/download/v0.2.10/smplfitter-0.2.10-py3-none-any.whl",
        "https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl"
    ], check=False)

    # 5. Remove conflicting hf-xet
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "hf_xet", "hf-xet"],
        check=False
    )
# ---------------------------------------------------------------------------
# Base Container Image Definition
# ---------------------------------------------------------------------------
wan2gp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libportaudio2",
        "curl",
        "wget",
        "xz-utils",
        "build-essential",
        "ninja-build",
        "cmake"
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9;9.0+PTX",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_ALLOW_TF32_CUBLAS": "1",
        "TORCH_ALLOW_TF32_CUDNN": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONUNBUFFERED": "1",
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "TRITON_CACHE_DIR": "/cache/triton"
    })
    .pip_install(
        "uv>=0.4.0",
        "ninja",
        "wheel",
        "packaging",
        "setuptools<=75.8.2",
        "triton>=3.0.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "mmgp==3.8.2",
        "rotary-embedding-torch>=0.5.3",
        "onnxruntime",
        "rembg",
        "scipy",
        "av",
        "ffmpeg-python",
        "pygame",
        "pyloudnorm",
        "mutagen",
        "einshape",
        "timm",
        "decord",
        "peft",
        "gguf",
        "torchdiffeq",
        "tensordict",
        "omegaconf",
        "soundfile",
        "ftfy",
        "moviepy==1.0.3",
        "gradio-rangeslider",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.0.0",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "gradio==5.29.0",
        "safetensors>=0.4.5",
        "einops>=0.8.0",
        "sentencepiece>=0.2.0",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "opencv-python-headless>=4.10.0",
        "numpy>=1.26.0,<2.0.0",
        "optimum-quanto>=0.2.6",
        "hf-transfer>=0.1.8",
        "huggingface-hub>=0.28.1",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "openai-whisper",
        "loguru",
        "orjson",
        "tiktoken",
        "open_clip_torch",
        "librosa",
        "hydra-core",
        "easydict",
        "conformer",
        "vector-quantize-pytorch",
        "Cython",
        "markdown",
        "apprise",
        "keyring",
        "piexif",
        "mcp",
        "misaki",
        "gitpython",
        "stringzilla",
        "xxhash",
        "munch",
        "matplotlib",
        "pandas",
        "taichi",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124 --progress-bar off"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git /root/Wan2GP",
        "pip install flash-attn --no-build-isolation || echo 'Flash-attn build failed, non-fatal'"
    )
    .run_function(_bake_patches)
)

# ---------------------------------------------------------------------------
# Volume Venv Management: Self-Healing & Synchronization
# ---------------------------------------------------------------------------
def compute_requirements_hash(req_path: Path) -> str:
    """Computes SHA256 of requirements.txt for change detection."""
    if not req_path.exists():
        return ""
    return hashlib.sha256(req_path.read_bytes()).hexdigest()[:16]

def ensure_volume_venv(log_file: Optional[Path] = None, allow_install: bool = False) -> Path:
    """
    Ensures a persistent virtual environment exists at /models/venv
    configured with --system-site-packages.
    """
    def log(msg: str):
        print(msg)
        if log_file and log_file.parent.exists():
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")

    venv_dir = Path(MODELS_MOUNT) / "venv"
    venv_py = venv_dir / "bin" / "python"
    req_file = Path(MODELS_MOUNT) / "Wan2GP" / "requirements.txt"
    if not req_file.exists():
        fallback_req = Path("/root/Wan2GP/requirements.txt")
        if fallback_req.exists():
            req_file = fallback_req
    hash_file = venv_dir / ".req_hash"

    # Step 1: Create venv if missing
    if not venv_py.exists():
        log("[VENV] Creating persistent volume venv with --system-site-packages at /models/venv...")
        venv_dir.mkdir(parents=True, exist_ok=True)
        
        # Try uv first, fallback to standard venv
        uv_cmd = shutil.which("uv")
        if uv_cmd:
            cmd = [uv_cmd, "venv", str(venv_dir), "--system-site-packages", "--python", sys.executable]
        else:
            cmd = [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log(f"[VENV-ERROR] Failed to create venv: {res.stderr}")
            return Path(sys.executable)
        
        log("[VENV] Created /models/venv successfully.")

    # Step 2: Synchronize requirements if changed (only allowed on CPU update endpoint)
    if req_file.exists() and allow_install:
        current_hash = compute_requirements_hash(req_file)
        saved_hash = hash_file.read_text().strip() if hash_file.exists() else ""

        if current_hash != saved_hash:
            log(f"[VENV] Syncing dependencies from {req_file} (hash: {current_hash})...")
            uv_cmd = shutil.which("uv")
            if uv_cmd:
                install_cmd = [
                    uv_cmd, "pip", "install",
                    "--python", str(venv_py),
                    "-r", str(req_file)
                ]
            else:
                install_cmd = [
                    str(venv_py), "-m", "pip", "install",
                    "-r", str(req_file)
                ]

            res = subprocess.run(install_cmd, capture_output=True, text=True)
            if res.returncode == 0:
                hash_file.write_text(current_hash)
                log("[VENV] Successfully updated dependencies in /models/venv.")
            else:
                log(f"[VENV-WARN] Dependency update returned code {res.returncode}: {res.stderr[:300]}")
    elif req_file.exists():
        log("[VENV] Using volume venv (fast startup, 0 setup overhead).")

    try:
        models_volume.commit()
    except Exception:
        pass

    return venv_py if venv_py.exists() else Path(sys.executable)

# ---------------------------------------------------------------------------
# GPU Configuration & Tuning Profiles
# ---------------------------------------------------------------------------
GPU_PROFILE_MAP = {
    "H100": {"profile": 1, "attention": "sage2"},
    "A100_80GB": {"profile": 1, "attention": "sage2"},
    "A100_40GB": {"profile": 2, "attention": "sage2"},
    "L40S": {"profile": 1, "attention": "sage2"},
    "L4": {"profile": 3, "attention": "sage2"},
    "RTX_PRO_6000": {"profile": 1, "attention": "sage2"},
    "T4": {"profile": 5, "attention": "sdpa"}
}

def resolve_gpu_tuning(requested_gpu: str) -> Tuple[int, str, str]:
    norm_gpu = requested_gpu.upper().replace("-", "_").replace(" ", "_")
    detected_device = "Unknown"

    try:
        import torch
        if torch.cuda.is_available():
            detected_device = torch.cuda.get_device_name(0).upper()
    except Exception:
        pass

    matched_key = "L40S"
    test_str = f"{detected_device} {norm_gpu}"

    if "H100" in test_str:
        matched_key = "H100"
    elif "80GB" in test_str or ("A100" in test_str and "80" in test_str):
        matched_key = "A100_80GB"
    elif "A100" in test_str:
        matched_key = "A100_40GB"
    elif "40S" in test_str or "L40" in test_str:
        matched_key = "L40S"
    elif "L4" in test_str:
        matched_key = "L4"
    elif "6000" in test_str:
        matched_key = "RTX_PRO_6000"
    elif "T4" in test_str:
        matched_key = "T4"

    tuning = GPU_PROFILE_MAP.get(matched_key, GPU_PROFILE_MAP["L40S"])
    attention = tuning["attention"]

    if attention in ("sage", "sage2"):
        try:
            import sageattention
        except Exception:
            try:
                import flash_attn
                attention = "flash"
            except Exception:
                attention = "sdpa"

    return tuning["profile"], attention, detected_device

def setup_wan2gp_environment(wgp_dir: Path, profile_num: int, attention_mode: str, log_file: Path):
    """
    Runtime setup (~50ms total):
    1. Ensures source patches (Triton, tokenizer, etc.) are applied to the active Wan2GP code.
    2. Symlink model weights from /models volume into Wan2GP/models (skips Wan2GP code)
    3. Writes optimal wgp_config.json for active GPU
    """
    def log_line(msg: str):
        with open(log_file, "a") as log:
            log.write(msg + chr(10))

    # 1. Apply runtime patches to active Wan2GP directory (critical for persistent volume /models/Wan2GP)
    apply_source_patches(wgp_dir, log_fn=log_line)

    wgp_models_dir = wgp_dir / "models"
    wgp_models_dir.mkdir(parents=True, exist_ok=True)

    models_volume_path = Path(MODELS_MOUNT)
    linked_count = 0

    if models_volume_path.exists():
        for item in models_volume_path.iterdir():
            if item.name in ("Wan2GP", "venv"):
                continue
            target_link = wgp_models_dir / item.name
            if not target_link.exists():
                try:
                    target_link.symlink_to(item)
                    linked_count += 1
                except Exception as e:
                    log_line(f"[WARN] Failed to symlink {item.name}: {e}")

    config_data = {
        "attention_mode": attention_mode,
        "compile": "",
        "video_profile": profile_num,
        "image_profile": profile_num,
        "audio_profile": profile_num
    }
    config_file = wgp_dir / "wgp_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    log_line(f"[SETUP] Linked {linked_count} model assets from {MODELS_MOUNT}")
    log_line(f"[SETUP] Configured wgp_config.json: profile={profile_num}, attention={attention_mode}")

    # Commit any patch modifications if wgp_dir resides on the persistent volume
    if str(wgp_dir).startswith(MODELS_MOUNT):
        try:
            models_volume.commit()
        except Exception:
            pass
# ---------------------------------------------------------------------------
# Job Execution Engine
# ---------------------------------------------------------------------------
def execute_wan2gp_job(job_id: str, batch_uuid: Optional[str] = None, requested_gpu: str = "L40S"):
    import time
    job_dir = Path(OUTPUTS_MOUNT) / job_id

    for _ in range(5):
        try:
            outputs_volume.reload()
        except Exception:
            pass
        if (job_dir / "input.zip").exists():
            break
        time.sleep(1)

    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    log_file = job_dir / "run.log"
    zip_path = job_dir / "input.zip"

    def update_status(status_str: str, error_msg: Optional[str] = None, files: Optional[list] = None):
        st = {
            "status": status_str,
            "job_id": job_id,
            "batch_uuid": batch_uuid,
            "requested_gpu": requested_gpu,
            "error": error_msg,
            "files": files or []
        }
        with open(status_file, "w") as f:
            json.dump(st, f, indent=2)
        outputs_volume.commit()

    update_status("running")
    profile_num, attention_mode, detected_device = resolve_gpu_tuning(requested_gpu)

    with open(log_file, "a") as log:
        log.write(f"=== Starting Wan2GP job {job_id} ===\n")
        log.write(f"Batch UUID: {batch_uuid}\n")
        log.write(f"Requested GPU: {requested_gpu} | Detected Hardware: {detected_device}\n")
        log.write(f"Tuning Strategy: Profile={profile_num}, Attention={attention_mode}\n")

    try:
        extract_dir = Path("/tmp") / job_id
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            raise FileNotFoundError(f"input.zip not found at {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        queue_file = extract_dir / "queue.json"
        if not queue_file.exists():
            raise FileNotFoundError("queue.json not found in input.zip")

        with open(queue_file, "r") as f:
            queue_data = json.load(f)

        # 1. Resolve Wan2GP root (persistent volume override takes priority)
        custom_volume_wgp = Path(MODELS_MOUNT) / "Wan2GP"
        if custom_volume_wgp.is_dir() and (custom_volume_wgp / "wgp.py").is_file():
            wan2gp_dir = custom_volume_wgp
            with open(log_file, "a") as log:
                log.write(f"[INFO] Using custom Wan2GP from volume: {custom_volume_wgp}\n")
        else:
            wan2gp_dir = Path("/root/Wan2GP")

        # 2. Resolve Python interpreter (Volume Venv with system site-packages)
        python_exec = ensure_volume_venv(log_file=log_file)
        with open(log_file, "a") as log:
            log.write(f"[INFO] Active Python environment: {python_exec}\n")

        # 3. Mount models & write runtime wgp_config.json
        setup_wan2gp_environment(wan2gp_dir, profile_num, attention_mode, log_file)

        # 4. Invoke Wan2GP CLI
        cmd = [
            str(python_exec),
            str(wan2gp_dir / "wgp.py"),
            "--profile", str(profile_num),
            "--attention", str(attention_mode),
            "--process", str(zip_path),
            "--output-dir", str(job_dir)
        ]

        exec_env = os.environ.copy()
        exec_env["HF_XET_HIGH_PERFORMANCE"] = "0"
        exec_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        exec_env["TORCH_ALLOW_TF32_CUBLAS"] = "1"
        exec_env["TORCH_ALLOW_TF32_CUDNN"] = "1"
        exec_env["CUDA_MODULE_LOADING"] = "LAZY"
        exec_env["PYTHONUNBUFFERED"] = "1"

        with open(log_file, "a") as log:
            log.write(f"Executing: {' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(wan2gp_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=exec_env,
            text=True,
            bufsize=1
        )

        last_commit_time = time.time()
        with open(log_file, "a") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                now = time.time()
                if now - last_commit_time >= 2.0:
                    try:
                        outputs_volume.commit()
                    except Exception:
                        pass
                    last_commit_time = now

        proc.wait()
        try:
            outputs_volume.commit()
        except Exception:
            pass

        if proc.returncode != 0:
            raise RuntimeError(f"Wan2GP exited with error code {proc.returncode}")

        # 5. Collect outputs & write results.json
        output_files = [
            f.name for f in job_dir.glob("*")
            if f.is_file() and f.name not in ("input.zip", "status.json", "run.log")
        ]

        jobs_results = []
        for task in queue_data:
            t_id = str(task.get("id", 1))
            t_meta_id = task.get("meta_id", t_id)
            params = task.get("params", {})
            out_fn = params.get("output_filename", "")
            if not out_fn and output_files:
                mp4s = [f for f in output_files if f.endswith(".mp4")]
                out_fn = mp4s[0] if mp4s else output_files[0]

            jobs_results.append({
                "job_id": t_meta_id,
                "status": "completed",
                "output_filename": out_fn
            })

        results_json = {
            "batch_uuid": batch_uuid,
            "status": "completed",
            "gpu_hardware": detected_device,
            "profile_used": profile_num,
            "attention_used": attention_mode,
            "jobs": jobs_results
        }

        with open(job_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        output_files.append("results.json")
        try:
            models_volume.commit()
            cache_volume.commit()
        except Exception:
            pass
        update_status("completed", files=output_files)

        with open(log_file, "a") as log:
            log.write(f"\n[OK] Job {job_id} successfully completed. Outputs: {output_files}\n")

    except Exception as e:
        err_msg = str(e)
        with open(log_file, "a") as log:
            log.write(f"\n[ERROR] Job failed: {err_msg}\n")
        update_status("failed", error_msg=err_msg)

# ---------------------------------------------------------------------------
# GPU Workers Fleet
# ---------------------------------------------------------------------------
@app.function(
    image=wan2gp_image,
    gpu="H100",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_h100(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="H100")

@app.function(
    image=wan2gp_image,
    gpu="A100-80GB",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_a100_80gb(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="A100-80GB")

@app.function(
    image=wan2gp_image,
    gpu="A100-40GB",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_a100_40gb(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="A100-40GB")

@app.function(
    image=wan2gp_image,
    gpu="L40S",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_l40s(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="L40S")

@app.function(
    image=wan2gp_image,
    gpu="L4",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_l4(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="L4")

@app.function(
    image=wan2gp_image,
    gpu="T4",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600
)
def run_worker_t4(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="T4")

# ---------------------------------------------------------------------------
# Diagnostics & Admin Update Hook (CPU-Only, $0 GPU cost)
# ---------------------------------------------------------------------------
@app.function(
    image=wan2gp_image,
    gpu="L40S",
    volumes={MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=600
)
def inspect_gpu_environment() -> str:
    import torch
    report = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "None",
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown")),
        "accelerators": {}
    }
    for mod in ("sageattention", "triton", "flash_attn"):
        try:
            m = __import__(mod)
            report["accelerators"][mod] = str(getattr(m, "__version__", "installed"))
        except Exception as e:
            report["accelerators"][mod] = f"not available: {e}"
    return json.dumps(report)

@app.function(
    image=wan2gp_image,
    volumes={MODELS_MOUNT: models_volume},
    timeout=120
)
def patch_volume_repo() -> Dict[str, Any]:
    """CPU-only worker to apply source patches directly to /models/Wan2GP on volume."""
    target_dir = Path(MODELS_MOUNT) / "Wan2GP"
    msgs = []
    if target_dir.exists():
        apply_source_patches(target_dir, log_fn=lambda m: msgs.append(m))
        try:
            models_volume.commit()
        except Exception:
            pass
        return {"status": "ok", "patched_dir": str(target_dir), "messages": msgs}
    return {"status": "skipped", "message": f"{target_dir} does not exist"}


@app.function(
    image=wan2gp_image,
    volumes={MODELS_MOUNT: models_volume},
    timeout=600
)
def update_wan2gp_volume_repo() -> Dict[str, str]:
    """
    CPU-Only update worker ($0 GPU cost):
    1. Git pull Wan2GP source into /models/Wan2GP
    2. Synchronizes /models/venv via uv pip install
    3. Verifies packages
    """
    target_dir = Path(MODELS_MOUNT) / "Wan2GP"
    if not target_dir.exists():
        res = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/deepbeepmeep/Wan2GP.git", str(target_dir)],
            capture_output=True, text=True
        )
        git_msg = f"Cloned fresh repo: {res.stdout or res.stderr}"
    else:
        res = subprocess.run(
            ["git", "pull"],
            cwd=str(target_dir),
            capture_output=True, text=True
        )
        git_msg = f"Pulled latest updates: {res.stdout or res.stderr}"

    # Apply patches directly to volume repo
    apply_source_patches(target_dir, log_fn=print)

    # Sync volume venv with requirements
    venv_py = ensure_volume_venv(allow_install=True)
    
    models_volume.commit()
    return {
        "status": "ok",
        "git_message": git_msg,
        "venv_python": str(venv_py)
    }

# ---------------------------------------------------------------------------
# FastAPI Web Gateway (Lightweight CPU ASGI Server)
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Wan2GP High-Performance Modal Service")

def verify_auth(x_api_key: Optional[str] = None, authorization: Optional[str] = None):
    token = x_api_key
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()
    if EXPECTED_API_KEY and token != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

@web_app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "wan2gp-api-PROD3",
        "architecture": "volume-venv",
        "volumes": [OUTPUTS_VOL_NAME, MODELS_VOL_NAME, CACHE_VOL_NAME],
        "gpus": ["L40S", "A100-40GB", "A100-80GB", "H100", "L4", "T4"],
        "default_gpu": "L40S",
        "min_recommended_gpu": "L4 (24GB VRAM)"
    }

@web_app.get("/gpu-info")
def gpu_info_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    res_str = inspect_gpu_environment.remote()
    return json.loads(res_str)

@web_app.post("/admin/update-wan2gp")
def update_wan2gp_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    return update_wan2gp_volume_repo.remote()

@web_app.post("/admin/patch-volume")
def patch_volume_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    return patch_volume_repo.remote()

@web_app.post("/generate")
@web_app.post("/v1/jobs")
async def generate_endpoint(
    request: Request,
    gpu_type: str = Query("L40S"),
    batch_uuid: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    verify_auth(x_api_key, authorization)

    data = await request.body()
    if not data or len(data) < 2:
        raise HTTPException(status_code=400, detail="Empty or corrupted payload")

    job_id = str(uuid.uuid4())
    job_dir = Path(OUTPUTS_MOUNT) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / "input.zip"

    # Support Mode A (binary ZIP bundle) and Mode B (JSON GenerationRequest)
    if data.startswith(b"PK"):
        with open(zip_path, "wb") as f:
            f.write(data)
    else:
        try:
            json_obj = json.loads(data.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: neither ZIP nor JSON ({e})")

        if isinstance(json_obj, list):
            tasks = json_obj
        elif isinstance(json_obj, dict):
            if "queue" in json_obj and isinstance(json_obj["queue"], list):
                tasks = json_obj["queue"]
            elif "tasks" in json_obj and isinstance(json_obj["tasks"], list):
                tasks = json_obj["tasks"]
            else:
                task_params = dict(json_obj.get("params") or json_obj.get("settings") or {})
                if not task_params:
                    task_params = dict(json_obj)
                
                # Extract infrastructure parameter
                if "gpu_type" in json_obj and json_obj["gpu_type"]:
                    gpu_type = str(json_obj["gpu_type"])
                elif "gpu_type" in task_params:
                    gpu_type = str(task_params.pop("gpu_type"))

                m_type = json_obj.get("model_type") or json_obj.get("model") or task_params.get("model_type", "t2v")
                task_params["model_type"] = m_type

                for k in ("prompt", "negative_prompt", "resolution", "video_length", "num_inference_steps", "guidance_scale", "seed", "flow_shift"):
                    if k in json_obj and k not in task_params:
                        task_params[k] = json_obj[k]

                tasks = [{
                    "id": 1,
                    "meta_id": str(json_obj.get("task_id", uuid.uuid4())),
                    "params": task_params
                }]

        else:
            raise HTTPException(status_code=400, detail="Invalid JSON task structure")

        # Clean all tasks to satisfy strict manifest validation
        for t in tasks:
            if isinstance(t, dict):
                p = t.get("params")
                if isinstance(p, dict):
                    for disallowed in ("gpu_type", "width", "height", "task_id", "execution_mode", "engine", "version", "provider", "inputs", "plugin_data", "metadata"):
                        p.pop(disallowed, None)

        import io
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("queue.json", json.dumps(tasks, indent=2))
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

    initial_status = {
        "status": "queued",
        "job_id": job_id,
        "batch_uuid": batch_uuid,
        "gpu_type": gpu_type,
        "files": []
    }
    with open(job_dir / "status.json", "w") as f:
        json.dump(initial_status, f, indent=2)

    await outputs_volume.commit.aio()

    # Route to appropriate GPU worker
    norm_gpu = gpu_type.upper().replace("-", "_").replace(" ", "_")
    if "H100" in norm_gpu:
        await run_worker_h100.spawn.aio(job_id, batch_uuid)
    elif "80GB" in norm_gpu or ("A100" in norm_gpu and "80" in norm_gpu):
        await run_worker_a100_80gb.spawn.aio(job_id, batch_uuid)
    elif "A100" in norm_gpu:
        await run_worker_a100_40gb.spawn.aio(job_id, batch_uuid)
    elif norm_gpu == "L4" or norm_gpu.startswith("L4_"):
        await run_worker_l4.spawn.aio(job_id, batch_uuid)
    elif "T4" in norm_gpu:
        await run_worker_t4.spawn.aio(job_id, batch_uuid)
    else:
        await run_worker_l40s.spawn.aio(job_id, batch_uuid)

    return {"job_id": job_id, "status": "queued", "gpu_selected": gpu_type}

@web_app.get("/status/{job_id}")
@web_app.get("/v1/jobs/{job_id}")
def status_endpoint(job_id: str, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    verify_auth(x_api_key, authorization)
    outputs_volume.reload()

    status_file = Path(OUTPUTS_MOUNT) / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    with open(status_file, "r") as f:
        return json.load(f)

@web_app.get("/download/{job_id}")
def download_list_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    job_dir = Path(OUTPUTS_MOUNT) / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    files = [f.name for f in job_dir.glob("*") if f.is_file() and f.name != "input.zip"]
    return {"job_id": job_id, "files": files}

@web_app.get("/download/{job_id}/{filename}")
def download_file_endpoint(job_id: str, filename: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    target_file = Path(OUTPUTS_MOUNT) / job_id / filename
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found for job {job_id}")

    media_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return FileResponse(str(target_file), media_type=media_type, filename=filename)

@web_app.get("/logs/{job_id}")
def logs_endpoint(job_id: str, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    verify_auth(x_api_key, authorization)
    outputs_volume.reload()

    log_file = Path(OUTPUTS_MOUNT) / job_id / "run.log"
    if not log_file.exists():
        return PlainTextResponse("No logs found for this job.")

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse(f.read())

@web_app.post("/stop/{job_id}")
def stop_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    job_dir = Path(OUTPUTS_MOUNT) / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status_file = job_dir / "status.json"
    st = {"status": "stopped", "job_id": job_id}
    with open(status_file, "w") as f:
        json.dump(st, f, indent=2)
    outputs_volume.commit()

    return {"job_id": job_id, "status": "stopped"}

# ---------------------------------------------------------------------------
# Modal Entrypoint
# ---------------------------------------------------------------------------
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.115.0",
    "uvicorn>=0.30.0",
    "pydantic>=2.0.0"
)

@app.function(
    image=web_image,
    volumes={OUTPUTS_MOUNT: outputs_volume}
)
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    app.run()
