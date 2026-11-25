# Wan2GP Mobile / Cloud-Only Client Architecture & Audit

> **Goal**: Run the full `wgp.py` Gradio UI, Gallery, and Deepy (cloud/remote LLM only) on mobile/low-RAM devices (`--cloud`), with **zero local ML frameworks** (no PyTorch, no `transformers`, no `mmgp`, no `gguf`, no `opencv-python-headless`) and **zero risk of Android Low Memory Killer (LMK) termination**.

---

## 1. Core Architectural Principle

On mobile/Termux/MobCode, `wgp.py --cloud` acts strictly as a **thin UI & Cloud Dispatcher**:
- **Kept 100% Intact**:
  - Pixel-for-pixel upstream **Gradio UI** (`gradio==5.29.0`, `gradio_rangeslider`, all tabs, sliders, and model selectors).
  - **Gallery & Video Presentation** (`Pillow` for thumbnails/images, `ffmpeg` CLI / HTML5 `<video>` for video playback & metadata).
  - **Deepy (Cloud/Remote LLM Mode)** (`httpx`/`requests` to OpenCode/Claude/cloud endpoints, `markdown`, lazy-loaded `mcp`).
  - **Cloud Execution Adapters** (`shared/execution/providers/modal_provider.py`, `runpod_provider.py`, `GenerationRequest` JSON packaging).
- **Eliminated / Shimmed on Client** (runs only on the remote Modal/RunPod GPU worker):
  - **PyTorch (`torch`)**: Mocked by `shared/shims/cloud_shims.py`.
  - **`transformers` & RoPE patches**: Skipped when `IS_CLOUD_MODE` is active.
  - **`mmgp` (`offload`, `safetensors2`, `cuda_graphs`, `quant_router`, `transfer`)**: Bypassed when `IS_CLOUD_MODE` is active — no local VRAM/RAM allocators or CUDA device probes.
  - **`gguf` & local `lmgp` Deepy vision models**: Mocked in `cloud_shims.py` (`gguf` added to `MOCK_HEAVY_PACKAGES` with subscriptable `MockMeta`) so `shared/deepy/vision.py` does not crash or load local Qwen3.5 weights.
  - **Local CUDA / Triton / `flash_attn` / `shared/cuda_memory.py`**: Skipped in cloud mode.

---

## 2. Dependency Audit (`requirements-cloud.txt`)

### A. Strictly Needed for Cloud Client (UI + Gallery + Cloud Deepy)
| Package | Role |
| :--- | :--- |
| `gradio==5.29.0`, `gradio_rangeslider` | Exact upstream Gradio UI & range sliders |
| `fastapi`, `uvicorn`, `pydantic` | Gradio backend & `GenerationRequest` schemas |
| `pillow` | Gallery thumbnails, image previews, UI media handling |
| `numpy` | Minimal array support for Gradio/mock tensors (with `OPENBLAS_NUM_THREADS=1`) |
| `httpx`, `requests` | Modal/RunPod HTTP dispatch & remote Deepy LLM calls |
| `modal`, `runpod` | Cloud provider SDKs |
| `tqdm`, `loguru`, `markdown`, `typing-extensions` | UI progress, logging, Deepy markdown rendering |
| `mcp`, `claude-agent-sdk` | Deepy remote agent/MCP tools (should be lazy-loaded) |

### B. Ghost Dependencies (0 Imports — Safe to Delete)
Static analysis across the codebase showed **0 imports** for:
- `GitPython`
- `ffmpeg-python` (codebase calls `ffmpeg` via `subprocess.run`)
- `pydub`
- `easydict` (only used in local model configs)
- `orjson` (standard `json` is used)
- `python-dotenv` (`.env` is sourced in `run_cloud.sh`)

### C. Heavy Local Packages Not Needed on Mobile Client
- **`opencv-python-headless` (`cv2`)**: ~50–80 MB RSS; UI & Gallery use `Pillow` (`PIL`). Can be shimmed via `PIL` + `ffmpeg` CLI on the client.
- **`safetensors`**: ~25 MB RSS; model checkpoints live on the remote Modal/RunPod volume, never read on mobile.
- **`tokenizers`**: ~30 MB RSS; prompt tokenization runs on the cloud worker.

---

## 3. Why Termux/Mobile Killed the Process & How It Is Fixed

1. **OpenBLAS Thread Pool Bloat**: By default, importing `numpy` on an 8-core mobile CPU commits up to 32 MB per thread (~100–256 MB wasted).
   - **Fix**: `run_cloud.sh` exports `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`.
2. **`mmgp` & `cuda_memory` Startup Hooks**: `wgp.py` previously imported `mmgp` and ran `apply_startup_settings` even in `--cloud` mode.
   - **Fix**: Guarded under `if not IS_CLOUD_MODE:` in `wgp.py`.
3. **Android Background Process Killer (LMK)**: Switching from Termux to Chrome marks a normal terminal as a cached background process.
   - **Fix**: Running inside **MobCode** with the **floating overlay / foreground service** keeps the process at foreground priority. Measured RSS of `wgp.py --cloud` dropped to **~72 MB** during startup (vs ~720 MB+ unoptimized).

---

## 4. Upstream Sync & Modal >16 MB Downloads

- **Upstream Updates**: Because `wgp.py` renders Gradio directly from the repo, pulling upstream updates (`sync_upstream_and_modal.py`) automatically updates the mobile UI with zero manual UI rebuilding.
- **Large Video Downloads (>16 MB)**: Standard buffered `FileResponse` in `cloud/modal/modal_app.py` and full-body reads in `modal_provider.py` risk proxy limits/RAM spikes for >16 MB videos. Use `StreamingResponse` with 2 MB chunks (`Accept-Ranges: bytes`) on Modal and streaming chunked writes to disk on the client.

---

## 5. Quick Launch on Mobile (MobCode)

```bash
cd /sdcard/dev/Wan2GP-finetune
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
bash run_cloud.sh
```
Then open `http://127.0.0.1:7860` in your mobile browser.
