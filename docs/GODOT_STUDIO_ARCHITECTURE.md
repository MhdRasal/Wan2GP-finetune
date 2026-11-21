# Wan2GP Godot Studio: Editor Plugin & Cloud Architecture

> **Architecture Specification & Integration Guide**  
> Enables running Wan2GP generative AI workflows (Video, 3D Assets, Images, and Interactive Worlds) directly inside the **Godot 4 Editor** as an `EditorPlugin` dock.

---

## 1. Executive Summary & Core Motivation

Wan2GP upstream (by DeepBeepMeep) updates frequently with new models, prompt enhancers, camera controls, and scheduler features. 

Traditional approaches that attempt to build a custom, hardcoded mobile or desktop frontend suffer from immediate **upstream desynchronization**: every time Wan2GP adds or updates a parameter, the custom UI breaks and requires manual rewriting and recompilation.

The **Wan2GP Godot Studio Plugin** solves this permanently through **Dynamic Schema Reflection**:
* **Zero UI Rebuilding**: Godot queries the Wan2GP configuration contract at runtime and dynamically populates UI controls (sliders, dropdowns, toggles) directly into an Editor Dock.
* **100% Native Godot Integration**: Generates assets on your remote Modal / RunPod GPU worker and imports the resulting `.mp4` videos or `.glb` 3D meshes directly into Godot's `res://` project filesystem.
* **Ultra-Lightweight**: Operates with **< 10 MB RAM overhead** inside Godot (pure GDScript, zero local Python/CUDA needed on the client machine).

---

## 2. Architecture & Communication Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│  Godot 4 Editor (Desktop / Android Tablet)                             │
│                                                                        │
│  ┌───────────────────────┐           ┌──────────────────────────────┐  │
│  │  3D Viewport / Canvas │           │ Wan2GP Studio Editor Dock    │  │
│  │                       │           │                              │  │
│  │  [Auto-Imported Mesh] │ <───────  │ Dynamic Model: [Wan 2.1 ▼]   │  │
│  │  res://assets/wan3d/  │           │ Prompt: "cyberpunk city"     │  │
│  │                       │           │ Steps:  [----●--- 30]        │  │
│  └───────────────────────┘           │ [⚡ Generate on Modal GPU]   │  │
│                                      └──────────────┬───────────────┘  │
└─────────────────────────────────────────────────────┼──────────────────┘
                                                      │
                       1. GenerationRequest JSON      │  2. Server-Sent Events (SSE) Progress
                       HTTP POST /api/generate        │  3. Chunked Streaming Download (>16 MB)
                                                      ▼
                       ┌──────────────────────────────────────────────┐
                       │ Modal Cloud Worker (wan2gp-prod3)            │
                       │ FastAPI Web Endpoint + Persistent Volume     │
                       │ Heavy Diffusion Compute on L40S / A100 / H100│
                       └──────────────────────────────────────────────┘
```

---

## 3. Key Technical Pillars

### A. Dynamic Schema Reflection (Evergreen Upstream Sync)
Instead of hardcoding sliders for Wan 1.3B, Wan 14B, Hunyuan, or LTX, the Godot dock fetches the parameter schema at launch:
1. When opened, the plugin requests `GET /api/models` (or reads local `setup_config.json` and `models/wan/configs/*.py`).
2. The GDScript dock iterates through the schema and generates native Godot `Control` nodes:
   * String fields $\rightarrow$ `LineEdit` / `TextEdit`
   * Int / Float ranges $\rightarrow$ `HSlider` with synchronized `SpinBox`
   * Enum / Category choices $\rightarrow$ `OptionButton`
3. When upstream adds new models or modifies slider maximums, **the Godot Editor UI immediately adapts without modifying a single line of GDScript**.

### B. High-Performance Chunked Streaming (Solving the >16 MB Limit)
High-resolution 720p/1080p video outputs from Wan 2.1 frequently exceed 16 MB, which can trigger gateway buffering errors or timeouts when downloaded via standard unchunked HTTP endpoints.

The Godot plugin implements chunked streaming via `HTTPRequest`:
* Streams files in **2 MB binary chunks** directly into Godot's project directory (`res://assets/wan_outputs/`).
* Never buffers the full video in RAM, maintaining Godot's lightweight performance.
* Automatically verifies file integrity via MD5/SHA256 headers before triggering Godot's asset re-import.

### C. Direct Project Asset Auto-Import
Upon download completion, the plugin calls Godot's editor filesystem interface:
```gdscript
EditorInterface.get_resource_filesystem().scan()
```
* **Videos (`.mp4`)**: Automatically recognized by Godot's media pipeline, ready to attach to `VideoStreamPlayer`.
* **3D Models (`.glb`)**: Automatically imported as 3D scenes / materials from models like Trellis.2 or Hunyuan3D-2.1.
* **Images (`.png`)**: Auto-imported as `CompressedTexture2D`.

---

## 4. Proposed Plugin Structure (`addons/wan2gp/`)

The plugin is designed to be dropped into any Godot 4.x project:

```text
addons/wan2gp/
├── plugin.cfg                  # Godot EditorPlugin manifest
├── wan2gp_plugin.gd            # Registers the Editor dock and toolbar buttons
├── ui/
│   ├── wan_dock.tscn           # Base UI layout container
│   ├── wan_dock.gd             # Dynamic schema parser & control builder
│   └── components/
│       ├── parameter_slider.gd # Reusable synchronized slider + spinbox
│       └── gallery_preview.gd  # Thumbnail & video playback widget
└── core/
    ├── cloud_client.gd         # HTTP / SSE client targeting Modal / RunPod
    └── asset_importer.gd       # File stream writer and EditorFileSystem scanner
```

---

## 5. Upstream Lockstep Maintenance

To ensure this architecture remains permanently compatible with upstream:

1. **Local & Cloud Sync**: Use [`sync_upstream_and_modal.py`](../sync_upstream_and_modal.py) to pull new upstream releases (`git fetch upstream main`) and sync them to your Modal worker volume.
2. **Contract Preservation**: All communication conforms to the normalized `GenerationRequest` dataclass defined in [`shared/execution/types.py`](../shared/execution/types.py).
3. **No Code Duplication**: All model definitions remain in upstream's format; Godot acts strictly as a lightweight, reactive studio frontend.
