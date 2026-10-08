# Wan2GP Headless Mode Reference

> **Source of truth**: All parameter names, keys, and defaults are read directly from the Wan2GP source code — specifically `models/_settings.json` (the global `primary_settings` baseline), `defaults/*.json` (per-model overrides), and `shared/execution/types.py` / `wgp.py` (`validate_settings`, `add_video_task`, `_parse_task_manifest`).  
> No parameter is invented — every key below maps 1-to-1 with a Gradio input component inside `wgp.py`.

---

## Table of Contents

1. [How Headless Mode Works](#1-how-headless-mode-works)
2. [CLI Entry Points](#2-cli-entry-points)
3. [queue.zip — The Package Format](#3-queuezip--the-package-format)
4. [queue.json — Schema Overview](#4-queuejson--schema-overview)
5. [Task Object — Full Field Reference](#5-task-object--full-field-reference)
6. [model_type Values](#6-model_type-values)
7. [image_prompt_type Letter Flags](#7-image_prompt_type-letter-flags)
8. [video_prompt_type Letter Flags](#8-video_prompt_type-letter-flags)
9. [audio_prompt_type Letter Flags](#9-audio_prompt_type-letter-flags)
10. [multi_prompts_gen_type Values](#10-multi_prompts_gen_type-values)
11. [Per-Model Parameter Overrides](#11-per-model-parameter-overrides)
12. [Minimal Examples](#12-minimal-examples)
13. [Error Handling & Validation Flow](#13-error-handling--validation-flow)

---

## 1. How Headless Mode Works

When Wan2GP runs with `--process`, it **does not launch the Gradio web UI** at all.  
Instead it:

1. Parses your `.json` or `.zip` into an in-memory queue via `_parse_task_manifest()`.
2. Runs `validate_settings()` for each task — **exactly the same validation the Gradio UI does** on submit.
3. Passes each valid task through the full generation pipeline (model load → inference → VAE decode → save output).
4. Writes outputs to the configured `save_path` (default: `outputs/`).

```
wgp.py
 └── _parse_queue_zip_tasks()  / _parse_json_tasks()
      └── _parse_task_manifest()
           └── validate_settings()   ← same code as UI "Add to Queue"
                └── generation pipeline
```

---

## 2. CLI Entry Points

```bash
# Process a single settings file (one task)
python wgp.py --process /path/to/settings.json

# Process a saved queue bundle (multiple tasks + media)
python wgp.py --process /path/to/queue.zip

# Dry-run (validate only, no generation)
python wgp.py --process /path/to/queue.zip --dry-run

# Override output directory
python wgp.py --process /path/to/queue.zip --output-dir /custom/outputs

# Force Cloud mode (dispatch to remote worker)
python wgp.py --process /path/to/queue.zip --mode cloud

# Use a specific config folder (wgp_config.json + queue.zip autosave location)
python wgp.py --process /path/to/queue.zip --config /config/dir
```

> **Note:** `--process` accepts a `.json` that is either a single task `params` dict, or a `queue.json`-formatted array. Both are detected automatically.

---

## 3. queue.zip — The Package Format

A `queue.zip` is a standard ZIP archive with the following structure:

```
queue.zip
├── queue.json                    ← REQUIRED — the task manifest
├── task1_image_start_0.png       ← media referenced from queue.json params
├── task1_video_guide_0.mp4
├── task2_audio_guide_0.wav
└── ...
```

**How media paths work inside the zip:**  
When the Gradio UI saves a queue (`Save Queue` button), it serializes each media attachment into files named:

```
task{task_id}_{key}_{index}{extension}
```

For example: `task3_image_start_0.png`, `task3_video_guide_0.mp4`.

Inside `queue.json`, the corresponding params field then holds the **relative filename** (not an absolute path):

```json
"image_start": "task3_image_start_0.png"
```

When `--process` loads the zip, it extracts everything to a temp directory and resolves those relative paths automatically.

---

## 4. queue.json — Schema Overview

`queue.json` is a **JSON array** of task objects:

```json
[
  {
    "id": 1,
    "params": { ... },
    "execution_mode": "local",
    "plugin_data": {}
  },
  {
    "id": 2,
    "params": { ... },
    "execution_mode": "cloud"
  }
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | int | No | Task ID. Auto-assigned if omitted. |
| `params` | object | **Yes** | All generation parameters (see §5). |
| `execution_mode` | string | No | `"local"` or `"cloud"`. Defaults to `"local"`. |
| `plugin_data` | object | No | Data passed to plugins. Usually empty `{}`. |

> **Single-file shortcut:** If you pass a `.json` that is a plain object (not an array), Wan2GP wraps it in an array automatically.

---

## 5. Task Object — Full Field Reference

All fields live inside `params`. The source of truth for keys and defaults is [`models/_settings.json`](file:///run/media/rasal/FAFF-6A46/dirve%20Rasal/work/wan2gp%20fine%20tuned/Wan2GP/models/_settings.json).

### 5.1 Identity & Routing

| Key | Type | Default | Description |
|---|---|---|---|
| `model_type` | string | *(required)* | Which model to use. See §6 for all valid values. |
| `base_model_type` | string | *(derived)* | Base architecture (e.g. `"t2v"`, `"i2v"`). Auto-derived; normally do not set this. |
| `settings_version` | float | `2.79` | Settings schema version. Enables auto-migration of older settings. |
| `execution_mode` | string | `"local"` | `"local"` or `"cloud"`. Also settable at the task level (outside `params`). |
| `client_id` | string | `""` | Optional client identifier for tracking/logging. |
| `output_filename` | string | `""` | Override the generated output filename (no extension). |
| `config` | string | `""` | Override path to a config folder for this task. |

---

### 5.2 Core Generation Settings

| Key | Type | Default | Gradio Label | Description |
|---|---|---|---|---|
| `prompt` | string | `""` | **Prompt** | The main generation prompt. Supports multi-prompt syntax (see §10). |
| `negative_prompt` | string | `""` | Negative Prompt | Negative prompt for CFG. |
| `alt_prompt` | string | `""` | Alt / Secondary Prompt | Secondary prompt for dual-text-encoder models (e.g. Wan2.2). |
| `resolution` | string | `"832x480"` | Resolution | Output resolution as `"WIDTHxHEIGHT"`. Parsed into `width`/`height` automatically. Common values: `"832x480"`, `"1280x720"`, `"1920x1080"`, `"480x832"` (portrait). |
| `video_length` | int | `81` | Number of Frames | Total output frames. Must satisfy model's frame-step requirement. |
| `duration_seconds` | float | `0` | Duration (s) | Alternative to `video_length`. Set > 0 to compute frame count from FPS. |
| `force_fps` | string/float | `""` | Force FPS | Override the output FPS (e.g. `"24"`, `"16"`). Empty = model default. |
| `num_inference_steps` | int | `30` | Steps | Diffusion denoising steps. |
| `seed` | int | `-1` | Seed | RNG seed. `-1` = random per generation. |
| `batch_size` | int | `1` | Batch Size | Number of images to generate per task. For image models only (`image_mode > 0`). |
| `repeat_generation` | int | `1` | Repeat | Number of times to repeat the full generation. |
| `image_mode` | int | `0` | Output Mode | `0` = video output, `1` = image output, `2` = alternate image mode. |

---

### 5.3 Prompt & Multi-Prompt Control

| Key | Type | Default | Description |
|---|---|---|---|
| `multi_prompts_gen_type` | string | `"PG"` | Controls how multi-line prompts are split. See §10. |
| `prompt_enhancer` | string | `""` | LLM prompt enhancer mode. Empty = disabled. Add `"K"` flag to keep original alongside enhanced. |
| `pause_seconds` | float | `0` | Seconds to pause between repeated generations. |

---

### 5.4 Image / Video Mode Control

| Key | Type | Default | Description |
|---|---|---|---|
| `image_prompt_type` | string | `""` | Which still-image inputs to use. See §7. |
| `video_prompt_type` | string | `""` | Which video/control inputs to use. See §8. |
| `audio_prompt_type` | string | `""` | Which audio inputs to use. See §9. |
| `model_mode` | string/null | `null` | Sub-mode for models that expose multiple operating modes. `null` = default mode. |
| `keep_frames_video_source` | string | `""` | Frames to keep from `video_source` when continuing a video. Integer string like `"16"`. Requires `"V"` or `"L"` in `image_prompt_type`. |
| `keep_frames_video_guide` | string | `""` | Sparse frame selection from the control video. Format: `"start:end"` or comma-separated frame indices. |
| `frames_positions` | string | `""` | Explicit frame positions for multi-frame reference injection. Space/comma separated integers, `"L"` (last), or `"X"` (skip). Requires `"F"` in `video_prompt_type`. |
| `input_video_strength` | float | `1.0` | Blending strength for the source video during video continuation (0.0–1.0). |
| `denoising_strength` | float | `1.0` | For Video-to-Video (`"G"` in `video_prompt_type`): how much noise to add. `1.0` = full noise, `0.0` = copy source. |
| `masking_strength` | float | `1.0` | For Masked Video-to-Video (`"GA"` in `video_prompt_type`): how long masking is applied (fraction of steps). |
| `mask_expand` | int | `0` | Pixel radius to expand/dilate the provided mask. |
| `video_guide_outpainting` | string | `"#"` | Outpainting mode for control video. `"#"` = disabled. |
| `video_guide_outpainting_ratio` | string | `""` | Ratio string for outpainting layout. |
| `image_refs_relative_size` | int | `50` | Relative size of reference images when composited (0–100%). |
| `remove_background_images_ref` | int | `1` | `1` = remove background from reference images, `0` = keep full image. |
| `min_frames_if_references` | int | `1` | Minimum frames to generate when reference images are provided. |

---

### 5.5 Guidance & Scheduler Control

| Key | Type | Default | Description |
|---|---|---|---|
| `guidance_scale` | float | `5.0` | CFG guidance scale for phase 1 (primary/only phase for Wan2.1 models). |
| `guidance2_scale` | float | `5.0` | CFG guidance scale for phase 2 (Wan2.2 dual-phase models). |
| `guidance3_scale` | float | `5.0` | CFG guidance scale for phase 3. |
| `guidance_phases` | int | `1` | Number of guidance phases. `1` = standard, `2` = dual-phase (Wan2.2), `3` = 3-phase. Clamped to model's `guidance_max_phases`. |
| `switch_threshold` | int | `0` | Phase 1→2 switch noise level (0–1000, noise goes from 1000→0). Active when `guidance_phases >= 2`. |
| `switch_threshold2` | int | `0` | Phase 2→3 switch noise level. Must be < `switch_threshold`. Active when `guidance_phases == 3`. |
| `model_switch_phase` | int | `1` | Which guidance phase triggers a model weight switch. Model-specific. |
| `embedded_guidance_scale` | float | `6.0` | Embedded guidance scale for models that support it (e.g. HunyuanVideo). |
| `audio_guidance_scale` | float | `1.0` | Guidance scale for audio conditioning. |
| `audio_scale` | float | `1.0` | Overall audio influence scale. |
| `alt_guidance_scale` | float | `1.0` | Alternative/auxiliary guidance scale. |
| `alt_scale` | float | `0.0` | Scale of auxiliary branch outputs blended with main branch. |
| `NAG_scale` | float | `1.0` | Normalized Attention Guidance scale. `1.0` = disabled. |
| `NAG_tau` | float | `3.5` | NAG threshold parameter. |
| `NAG_alpha` | float | `0.5` | NAG alpha blending factor. |
| `apg_switch` | int | `0` | `1` = enable Adaptive Progressive Guidance. Cannot combine with `cfg_star_switch`. |
| `cfg_star_switch` | int | `0` | `1` = enable CFG Star. Cannot combine with `apg_switch`. |
| `cfg_zero_step` | int | `-1` | Step at which CFG Zero kicks in. `-1` = disabled. |
| `flow_shift` | float | `3.0` | Flow-matching shift parameter. Affects noise schedule. Varies by model (see §11). |
| `sample_solver` | string | `""` | Sampler/solver override. `""` = model default. Values: `""`, `"unipc"`, `"dpm"`. |
| `temperature` | float | `0.8` | Sampling temperature for LM-based models (e.g. ACE-Step). |
| `top_p` | float | `0.9` | Top-p (nucleus) sampling cutoff for LM-based models. |
| `top_k` | int | `50` | Top-k sampling cutoff for LM-based models. |
| `control_net_weight` | float | `1.0` | ControlNet weight for the primary control input. |
| `control_net_weight2` | float | `1.0` | ControlNet weight for the secondary control input. |
| `control_net_weight_alt` | float | `1.0` | ControlNet weight for an alternate/auxiliary control channel. |
| `motion_amplitude` | float | `1.0` | Motion amplitude multiplier. Only active on models with `"motion_amplitude": true` in model def. |

---

### 5.6 LoRA Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `activated_loras` | list[string] | `[]` | List of LoRA filenames (relative to the model's `loras/` subdirectory). E.g. `["my_style.safetensors"]`. |
| `loras_multipliers` | string | `""` | Multiplier specification. Formats: `"0.8"` (global), `"0.8,1.0"` (per-LoRA), or phase/step ranges like `"0.8@0:500,1.0@500:1000"`. Empty = all LoRAs use multiplier 1.0. |

---

### 5.7 Sliding Window Settings

Sliding Windows are automatically activated when `video_length > sliding_window_size` on supported models.

| Key | Type | Default | Description |
|---|---|---|---|
| `sliding_window_size` | int | `129` | Size of each sliding window (frames). |
| `sliding_window_overlap` | int | `5` | Number of frames overlapping between consecutive windows. |
| `sliding_window_discard_last_frames` | int | `0` | Frames to discard from the end of each window before stitching. Rounded to latent size. |
| `sliding_window_trim_first_frames` | int | `0` | Frames to trim from the very start of the final output. |
| `sliding_window_overlap_noise` | float | `0` | Amount of noise added to overlapping frames on each new window (prevents color drift). Range 0.0–1.0. |
| `sliding_window_color_correction_strength` | float | `0` | Strength of inter-window color correction. 0 = off. |
| `sub_parallel_window_size` | int | `0` | Sub-window size for parallel generation (experimental). `0` = disabled. |
| `sub_parallel_window_overlap` | int | `17` | Overlap for sub-parallel windows. |
| `RIFLEx_setting` | int | `0` | `0` = off, `1` = enable RIFLEx (Rope Interpolation for Flexible Length Extension) for long video generation. |

---

### 5.8 Skip-Steps / Step-Caching

Step-caching skips computing certain layers in non-critical denoising steps to speed up inference.

| Key | Type | Default | Possible Values | Description |
|---|---|---|---|---|
| `skip_steps_cache_type` | string | `""` | `""`, `"tea"`, `"mag"`, `"spectrum"`, `"first_block"` | Cache type. Must be supported by the chosen model. `""` = no caching. |
| `skip_steps_multiplier` | float | `1.75` | Model-specific | Cache aggressiveness. Higher = more steps skipped = faster but lower quality. |
| `skip_steps_start_step_perc` | float | `0` | `0`–`100` | Start applying caching after this % of total steps. `0` = from beginning. |

---

### 5.9 Post-Processing — Spatial & Temporal Upsampling

| Key | Type | Default | Description |
|---|---|---|---|
| `spatial_upsampling` | string | `""` | Spatial (resolution) upsampler after generation. `""` = none. E.g. `"realesr"`, `"topaz"`, `"vae_1080p"`. |
| `spatial_upsampler_prompt` | string | `""` | Prompt for the spatial upsampler (for models that accept prompts). |
| `spatial_upsampler_reference_images` | list/null | `null` | Reference images for the spatial upsampler. |
| `spatial_upsampler_param` | any/null | `null` | First extra parameter for the spatial upsampler (model-specific). |
| `spatial_upsampler_param2` | any/null | `null` | Second extra parameter for the spatial upsampler. |
| `temporal_upsampling` | string | `""` | Temporal (frame-rate) upsampler. `""` = none. E.g. `"rife"`, `"film"`. |

---

### 5.10 Post-Processing — Audio

| Key | Type | Default | Description |
|---|---|---|---|
| `postprocess_audio` | string | `""` | Soundtrack post-processor. `""` = none. `"control"` = use `audio_source` as-is. Not compatible with `image_mode > 0`. |
| `postprocess_audio_prompt` | string | `""` | Prompt for soundtrack generation. |
| `postprocess_audio_neg_prompt` | string | `""` | Negative prompt for soundtrack generation. |
| `speakers_locations` | string | `"0:45 55:100"` | Speaker bounding boxes for Multitalk models. Format: `"left_pct:right_pct"` space-separated pairs per speaker. Used with `"B"` or `"X"` in `audio_prompt_type`. |
| `replace_voice_method` | string | `""` | Voice replacement processor ID. `""` = none. |

---

### 5.11 Advanced / Fine-Tuning Controls

| Key | Type | Default | Description |
|---|---|---|---|
| `perturbation_switch` | int | `0` | `1` = enable attention perturbation (diversifies outputs). |
| `perturbation_layers` | list[int] | `[9]` | Transformer layer indices to apply perturbation to. |
| `perturbation_start_perc` | int | `10` | Perturbation starts at this % of total steps. |
| `perturbation_end_perc` | int | `90` | Perturbation ends at this % of total steps. |
| `self_refiner_setting` | int | `0` | `0` = off. `1`+ = enable Self-Refiner (iterative self-improvement). |
| `self_refiner_plan` | string | `""` | Self-refiner plan string specifying refinement schedule. |
| `self_refiner_f_uncertainty` | float | `0.0` | Frame uncertainty threshold for selective refinement. |
| `self_refiner_certain_percentage` | float | `0.999` | Fraction of "certain" frames below which refinement is triggered. |
| `film_grain_intensity` | float | `0` | Film grain intensity overlay. 0 = off. |
| `film_grain_saturation` | float | `0.5` | Film grain color saturation. |
| `override_profile` | int | `-1` | Override the loaded model memory profile index. `-1` = server default. |
| `override_attention` | string | `""` | Override the attention implementation for this task. `""` = server default. |
| `attention_sparsity` | float | `1.0` | Attention sparsity factor (0.0–1.0). `1.0` = no sparsity. |
| `multi_images_gen_type` | int | `0` | How to handle multiple start images: `0` = animate together, `1` = one generation per image. |
| `custom_settings` | object/null | `null` | Model-specific extra settings defined in the model def JSON. Structure is model-dependent. |

---

### 5.12 Media Attachments (Inputs)

All media fields accept either:
- `null` (not used)
- An **absolute filesystem path** (when running locally)
- A **relative filename** inside the zip archive (when packed into `queue.zip`)
- A **list of paths** (for fields that accept multiple images)

| Key | Type | Activated By | Description |
|---|---|---|---|
| `image_start` | string/null | `"S"` in `image_prompt_type` | Start image path (single image). |
| `image_end` | list[string]/null | `"E"` in `image_prompt_type` | End image(s) path. |
| `image_refs` | list[string]/null | `"I"` in `video_prompt_type` | Reference image(s) for identity/style. |
| `image_guide` | string/null | `"V"` in `video_prompt_type` + `image_mode > 0` | Control image for image-output controlnet. |
| `image_mask` | string/null | `"VA"` in `video_prompt_type` + `image_mode > 0` | Mask image for masked image-to-image. |
| `video_guide` | string/null | `"V"` in `video_prompt_type` | Primary control video (pose, depth, etc.). |
| `video_guide2` | string/null | `"V+"` in `video_prompt_type` | Secondary control video. |
| `video_guide3` | string/null | `"V*"` in `video_prompt_type` | Tertiary control video. |
| `video_mask` | string/null | `"VA"` in `video_prompt_type` | Video mask for masked video-to-video. |
| `video_source` | string/null | `"V"` or `"L"` in `image_prompt_type` | Source video for video continuation / video-to-video. |
| `audio_guide` | string/null | `"A"` in `audio_prompt_type` | Primary audio input (speech, music). |
| `audio_guide2` | string/null | `"B"` in `audio_prompt_type` | Second speaker / second audio track. |
| `audio_guide3` | string/null | `"D"` in `audio_prompt_type` | Third audio track. |
| `audio_source` | string/null | `postprocess_audio != ""` | Source audio for soundtrack post-processing or `"K"` in `audio_prompt_type`. |
| `custom_guide` | string/null | Model-specific | Custom media input defined per-model. |
| `replace_voice_sample` | string/null | `replace_voice_method != ""` | Primary voice sample for voice cloning/replacement. |
| `replace_voice_sample2` | string/null | `replace_voice_method != ""` | Secondary voice sample (if method requires two). |

---

## 6. model_type Values

`model_type` must be the **filename stem** of any `.json` file in the `defaults/` folder, or any `.json` file in the `finetunes/` folder.

### Core Wan2.1 Models

| `model_type` | Name | Architecture |
|---|---|---|
| `t2v` | Wan2.1 Text2Video 14B | `t2v` |
| `t2v_1.3B` | Wan2.1 Text2Video 1.3B | `t2v` |
| `i2v` | Wan2.1 Image2Video 480p 14B | `i2v` |
| `i2v_720p` | Wan2.1 Image2Video 720p 14B | `i2v` |
| `fun_inp` | Wan2.1 FunControl Inpainting 14B | `fun_inp` |
| `fun_inp_1.3B` | Wan2.1 FunControl Inpainting 1.3B | `fun_inp` |

### Core Wan2.2 Models

| `model_type` | Name | Architecture |
|---|---|---|
| `t2v_2_2` | Wan2.2 Text2Video 14B | `t2v_2_2` |
| `i2v_2_2` | Wan2.2 Image2Video 14B | `i2v_2_2` |
| `vace_14B_2_2` | Wan2.2 VACE 14B | `vace_14B` |

### VACE ControlNet Models (Wan2.1 base)

| `model_type` | Name |
|---|---|
| `vace_14B` | VACE 14B |
| `vace_1.3B` | VACE 1.3B |
| `vace_14B_cocktail` | VACE 14B Cocktail (multi-control) |
| `vace_14B_sf` | VACE 14B Stepflow |

### HunyuanVideo Models

| `model_type` | Name |
|---|---|
| `hunyuan_i2v` | HunyuanVideo I2V |
| `hunyuan_t2v_fast` | HunyuanVideo T2V Fast |
| `hunyuan_custom` | HunyuanVideo Custom |
| `hunyuan_avatar` | HunyuanVideo Avatar |

### LTX Video Models

| `model_type` | Name |
|---|---|
| `ltxv_13B` | LTX Video 13B |
| `ltxv_distilled` | LTX Video Distilled |
| `ltx2_19B` | LTX2 19B |
| `ltx2_22B` | LTX2 22B |

### FLUX Image Models

| `model_type` | Name |
|---|---|
| `flux` | FLUX.1 Dev |
| `flux_schnell` | FLUX.1 Schnell |
| `flux2_dev` | FLUX2 Dev |
| `flux_dev_kontext` | FLUX Dev Kontext |

> **Custom Finetunes:** Place your `.json` in `finetunes/` and use the stem as `model_type`. The JSON must follow the model def format: `{"model": {"architecture": "<base_arch>", ...}, "setting_key": value, ...}`.

> **All valid model_type values** can be found by listing the `defaults/` directory — there are 237 total model definition files across all architectures.

---

## 7. image_prompt_type Letter Flags

`image_prompt_type` controls which **still-image inputs** feed into generation. Build the value by concatenating letters.

| Letter | Meaning | Required Input | Notes |
|---|---|---|---|
| `S` | Use **Start Image** | `image_start` | Anchors the first frame. Required for I2V conditioning. |
| `E` | Use **End Image** | `image_end` | Anchors the final frame. Requires `S`, `V`, or `L` active. |
| `V` | Use **Source Video** for continuation | `video_source` | Continue an existing video. Sets the initial frames from `video_source`. |
| `L` | **Long video continuation** | `video_source` | Same as `V` but using a long-form continuation method. |
| `G` | Source video with **denoising** | `video_source` + `denoising_strength` | Video-to-video with noise-based transformation. |

**Common combinations:**

| Value | Meaning |
|---|---|
| `""` | Pure text-to-video, no image conditioning |
| `"S"` | Start image only (standard I2V) |
| `"SE"` | Start + End image (FLF2V / flf models) |
| `"SV"` | Animate from start image, continuing source video |
| `"VG"` | Video-to-video with denoising |

---

## 8. video_prompt_type Letter Flags

`video_prompt_type` controls **control signals** from video sources (ControlNet-style).

| Letter | Meaning | Required Input | Notes |
|---|---|---|---|
| `V` | Use primary **Control Video** | `video_guide` | Enables ControlNet conditioning (depth, pose, etc.). |
| `+` | Use second Control Video | `video_guide2` | Requires `V`. |
| `*` | Use third Control Video | `video_guide3` | Requires `V+`. |
| `I` | Use **Reference Images** | `image_refs` | Injects identity/style from reference images. |
| `G` | **Guided generation** (denoising respect) | `video_guide` + `denoising_strength` | Control video also acts as generation base. |
| `A` | Use **Mask** (for inpainting) | `video_mask` / `image_mask` | Masked region is inpainted. Requires `V`. |
| `U` | **Unmask** (fill outside mask) | — | Inverts the mask region. Used with `A`. |
| `F` | Use **Frame positions** | `frames_positions` | Reference images placed at specific frame positions. |
| `K` | Use **Control Video Audio** | `video_guide` with audio track | Extracts audio from control video for audio conditioning. |
| `O` | **Aligned** pose/depth transfer | — | Aligns the control video to the scene. Requires start image or source video. |
| `N` | **Noise** control injection | — | Injects noise pattern from control video. |

**Common combinations:**

| Value | Meaning |
|---|---|
| `""` | No video control inputs |
| `"V"` | Standard ControlNet control video |
| `"V+"` | Dual control video |
| `"VGA"` | Video-to-video with denoising + inpainting mask |
| `"VI"` | ControlNet + reference image identity |
| `"VIF"` | ControlNet + reference images at specific frame positions |

---

## 9. audio_prompt_type Letter Flags

`audio_prompt_type` controls **audio conditioning** for audio-capable models.

| Letter | Meaning | Required Input | Notes |
|---|---|---|---|
| `A` | Use **primary audio** | `audio_guide` | First speaker audio / main audio track. |
| `B` | Use **second speaker** audio | `audio_guide2` | Dual-speaker mode (Multitalk). |
| `D` | Use **third audio** | `audio_guide3` | Third audio track. |
| `X` | **Mix** mode (multi-speaker spatial) | `audio_guide`, `audio_guide2` | Requires `speakers_locations`. |
| `N` | **Noise** from audio | — | Injects audio-derived noise. Requires `A` or `B`. |
| `K` | Use **audio source** track | `audio_source` | Use audio from `audio_source` directly. Requires `V` in `video_prompt_type`. |

---

## 10. multi_prompts_gen_type Values

Controls how newlines in `prompt` are interpreted when there are multiple lines.

| Value | Meaning |
|---|---|
| `"FG"` | **Free — one generation**: All lines are one single prompt (newlines ignored as separators). |
| `"PG"` | **Prompt-per-generation** (default): Each paragraph (double newline) becomes a separate queued generation. |
| `"G"` | **Line-per-generation**: Each line becomes a separate generation. |
| `"W"` | **Window-per-line**: Each line is a prompt for a different sliding window shot. Requires a sliding window model. |
| `"PW"` | **Paragraph-per-window**: Each paragraph is a prompt for a different window. |
| `"WG"` | **Window+Generation**: Combines window splitting with multiple generations. |

> **Tip:** Use `"FG"` for simple single-generation use cases. Use `"W"` or `"PW"` for long-form multi-shot videos.

---

## 11. Per-Model Parameter Overrides

Each model's `defaults/<model_type>.json` contains flat key-value overrides that replace the global defaults. These apply automatically when you set `model_type`. You only need to override keys that differ from those in your task.

**Key differences from source:**

| Model | Key | Override Value | Reason |
|---|---|---|---|
| `t2v_2_2` | `guidance_phases` | `2` | Wan2.2 uses dual-phase guidance |
| `t2v_2_2` | `switch_threshold` | `875` | Phase 1→2 noise level |
| `t2v_2_2` | `guidance_scale` | `4` | Lower CFG for 2.2 |
| `t2v_2_2` | `guidance2_scale` | `3` | Phase 2 CFG |
| `t2v_2_2` | `flow_shift` | `12` | Higher flow shift for 2.2 |
| `i2v_2_2` | `guidance_phases` | `2` | Same dual-phase |
| `i2v_2_2` | `switch_threshold` | `900` | I2V 2.2 switch point |
| `i2v_2_2` | `masking_strength` | `0.1` | I2V 2.2 uses light masking |
| `i2v_2_2` | `denoising_strength` | `0.9` | I2V 2.2 denoising |
| `i2v_2_2` | `flow_shift` | `5` | I2V 2.2 flow |
| `i2v` (480p) | `flow_shift` | `5.0` | Lower for I2V |
| `i2v_720p` | `flow_shift` | `7.0` | Higher for 720p I2V |
| `i2v_720p` | `resolution` | `"1280x720"` | Default 720p |
| `t2v` / `t2v_1.3B` | `resolution` | `"832x480"` | Default 480p |

---

## 12. Minimal Examples

### 12.1 Text-to-Video (T2V) — settings.json only

```json
[
  {
    "id": 1,
    "params": {
      "model_type": "t2v",
      "prompt": "A cinematic wide-angle shot of a golden retriever running on a beach at sunset, slow motion, film grain, high detail",
      "negative_prompt": "blurry, low quality, watermark",
      "resolution": "832x480",
      "video_length": 81,
      "num_inference_steps": 30,
      "guidance_scale": 5.0,
      "flow_shift": 3.0,
      "seed": 42,
      "image_prompt_type": "",
      "video_prompt_type": "",
      "audio_prompt_type": "",
      "multi_prompts_gen_type": "FG",
      "activated_loras": [],
      "loras_multipliers": "",
      "image_start": null,
      "image_end": null,
      "image_refs": null,
      "video_guide": null,
      "video_source": null,
      "audio_guide": null
    },
    "execution_mode": "local"
  }
]
```

```bash
python wgp.py --process t2v_task.json
```

---

### 12.2 Image-to-Video (I2V) — queue.zip with media

**queue.json** (inside the zip):

```json
[
  {
    "id": 1,
    "params": {
      "model_type": "i2v",
      "prompt": "The dog slowly turns its head and wags its tail",
      "negative_prompt": "blurry, distorted",
      "resolution": "832x480",
      "video_length": 81,
      "num_inference_steps": 30,
      "guidance_scale": 5.0,
      "flow_shift": 5.0,
      "seed": -1,
      "image_prompt_type": "S",
      "video_prompt_type": "",
      "audio_prompt_type": "",
      "multi_prompts_gen_type": "FG",
      "image_start": "task1_image_start_0.png",
      "image_end": null,
      "image_refs": null,
      "video_guide": null,
      "video_source": null,
      "audio_guide": null,
      "activated_loras": [],
      "loras_multipliers": ""
    }
  }
]
```

**ZIP structure:**
```
queue.zip
├── queue.json
└── task1_image_start_0.png
```

```bash
python wgp.py --process i2v_bundle.zip
```

---

### 12.3 VACE ControlNet Video-to-Video

```json
[
  {
    "id": 1,
    "params": {
      "model_type": "vace_14B",
      "prompt": "A knight walking in a medieval village, cinematic lighting",
      "negative_prompt": "",
      "resolution": "832x480",
      "video_length": 81,
      "num_inference_steps": 30,
      "guidance_scale": 5.0,
      "flow_shift": 3.0,
      "seed": -1,
      "image_prompt_type": "",
      "video_prompt_type": "VGA",
      "audio_prompt_type": "",
      "multi_prompts_gen_type": "FG",
      "denoising_strength": 0.85,
      "masking_strength": 1.0,
      "control_net_weight": 1.0,
      "video_guide": "task1_video_guide_0.mp4",
      "video_mask": "task1_video_mask_0.mp4",
      "image_start": null,
      "image_end": null,
      "video_source": null,
      "audio_guide": null,
      "image_refs": null,
      "activated_loras": ["my_lora.safetensors"],
      "loras_multipliers": "0.8"
    }
  }
]
```

---

### 12.4 Multi-task queue.json (Wan2.2)

```json
[
  {
    "id": 1,
    "params": {
      "model_type": "t2v_2_2",
      "prompt": "A lone wolf howling at the moon on a snowy mountain peak",
      "resolution": "1280x720",
      "video_length": 81,
      "num_inference_steps": 30,
      "guidance_scale": 4.0,
      "guidance2_scale": 3.0,
      "guidance_phases": 2,
      "switch_threshold": 875,
      "flow_shift": 12,
      "seed": 100,
      "image_prompt_type": "",
      "video_prompt_type": "",
      "audio_prompt_type": "",
      "multi_prompts_gen_type": "FG",
      "activated_loras": []
    },
    "execution_mode": "local"
  },
  {
    "id": 2,
    "params": {
      "model_type": "i2v_2_2",
      "prompt": "The wolf shakes the snow off its fur and trots away",
      "resolution": "832x480",
      "video_length": 81,
      "num_inference_steps": 30,
      "guidance_scale": 3.5,
      "guidance2_scale": 3.5,
      "guidance_phases": 2,
      "switch_threshold": 900,
      "denoising_strength": 0.9,
      "masking_strength": 0.1,
      "flow_shift": 5,
      "seed": -1,
      "image_prompt_type": "S",
      "image_start": "task2_image_start_0.png",
      "video_prompt_type": "",
      "audio_prompt_type": "",
      "multi_prompts_gen_type": "FG",
      "activated_loras": []
    },
    "execution_mode": "local"
  }
]
```

---

## 13. Error Handling & Validation Flow

When `--process` runs, each task goes through `validate_settings()` which checks:

1. **Unknown keys** — Any key not in `models/_settings.json` or model custom settings is rejected.
2. **`prompt`** — Must not be empty after template processing.
3. **`resolution`** — Must be parseable as `"WxH"`.
4. **`video_length`** — Must meet model minimum frame count and frame-step divisibility.
5. **`guidance_phases`** — Clamped to model's `guidance_max_phases`.
6. **`switch_threshold`** — Must be > `switch_threshold2` when phases = 3.
7. **`skip_steps_cache_type`** — Must be supported by the model (`tea_cache`, `mag_cache`, etc. in model def).
8. **LoRAs** — Validated to exist in the model's `loras/` directory.
9. **Media attachments** — Required attachments (enforced by `image_prompt_type` / `video_prompt_type` / `audio_prompt_type` flags) must exist and be valid files.
10. **`apg_switch` + `cfg_star_switch`** — Cannot both be non-zero.

**If a task fails validation, it is skipped** and processing continues with the next task. Failed tasks are reported to the console.

With `--dry-run`, no generation runs but all validation errors are reported.

| Condition | Exit Code |
|---|---|
| All tasks succeeded | `0` |
| Some tasks failed validation (others ran) | `0` |
| All tasks failed / empty queue | Non-zero |
| `--dry-run` with validation errors | Non-zero |
| File not found / corrupt zip | Non-zero |
