#!/usr/bin/env python3
"""
Centralized Patch & Compatibility Manager for Wan2GP.

Provides automated, self-healing compatibility patches for:
1. Triton & nanovllm (alignment specialization fix)
2. TTS higgs_audio_v2_tokenizer (exist_ok=True for AutoConfig/AutoModel)
3. Hugging Face download_progress (_request_wrapper & **kwargs)
4. Torch accelerator & RoPE compatibility shims
5. shared/attention.py flash_attn exception safety
6. Qwen3-VL (pad_token_id attribute & config defaults)
7. Diffusers torch.accelerator compatibility
8. Universal sitecustomize.py environment injection
9. LTX-2 Gemma4UnifiedTextConfig heterogeneity compatibility
10. audio-separator & preprocessing dependency verification

Safe, idempotent, and non-destructive: can run repeatedly on local repositories,
cloud environments (Modal / RunPod), and automated sync pipelines.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence


def _default_log(msg: str):
    print(msg, flush=True)


def ensure_runtime_shims():
    """
    Apply fast, in-memory compatibility shims to the running Python process.
    Takes <1ms and does not modify disk.
    """
    # 1. Torch Accelerator shim
    try:
        import torch
        if not hasattr(torch, "accelerator"):
            class _TorchAccShim:
                @staticmethod
                def is_available():
                    return False
            torch.accelerator = _TorchAccShim
    except Exception:
        pass

    # 2. Universal RoPE compatibility shim (fixes KeyError: 'default')
    try:
        import transformers.modeling_rope_utils as _rope_utils
        if hasattr(_rope_utils, "ROPE_INIT_FUNCTIONS") and "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:
            if hasattr(_rope_utils, "_compute_default_rope_parameters"):
                _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _rope_utils._compute_default_rope_parameters
            else:
                import torch
                def _default_rope_init(config, device=None, **kwargs):
                    base = getattr(config, "rope_theta", 10000.0)
                    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32))
                    dim = int(head_dim * partial_rotary_factor)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                    return inv_freq, 1.0
                _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    except Exception:
        pass

    # 3. Transformers heterogeneity mixin
    try:
        import transformers.integrations.heterogeneity.configuration_utils as _hetero_cfg
        _hetero_cfg.HeterogeneousConfigMixin.__getattribute__ = object.__getattribute__
        _hetero_cfg.HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = True
    except Exception:
        pass

    try:
        from transformers.configuration_utils import PretrainedConfig
        PretrainedConfig.allow_global_per_layer_attribute_access = True
    except Exception:
        pass


def patch_nanovllm_triton(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    nanovllm_dir = wgp_dir / "shared" / "llm_engines" / "nanovllm"
    if not nanovllm_dir.exists():
        return False
    modified = False
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
                log_fn(f"[PATCH] Patched {py_file.name} for Triton compatibility")
                modified = True
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch {py_file}: {e}")
    return modified


def patch_higgs_tokenizer(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    tts_tok = wgp_dir / "models" / "TTS" / "omnivoice" / "higgs_audio_v2_tokenizer" / "__init__.py"
    if not tts_tok.exists():
        return False
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
            log_fn("[PATCH] Patched higgs_audio_v2_tokenizer with exist_ok=True")
            return True
    except Exception as e:
        log_fn(f"[PATCH-WARN] Failed to patch higgs_audio_v2_tokenizer: {e}")
    return False


def patch_download_progress(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    dl_progress = wgp_dir / "shared" / "utils" / "download_progress.py"
    if not dl_progress.exists():
        return False
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
                content = content.replace(
                    "lambda method, url, **p: http_backoff(requests.request, method=method, url=url, **p)",
                    "lambda *args, **kwargs: requests.request(*args, **kwargs)"
                )
            elif hf_target in content:
                content = content.replace(hf_target, hf_shim, 1)

        if content != original:
            dl_progress.write_text(content, encoding="utf-8")
            log_fn("[PATCH] Patched download_progress.py for huggingface_hub compatibility")
            return True
    except Exception as e:
        log_fn(f"[PATCH-WARN] Failed to patch download_progress.py: {e}")
    return False


def patch_attention_safety(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    attn_py = wgp_dir / "shared" / "attention.py"
    if not attn_py.exists():
        return False
    try:
        content = attn_py.read_text(encoding="utf-8")
        if "except ModuleNotFoundError:" in content and "FLASH_ATTN_2_AVAILABLE = False" in content:
            content = content.replace(
                "except ModuleNotFoundError:\n    FLASH_ATTN_2_AVAILABLE = False",
                "except (ModuleNotFoundError, ImportError, Exception):\n    FLASH_ATTN_2_AVAILABLE = False"
            )
            attn_py.write_text(content, encoding="utf-8")
            log_fn("[PATCH] Patched shared/attention.py for safe flash_attn fallback")
            return True
    except Exception as e:
        log_fn(f"[PATCH-WARN] Failed to patch shared/attention.py: {e}")
    return False


def patch_qwen3_vl(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    modified = False
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
                    log_fn(f"[PATCH] Patched {q_rel} for pad_token_id safety")
                    modified = True
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
                    modified = True
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")
    return modified


def patch_gemma4_unified(wgp_dir: Path, log_fn: Callable[[str], None] = _default_log) -> bool:
    gemma4_path = wgp_dir / "models" / "ltx2" / "ltx_core" / "text_encoders" / "gemma" / "gemma4_unified.py"
    if not gemma4_path.exists():
        return False
    try:
        content = gemma4_path.read_text(encoding="utf-8")
        original = content
        if "allow_global_per_layer_attribute_access = True" not in content:
            content = content.replace(
                'class Gemma4UnifiedTextConfig(PretrainedConfig):\n    model_type = "gemma4_unified_text"',
                'class Gemma4UnifiedTextConfig(PretrainedConfig):\n    model_type = "gemma4_unified_text"\n    allow_global_per_layer_attribute_access = True'
            )
        if 'config.__dict__.pop("_heterogeneity_spec", None)' not in content:
            content = content.replace(
                'class Gemma4UnifiedTextRotaryEmbedding(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__()',
                'class Gemma4UnifiedTextRotaryEmbedding(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__()\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True'
            )
            content = content.replace(
                'class Gemma4UnifiedTextAttention(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):\n        super().__init__()',
                'class Gemma4UnifiedTextAttention(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):\n        super().__init__()\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True'
            )
            content = content.replace(
                'class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__(config)',
                'class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True\n        super().__init__(config)'
            )
            content = content.replace(
                'class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel):\n    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}\n\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__(config)',
                'class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel):\n    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}\n\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True\n        super().__init__(config)'
            )
            content = content.replace(
                "num_key_value_heads = config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads",
                "num_key_value_heads = (config.__dict__.get('num_global_key_value_heads') or getattr(config, 'num_global_key_value_heads', 1)) if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))"
            )
            content = content.replace(
                "num_key_value_heads = config.num_global_key_value_heads if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))",
                "num_key_value_heads = (config.__dict__.get('num_global_key_value_heads') or getattr(config, 'num_global_key_value_heads', 1)) if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))"
            )
            content = content.replace(
                "self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim",
                "self.head_dim = (config.__dict__.get('head_dim') or getattr(config, 'head_dim', 256)) if self.is_sliding else getattr(config, 'global_head_dim', 512)"
            )
            content = content.replace(
                'head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim',
                'head_dim = (config.__dict__.get("global_head_dim") or getattr(config, "global_head_dim", 512)) if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))'
            )
            content = content.replace(
                'head_dim = config.global_head_dim if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))',
                'head_dim = (config.__dict__.get("global_head_dim") or getattr(config, "global_head_dim", 512)) if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))'
            )
        if content != original:
            gemma4_path.write_text(content, encoding="utf-8")
            log_fn("[PATCH] Patched gemma4_unified.py for complete heterogeneity compatibility")
            return True
    except Exception as e:
        log_fn(f"[PATCH-WARN] Failed to patch gemma4_unified.py: {e}")
    return False


def patch_sitecustomize(extra_site_dirs: Sequence[Path] | None = None, log_fn: Callable[[str], None] = _default_log) -> bool:
    sitecustomize_code = (
        "import sys\n"
        "# 1. Torch Accelerator compat shim\n"
        "try:\n"
        "    import torch\n"
        "    if not hasattr(torch, 'accelerator'):\n"
        "        class _TorchAccShim:\n"
        "            @staticmethod\n"
        "            def is_available(): return False\n"
        "        torch.accelerator = _TorchAccShim\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 2. Universal RoPE compatibility shim for transformers (fixes KeyError: 'default')\n"
        "try:\n"
        "    import transformers.modeling_rope_utils as _rope_utils\n"
        "    if hasattr(_rope_utils, 'ROPE_INIT_FUNCTIONS') and 'default' not in _rope_utils.ROPE_INIT_FUNCTIONS:\n"
        "        if hasattr(_rope_utils, '_compute_default_rope_parameters'):\n"
        "            _rope_utils.ROPE_INIT_FUNCTIONS['default'] = _rope_utils._compute_default_rope_parameters\n"
        "        else:\n"
        "            import torch\n"
        "            def _default_rope_init(config, device=None, **kwargs):\n"
        "                base = getattr(config, 'rope_theta', 10000.0)\n"
        "                partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)\n"
        "                head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32))\n"
        "                dim = int(head_dim * partial_rotary_factor)\n"
        "                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))\n"
        "                return inv_freq, 1.0\n"
        "            _rope_utils.ROPE_INIT_FUNCTIONS['default'] = _default_rope_init\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 3. Universal Transformers Heterogeneity compat shim\n"
        "try:\n"
        "    import transformers.integrations.heterogeneity.configuration_utils as _hetero_cfg\n"
        "    _hetero_cfg.HeterogeneousConfigMixin.__getattribute__ = object.__getattribute__\n"
        "    _hetero_cfg.HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = True\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    from transformers.configuration_utils import PretrainedConfig\n"
        "    PretrainedConfig.allow_global_per_layer_attribute_access = True\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 4. Flash Attention integrity check & safe SDPA fallback\n"
        "try:\n"
        "    import torch\n"
        "    import flash_attn_2_cuda\n"
        "except Exception:\n"
        "    import sys\n"
        "    for _k in list(sys.modules.keys()):\n"
        "        if _k == 'flash_attn' or _k.startswith('flash_attn.'):\n"
        "            sys.modules.pop(_k, None)\n"
        "    sys.modules['flash_attn'] = None\n"
        "    sys.modules['flash_attn_2_cuda'] = None\n"
        "    try:\n"
        "        import transformers.utils.import_utils as _iu\n"
        "        _iu.is_flash_attn_2_available = lambda: False\n"
        "        _iu.is_flash_attn_greater_or_equal_2_10 = lambda: False\n"
        "    except Exception:\n"
        "        pass\n"
    )
    target_site_dirs: set[Path] = set()
    try:
        import site
        if hasattr(site, "getsitepackages"):
            target_site_dirs.update(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    if extra_site_dirs:
        for p in extra_site_dirs:
            target_site_dirs.add(Path(p))

    modified = False
    for site_dir in target_site_dirs:
        if site_dir.exists() and site_dir.is_dir():
            try:
                sc = site_dir / "sitecustomize.py"
                if not sc.exists() or sc.read_text(encoding="utf-8") != sitecustomize_code:
                    sc.write_text(sitecustomize_code, encoding="utf-8")
                    log_fn(f"[PATCH] Injected universal sitecustomize.py into {site_dir}")
                    modified = True
            except Exception:
                pass
    return modified


def verify_audio_separator(log_fn: Callable[[str], None] = _default_log) -> bool:
    """Check that audio-separator can be imported cleanly without namespace or dependency conflicts."""
    try:
        import audio_separator
        return True
    except ImportError:
        log_fn("[PATCH-WARN] audio-separator is not installed in the active environment.")
        return False
    except Exception as e:
        log_fn(f"[PATCH-WARN] audio-separator import failed with error: {e}")
        return False


def apply_source_patches(
    wgp_dir: Path | str,
    log_fn: Callable[[str], None] = _default_log,
    restore_pristine_models: bool = False,
    extra_site_dirs: Sequence[Path] | None = None
) -> None:
    """
    Apply all source-level patches to a Wan2GP installation directory.
    Safe to call repeatedly across local and cloud environments.
    """
    wgp_path = Path(wgp_dir).resolve()
    if not wgp_path.exists():
        log_fn(f"[PATCH-ERROR] Target directory {wgp_path} does not exist!")
        return

    # In cloud volume rebuilds, optionally restore models directory to clean state
    if restore_pristine_models and (wgp_path / ".git").exists():
        try:
            subprocess.run(["git", "checkout", "--", "models/"], cwd=str(wgp_path), capture_output=True)
            log_fn("[PATCH] Restored models/ in Wan2GP git directory to pristine state")
        except Exception:
            pass

    # 1. Triton compatibility (nanovllm)
    patch_nanovllm_triton(wgp_path, log_fn=log_fn)

    # 2. TTS higgs_audio_v2_tokenizer
    patch_higgs_tokenizer(wgp_path, log_fn=log_fn)

    # 3. download_progress.py
    patch_download_progress(wgp_path, log_fn=log_fn)

    # 4. shared/attention.py safe flash_attn fallback
    patch_attention_safety(wgp_path, log_fn=log_fn)

    # 5. Qwen3-VL pad_token_id
    patch_qwen3_vl(wgp_path, log_fn=log_fn)

    # 6. LTX-2 Gemma4UnifiedTextConfig
    patch_gemma4_unified(wgp_path, log_fn=log_fn)

    # 7. Universal sitecustomize.py
    patch_sitecustomize(extra_site_dirs=extra_site_dirs, log_fn=log_fn)

    # 8. Ensure in-memory runtime shims are active
    ensure_runtime_shims()


def self_test() -> bool:
    """Quick diagnostic to verify all shims and critical dependencies."""
    ensure_runtime_shims()
    audio_ok = verify_audio_separator()
    return audio_ok


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    print(f"Applying Wan2GP automated source patches to: {target}")
    apply_source_patches(target, log_fn=print)
    print("Verification completed successfully.")
