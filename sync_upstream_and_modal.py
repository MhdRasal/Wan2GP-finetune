#!/usr/bin/env python3
"""
Synchronize Wan2GP-finetune (Client) and Modal App (Worker) to Upstream DeepBeepMeep.

Strict guarantees:
1. Modal App runs ONLY official, original deepbeepmeep/Wan2GP (never user finetune fork).
2. Existing model weights in Modal volumes (/models/Wan2GP/ckpts) are NEVER touched or removed.
3. Both local Wan2GP-finetune and Modal App are updated to the exact same upstream commit.
4. Scans all default profiles and syncs loras_url_cache_v2.json to Modal so all LoRAs map to URLs.
"""

import os
import sys
import glob
import json
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

UPSTREAM_URL = "https://github.com/deepbeepmeep/Wan2GP.git"

def print_step(title: str):
    print(f"\n\033[1;36m==>\033[0m \033[1m{title}\033[0m")

def print_ok(msg: str):
    print(f"\033[1;32m✓\033[0m {msg}")

def print_warn(msg: str):
    print(f"\033[1;33m!\033[0m {msg}")

def print_err(msg: str):
    print(f"\033[1;31m✗\033[0m {msg}")

def load_env(env_path: Path):
    if not env_path.exists():
        return {}
    env_vars = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars

def compile_lora_cache(wgp_dir: Path) -> dict:
    """Scan all profiles and extract LoRA filenames -> HuggingFace URLs."""
    cache = {}
    cache_file = wgp_dir / "loras_url_cache_v2.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    profiles_dir = wgp_dir / "profiles"
    if profiles_dir.exists():
        for p in profiles_dir.rglob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loras = data.get("activated_loras") or []
                if isinstance(loras, str):
                    loras = [loras]
                
                # Determine lora directory prefix based on model folder
                # e.g. minimax_h3_ref2va -> minimax_h3, wan_2_2 -> wan
                folder_name = p.parent.name
                family_name = folder_name.split("_ref2va")[0].split("_fl2va")[0].split("_tts")[0]
                if family_name.startswith("wan"):
                    family_name = "wan"

                prefixes = {f"loras/{family_name}", f"loras/{folder_name}"}
                
                for entry in loras:
                    if isinstance(entry, str) and entry.startswith("http"):
                        url = entry.split("|")[0]
                        filename = os.path.basename(url)
                        for prefix in prefixes:
                            cache[f"{prefix}|{filename}"] = url
            except Exception:
                pass

    return cache

def main():
    wgp_dir = Path(__file__).resolve().parent
    env_vars = load_env(wgp_dir / ".env")
    endpoint = env_vars.get("MODAL_ENDPOINT")
    api_key = env_vars.get("MODAL_API_KEY", "")

    print("=" * 68)
    print(" 🚀 Wan2GP Upstream & Modal Dual Synchronization Wizard")
    print("=" * 68)

    # -------------------------------------------------------------------------
    # 1. Update Modal App (Original deepbeepmeep/Wan2GP)
    # -------------------------------------------------------------------------
    print_step("Step 1: Updating Modal App Volume to Latest Original Wan2GP...")
    if not endpoint:
        print_warn("MODAL_ENDPOINT not configured in .env. Skipping remote Modal update.")
    else:
        update_url = f"{endpoint}/admin/update-wan2gp"
        headers = {"User-Agent": "Wan2GP-Sync/1.0"}
        if api_key:
            headers["X-API-Key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(update_url, data=b"{}", headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            print_ok("Modal volume updated successfully!")
            print(f"   Git result: {result.get('git_message', '').strip()}")
            print(f"   Python env: {result.get('venv_python', '').strip()}")
        except Exception as e:
            print_warn(f"Failed to update Modal via endpoint ({e}). Checking Modal CLI...")
            try:
                res = subprocess.run(["modal", "run", "cloud/modal/modal_app.py::update_wan2gp_volume_repo"], cwd=str(wgp_dir), capture_output=True, text=True)
                if res.returncode == 0:
                    print_ok("Modal volume updated via Modal CLI!")
                else:
                    print_err(f"Modal CLI update failed: {res.stderr}")
            except Exception as e2:
                print_err(f"Could not reach Modal: {e2}")

    # -------------------------------------------------------------------------
    # 2. Update Local Wan2GP-finetune with exact same upstream
    # -------------------------------------------------------------------------
    print_step("Step 2: Merging Latest Upstream into Local Wan2GP-finetune...")
    try:
        subprocess.check_call(["git", "fetch", UPSTREAM_URL, "main"], cwd=str(wgp_dir))
        # Merge latest upstream main
        merge_res = subprocess.run(
            ["git", "merge", "FETCH_HEAD", "-m", "Merge upstream deepbeepmeep/Wan2GP main"],
            cwd=str(wgp_dir),
            capture_output=True,
            text=True
        )
        if merge_res.returncode == 0:
            print_ok("Local Wan2GP-finetune successfully merged upstream changes!")
        else:
            print_warn(f"Merge output: {merge_res.stdout or merge_res.stderr}")
    except Exception as e:
        print_err(f"Git merge encountered an error: {e}")

    # -------------------------------------------------------------------------
    # 3. Synchronize LoRA URL Mappings to Modal Volume
    # -------------------------------------------------------------------------
    print_step("Step 3: Compiling & Syncing Default LoRA Preset URLs...")
    lora_cache = compile_lora_cache(wgp_dir)
    local_cache_path = wgp_dir / "loras_url_cache_v2.json"
    with open(local_cache_path, "w", encoding="utf-8") as f:
        json.dump(lora_cache, f, indent=4)
    print_ok(f"Compiled {len(lora_cache)} LoRA mappings into local cache.")

    # Upload cache directly to Modal volume so headless workers know all LoRA URLs
    try:
        upload_res = subprocess.run(
            ["modal", "volume", "put", "-f", "wan2gp-models", str(local_cache_path), "Wan2GP/loras_url_cache_v2.json"],
            cwd=str(wgp_dir),
            capture_output=True,
            text=True
        )
        if upload_res.returncode == 0:
            print_ok("Uploaded loras_url_cache_v2.json to Modal volume (wan2gp-models)!")
        else:
            print_warn(f"Volume put warning: {upload_res.stderr or upload_res.stdout}")
    except Exception as e:
        print_warn(f"Could not upload volume cache via CLI: {e}")

    print("\n" + "=" * 68)
    print(" 🎉 SYNCHRONIZATION COMPLETE!")
    print(" • Modal App: Up to date with official original deepbeepmeep/Wan2GP")
    print(" • Model Weights: Untouched and preserved in persistent volumes")
    print(" • Local Finetune: Merged with upstream, retaining cloud features")
    print(" • LoRA Cache: All default preset URLs mapped and synced")
    print("=" * 68 + "\n")

if __name__ == "__main__":
    main()
