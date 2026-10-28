# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The .wgpworld file of a Waypoint world: an uncompressed zip with
- manifest.json: format version, model type, resolution, fps, seed, the world's description and its segments (latent range, video
  frames, keys, instruction);
- tensors.safetensors: the latents (frames, 32, h, w) BF16 (the start image's first, then one per latent frame) and their controls
  (frames, 259) FP32;
- thumbnail.jpg (small, for galleries) and poster.jpg (the last frame at the world's resolution).
The window cache is not stored: opening a world commits its last frames again."""
import io
import json
import zipfile

import torch
from safetensors.torch import load, save

FORMAT, VERSION, EXTENSION = "wgpworld", 1, ".wgpworld"


def build(manifest, session, thumbnail_jpeg, poster_jpeg):
    tensors = {"latents": torch.stack(session.latents).to(torch.bfloat16), "controls": torch.stack(session.controls).float()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps({"format": FORMAT, "version": VERSION, **manifest}, ensure_ascii=False, indent=1))
        archive.writestr("tensors.safetensors", save(tensors))
        archive.writestr("thumbnail.jpg", thumbnail_jpeg)
        archive.writestr("poster.jpg", poster_jpeg)
    return buffer.getvalue()


def read(path):
    """(manifest, tensors) of a world file."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError(f"{path} is not a version {VERSION} world file")
        return manifest, load(archive.read("tensors.safetensors"))
