# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The Worlds explorer's controls and menus for a Waypoint world (handler query_world_explorer): a player's keyboard and mouse without
text events (EXPLORER_CONTROLS), short moves (the model runs faster than playback on recent GPUs, so a move's clip is ready about as fast
as the previous one plays), no Quality choice (four fixed steps) and no Memory choice (the model's fixed windows: the last 1 s in detail,
the last 8.5 s sparsely), in latent frames (the explorer's unit)."""
from .controls import EXPLORER_CONTROLS
from .kv_cache import GLOBAL_FRAMES
from .waypoint_main import FRAMES_PER_LATENT, LATENT_FPS, latent_count

MOVE_SECONDS = (0.5, 1, 2)


def explorer_menus(manifest):
    return {"controls": EXPLORER_CONTROLS,
            "seconds": [(value, f"{value:g} s", latent_count(value)) for value in MOVE_SECONDS],
            "steps": [(4, "Turbo")],
            "memory": [(GLOBAL_FRAMES, f"{GLOBAL_FRAMES / LATENT_FPS:.1f}", "fixed")],
            "memory_title": "Waypoint keeps the last second of the explored video in detail and the last 8.5 seconds sparsely: places out of sight longer "
                            "than that are made anew when you come back",
            "defaults": {"seconds": 1, "steps": 4, "memory": GLOBAL_FRAMES},
            # the world's latents replace the start image; the task gives the model's locked resolution to pass WanGP's check
            "task": {"video_length": latent_count(1) * FRAMES_PER_LATENT, "image_prompt_type": "", "resolution": f"{manifest['width']}x{manifest['height']}"}}
