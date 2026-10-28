# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Waypoint's controller input of a latent frame (259 values): mouse dx, dy, the 256 button states (Windows virtual-key codes, as the
released pipeline encodes them) and the scroll sign, from the key names of the explorer controls it uses (EXPLORER_CONTROLS, from
shared.utils.world_instructions.CONTROLS): w / a / s / d press those keys, the arrows move the mouse (looking around), and the player's
actions press their usual keys (jump, interact and attack only at the start of their move: TAP_KEYS)."""
import torch

EXPLORER_CONTROLS = ("walk", "look", "jump", "sprint", "crouch", "interact", "attack")  # a player's keyboard and mouse: no text events
TAP_KEYS = ("jump", "interact", "attack")  # pressed once at the start of their move, however long the key is held
CONTROLS = 2 + 256 + 1
# virtual-key codes: the generic and left Shift / Ctrl codes are both down while a keyboard's left key is held (a key state poll reports both)
BUTTONS = {"w": (87,), "a": (65,), "s": (83,), "d": (68,), "jump": (32,), "sprint": (16, 160), "crouch": (17, 162), "interact": (69,), "attack": (1,)}
# mouse displacement per latent frame (raw counts, as recorded) while an arrow is held, measured on generated worlds (specs/WAYPOINT_*):
# horizontally 300 turns slowly, 450 about 60 degrees per second, 600 and more break the scene; vertically 200 tilts about 45 degrees in
# 2 s, 300 overshoots; below 100 the view barely moves
LOOK = {"left": (-400.0, 0.0), "right": (400.0, 0.0), "up": (0.0, -150.0), "down": (0.0, 150.0)}
TAP_LATENTS = 2  # latent frames (2/15 s) a tapped key stays down


def controls_for(keys, latent=0):
    """(259,) FP32 controls of the latent frame `latent` of a move with the explorer keys held."""
    controls = torch.zeros(CONTROLS)
    for key in keys:
        if key in LOOK:
            controls[0] += LOOK[key][0]
            controls[1] += LOOK[key][1]
        elif key not in TAP_KEYS or latent < TAP_LATENTS:
            for code in BUTTONS[key]:
                controls[2 + code] = 1.0
    return controls
