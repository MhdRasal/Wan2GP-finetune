# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""User and agent help of the Waypoint world models."""

WAYPOINT_INFOS = """
### What it makes
Waypoint-1.5 turns a **start image** (a game screenshot or any scene seen from the player's eyes) into a playable world: video at 60 frames
per second that reacts to walking and looking around, as a game would. It was trained on gameplay from hundreds of games.

### Inputs
- **Start Image**: the first view of the world, center-cropped to the model's 2:1 size (1024x512, or 512x256 for the 360P version).
  Views that look like a game work best: first person, or a character or vehicle seen from behind.
- **Prompt**: only moves and actions, one per line (no description: Waypoint does not read text). `[Forward 2s]`, `[Look_Left]`,
  `[Forward+Sprint 1.5s]`, `[Forward+Jump]`, `[Wait]`. Without moves the world stands still for the chosen length.
- **Continue a World**: a world of the Worlds gallery, continued with the prompt's moves instead of a start image.

### Exploring
Open a world in the **Worlds** gallery to explore it with the keyboard: W A S D walk, the arrows look around, Space jumps, Shift sprints,
C crouches, E interacts and F attacks (what the action does depends on the game the world looks like). Waypoint generates about
as fast as playback on an RTX 4090 at 1024x512 and twice as fast at 512x256, so moves follow each other with little waiting. Shorter
moves (0.5 s) react sooner; longer ones (2 s) keep the motion smoother.

### Limits
- The world remembers about one second of video in detail and 8.5 seconds sparsely: what stays out of sight longer is made anew.
- Text events (`[Event]`) are not supported.
- The 360P version is lighter and faster (for 8 GB GPUs and older cards), with less detail.
- The first second of a new world runs slower while the model fills its memory.
"""

WAYPOINT_PROMPT_INFOS = """
Waypoint reads no text: its prompt is a list of moves, one per line, in the shared world syntax.

- Walk: `[Forward]`, `[Backward]`, `[Left]`, `[Right]` (the W S A D keys).
- Look: `[Look_Left]`, `[Look_Right]`, `[Look_Up]`, `[Look_Down]` (the mouse).
- Act: `[Jump]`, `[Interact]` and `[Attack]` happen once at the start of their move; `[Sprint]` and `[Crouch]` last the whole move.
  Jumping or sprinting while walking shows best: `[Forward+Jump]`, `[Forward+Sprint 2s]`.
- Combine with `+` and give a length in seconds: `[Forward+Look_Left 2s]`. A move without a length lasts 1 second.
- `[Wait]` stands still. `[Event]` and descriptions are not supported.

Example:
```
[Forward 2s]
[Forward+Sprint+Look_Right 1.5s]
[Forward+Jump 1s]
[Look_Left 1s]
[Wait 1s]
```
"""

WAYPOINT_DEEPY_INFOS = """Waypoint-1.5 (model_type waypoint_1_5, 1024x512; waypoint_1_5_360p, 512x256): a real-time game world model. A new world needs
image_start (image_prompt_type "S", center-cropped to 2:1; game-like first or third person views work best); custom_guide (a .wgpworld)
continues a world instead. The prompt holds only moves (no text conditioning). Output 60 fps video; video_length in frames (multiple of 4)
applies when the prompt has no moves. Memory: ~1 s dense, ~8.5 s sparse. Fixed 4 steps, no guidance. Speed (RTX 4090, INT8): about real
time at 1024x512, 2x real time at 512x256. world_output "world" (default, the .wgpworld for the Worlds gallery) or "video"."""

WAYPOINT_DEEPY_PROMPT_INFOS = """Prompt = moves only, one per line (a player's controls): [Forward] [Backward] [Left] [Right] [Look_Left] [Look_Right] [Look_Up]
[Look_Down] [Wait] and the actions [Jump] [Interact] [Attack] (tapped at the move's start) [Sprint] [Crouch] (held), joined with + on a line,
optional length in seconds ([Forward+Sprint 2s], default 1 s); jump / sprint show best while walking. [Event] is rejected, description lines
are ignored."""
