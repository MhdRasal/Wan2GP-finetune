# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Waypoint-1.5 world generation: from a start image, one latent frame (4 video frames, 1/15 s) at a time, each denoised in four Euler
steps (sigmas 1, 0.9, 0.75, 0.3 rounded to BF16 as released) with the keyboard / mouse input held during it, then committed (a sigma 0
pass storing its keys and values in the window cache) and decoded at once by TAEHV 1.5's streaming decoder.

A world is returned with its .wgpworld file (world_file.py) and stays open as a session; given a world file, generation continues it
(its last frames committed again to rebuild the cache), or serves its interactive exploration (shared.utils.world_exploration) one move at
a time until the viewer leaves."""
import io
import itertools
import queue
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from PIL import Image

from shared.tinyvae.taehv import TAEHV, MemBlock
from shared.utils import files_locator as fl
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status
from shared.utils.world_exploration import EXPLORE_PROMPT, exploration_for
from shared.utils.world_instructions import split_instructions
from . import world_file
from .controls import CONTROLS, EXPLORER_CONTROLS, TAP_LATENTS, controls_for
from .kv_cache import GLOBAL_FRAMES, LOCAL_FRAMES, WindowCache
from .transformer import CHANNELS, WaypointTransformer

VAE_FILE = "preview_decoders/taehv1_5.safetensors"  # TAEHV 1.5 (madebyollin), shared with HunyuanVideo 1.5's previews
SIGMAS = torch.tensor([1.0, 0.9, 0.75, 0.3, 0.0], dtype=torch.bfloat16).tolist()  # the released schedule, as its BF16 sigmas
STEPS = len(SIGMAS) - 1
FRAMES_PER_LATENT, LATENT_FPS = 4, 15
FPS = FRAMES_PER_LATENT * LATENT_FPS
SPATIAL_RATIO, PATCH = 16, 2  # pixels per latent, latents per token side
MOVE_SECONDS = 1.0  # a move without a length
REPLAY_FRAMES = GLOBAL_FRAMES + LOCAL_FRAMES  # latent frames committed again when a world is opened: what its windows can still see


def latent_count(seconds):
    return max(1, round(seconds * LATENT_FPS))


class WaypointVAE(torch.nn.Module):
    """TAEHV 1.5 (encoder and decoder, FP16): the start image to its latent, and the streaming decoder of the generated latents."""

    def __init__(self):
        super().__init__()
        self.net = TAEHV(patch_size=2, latent_channels=CHANNELS, decoder_time_upscale=(False, True, True), encoder=True)
        self._convertWeightsFloatTo = None

    def encode_image(self, image):
        """Latent (32, h, w) of an image (3, H, W) in [-1, 1]: four identical frames, as the released pipeline encodes its seed."""
        frames = ((image.to(self.net.encoder[0].weight) + 1) / 2).clamp_(0, 1)[None, None]
        return self.net.encode_video(frames)[0, 0]

    def decode(self, latent, memory):
        """The 4 frames (3, 4, H, W) in [-1, 1] of a latent (32, h, w), memory: the decoder's state, a list carried along the world's
        latents ([] at its start: the first frame then serves as its own past, the closest match of the released decoder's warm-up)."""
        if not memory:
            memory.extend([None] * len(self.net.decoder))
        x = latent.to(self.net.decoder[1].weight)[None]
        for index, block in enumerate(self.net.decoder):
            if isinstance(block, MemBlock):
                past = torch.cat([x[:1] if memory[index] is None else memory[index], x[:-1]])
                memory[index] = x[-1:].clone()
                x = block(x, past)
            else:
                x = block(x)
        frames = self.net.postprocess_output_frames(x)  # (4, 3, H, W) in [0, 1]
        return frames.mul_(2).sub_(1).transpose(0, 1)


@dataclass
class WorldSession:
    """An open world: its manifest, its latents (the start image's, then one per generated latent frame) and their controls (CPU), the
    window cache, the decoder's state and the last frame shown."""
    manifest: dict
    latents: list  # (32, h, w) BF16 per frame
    controls: list  # (259,) FP32 per frame
    cache: WindowCache
    decoder_memory: list
    moves: int = 0
    last_frame: torch.Tensor | None = None

    @property
    def frames(self):
        return len(self.latents)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False, height=512, width=1024):
        self.device = torch.device("cuda")
        self.model_type = model_type
        model_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            transformer = WaypointTransformer(height // SPATIAL_RATIO // PATCH, width // SPATIAL_RATIO // PATCH)
        transformer.lock_layers_dtypes()
        offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16)  # the sigma embedding is locked to FP32
        for block in transformer.blocks:
            block.lamb = float(block.attn.v_lamb)
        self.transformer = transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename, torch.bfloat16, None)
        with init_empty_weights():
            vae = WaypointVAE()
        offload.load_model_data(vae, fl.locate_file(VAE_FILE), preprocess_sd=lambda state: {"net." + key: value for key, value in vae.net.patch_tgrow_layers(state).items()},
                                writable_tensors=False, default_dtype=torch.float16)
        self.vae = vae.eval().requires_grad_(False)
        self.size = (height, width)
        self.session = None
        self._cache = None  # the window cache, kept from one world to the next (its CUDA graphs stay valid)

    def _window_cache(self):
        if self._cache is None:
            height, width = self.size
            self._cache = WindowCache(height * width // (SPATIAL_RATIO * PATCH) ** 2, self.device)
        self._cache.reset()
        return self._cache

    # ---- frames

    def _generator(self, session, frame):
        return torch.Generator(device=self.device).manual_seed(session.manifest["seed"] * 1_000_003 + frame)

    def _commit(self, session, latent, controls, frame):
        session.cache.begin(frame)
        session.cache.start_commit(frame)
        self.transformer(latent, 0.0, controls, frame, session.cache, commit=True)
        session.cache.end_commit()

    def _frame(self, session, controls, on_step=None):
        """Generates and commits the next latent frame with controls (259) held: returns it (32, h, w), added to the session."""
        frame, (height, width) = session.frames, self.size
        cache = session.cache
        cache.begin(frame)
        latent = torch.randn((CHANNELS, height // SPATIAL_RATIO, width // SPATIAL_RATIO), generator=self._generator(session, frame), device=self.device)
        for step, (sigma, following) in enumerate(zip(SIGMAS[:-1], SIGMAS[1:])):
            velocity = self.transformer(latent, sigma, controls, frame, cache)
            latent.add_(velocity.float(), alpha=following - sigma)
            del velocity
            if on_step is not None:
                on_step(step, latent)
        latent = latent.to(torch.bfloat16)
        self._commit(session, latent, controls, frame)
        session.latents.append(latent.cpu())
        session.controls.append(controls.cpu())
        return latent

    def _decode(self, session, latent):
        """The latent's 4 frames (3, 4, H, W) in [-1, 1], FP16 on the CPU (a long world's video stays out of VRAM)."""
        runtime.ensure_loaded(self.vae)  # the decoder is driven here (persistent state), not through a hooked forward
        frames = self.vae.decode(latent, session.decoder_memory).clamp_(-1, 1).cpu()
        session.last_frame = frames[:, -1]
        return frames

    def _run(self, session, moves, on_step=None):
        """Generates the moves [(keys, latent frames)] at the end of the world: their frames (3, 4 n, H, W) in [-1, 1]."""
        frames, done = [], 0
        for keys, count in moves:
            tapped, held = (controls_for(keys, latent).to(self.device) for latent in (0, TAP_LATENTS))
            for index in range(count):
                step_callback = None if on_step is None else (lambda step, latent, done=done: on_step(done * STEPS + step, latent))
                latent = self._frame(session, tapped if index < TAP_LATENTS else held, step_callback)
                frames.append(self._decode(session, latent))
                done += 1
                check_abort()
        return torch.cat(frames, dim=1)

    # ---- worlds

    def _new_session(self, image, seed, description):
        runtime.ensure_loaded(self.vae)
        height, width = self.size
        latent = self.vae.encode_image(image.to(self.device)).to(torch.bfloat16)
        manifest = {"model_type": self.model_type, "width": width, "height": height, "fps": FPS, "latent_fps": LATENT_FPS, "seed": seed, "world_id": uuid.uuid4().hex,
                    "world": {"description": description, "segments": []}, "segments": []}
        session = WorldSession(manifest, [latent.cpu()], [torch.zeros(CONTROLS)], self._window_cache(), [])
        self._commit(session, latent, session.controls[0].to(self.device), 0)  # the start image primes the cache (not decoded)
        return session

    def close_session(self):
        self.session = None

    @staticmethod
    def _jpeg(frame, size=None):
        image = Image.fromarray(((frame.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        if size is not None:
            image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85 if size else 92)
        return buffer.getvalue()

    def _world_file(self, session):
        return world_file.build(session.manifest, session, self._jpeg(session.last_frame, 320), self._jpeg(session.last_frame))

    @staticmethod
    def _moves(instructions, frame_num):
        moves = [(instruction["keys"], latent_count(instruction["seconds"] or MOVE_SECONDS), instruction["text"]) for instruction in instructions]
        return moves or [([], max(1, frame_num // FRAMES_PER_LATENT), "")]  # no instructions: the requested length, no input

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, frame_num, image_start=None, input_custom=None, callback=None, world_output=None, **kwargs):
        if input_custom is not None and input_prompt.strip() == EXPLORE_PROMPT:
            exploration = exploration_for(input_custom)
            return None if exploration is None else self._explore(exploration, callback)
        self.transformer.reset_conditions()
        description, instructions = split_instructions(input_prompt, EXPLORER_CONTROLS)
        if input_custom is not None:
            session = self.open_world(input_custom)
        else:  # a start image (validate_generative_settings)
            set_phase_status("Encoding the Start Image")
            session = self._new_session(image_start, seed, description)
        self.close_session()
        moves = self._moves(instructions, frame_num)
        total = sum(count for _, count, _ in moves)
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=total * STEPS)

        def on_step(step, latent):  # the preview: the last latent frames, the current one at its denoising state
            if callback is not None:
                recent = torch.stack([*(frame.to(self.device, torch.float32) for frame in session.latents[-7:]), latent]) if step % STEPS == STEPS - 1 else None
                callback(step, None if recent is None else recent.transpose(0, 1), False)

        set_phase_status("Generating World")
        first = session.frames
        video = self._run(session, [(keys, count) for keys, count, _ in moves], on_step)
        for keys, count, text in moves:
            session.manifest["segments"].append({"latent_start": first, "latent_frames": count, "video_frames": count * FRAMES_PER_LATENT, "keys": "+".join(keys),
                                                  **({"instruction": text} if text else {})})
            first += count
        session.manifest["world_id"] = uuid.uuid4().hex
        self.session = session
        if (world_output or "world") == "video":
            return {"x": video}
        return {"x": video, "side_files": {world_file.EXTENSION: self._world_file(session)}}

    @torch.inference_mode()
    def open_world(self, path):
        """Makes the world of a .wgpworld file the open session: its latents and controls; its last frames are committed again (the
        cache), and the decoder's state is warmed on its last latents."""
        manifest, tensors = world_file.read(path)
        session = self.session
        if session is not None and session.manifest["world_id"] == manifest["world_id"] and session.moves == 0:
            return session
        self.close_session()
        latents, controls = list(tensors["latents"].unbind()), list(tensors["controls"].unbind())
        session = WorldSession(manifest, latents, controls, self._window_cache(), [])
        set_phase_status("Preparing World")
        start = max(0, session.frames - REPLAY_FRAMES)
        for frame in range(start, session.frames):
            self._commit(session, latents[frame].to(self.device), controls[frame].to(self.device), frame)
            check_abort()
        for frame in range(max(1, session.frames - 3), session.frames):
            self._decode(session, latents[frame].to(self.device))
        self.session = session
        return session

    # ---- exploration

    @torch.inference_mode()
    def move(self, keys, text="", seconds=MOVE_SECONDS, on_step=None, memory=None):
        """Continues the open world by one move with the explorer's held keys ({"keys": "w+left"}, None for none); text events and the
        memory are not used (no text input, fixed windows). Returns the frames (3, frames, H, W) in [-1, 1]."""
        session = self.session
        keys = [key for key in ((keys or {}).get("keys") or "").split("+") if key]
        first, count = session.frames, latent_count(seconds)
        video = self._run(session, [(keys, count)], on_step)
        session.manifest["segments"].append({"latent_start": first, "latent_frames": count, "video_frames": count * FRAMES_PER_LATENT, "keys": "+".join(keys), "explored": True})
        session.moves += 1
        return video

    def finish_move(self):
        pass

    @torch.inference_mode()
    def explore(self, world_path):
        """Serves the exploration of a world outside the queue (the Worlds gallery took the GPU from a running generation)."""
        exploration = exploration_for(world_path)
        if exploration is not None:
            self._explore(exploration, None)

    def _move_stats(self, seconds, frames, request):
        from shared.utils.stats import live_stats
        session = self.session
        move = latent_count(request["seconds"]) * FRAMES_PER_LATENT

        def amount(video_frames):
            return {"seconds": round(video_frames / FPS, 1), "frames": video_frames, "moves": round(video_frames / move)}

        world = sum(segment["video_frames"] for segment in session.manifest["segments"])
        explored = (session.frames - 1) * FRAMES_PER_LATENT
        memory = (f"Memory   the last {LOCAL_FRAMES / LATENT_FPS:.1f} s in detail and the last {GLOBAL_FRAMES / LATENT_FPS:.1f} s sparsely, of the {explored / FPS:.1f} s explored"
                  f" · {frames / FPS / seconds:.1f}x real time")
        capacity = GLOBAL_FRAMES * FRAMES_PER_LATENT  # video frames the sparse window reaches back
        return {"move_seconds": round(seconds, 2), "video_seconds": round(frames / FPS, 2), "steps": STEPS, "moves": session.moves,
                "peak_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1), **live_stats(), "world": amount(world), "explored_moves": amount(explored - world),
                "explored": amount(explored), "memory": amount(min(explored, capacity)), "memory_full": explored > capacity,
                "room_moves": max(0, capacity - explored) // move, "memory_line": memory}

    def _render(self, session):
        """The whole video explored so far, decoded again from the start (uint8, frames x H x W x 3)."""
        memory, parts = [], []
        for frame in range(1, session.frames):
            runtime.ensure_loaded(self.vae)
            frames = self.vae.decode(session.latents[frame].to(self.device), memory).float().clamp_(-1, 1)
            parts.append(((frames + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())
            check_abort()
        return np.concatenate(parts)

    def _explore(self, exploration, callback):
        """Serves the moves of an exploration until the viewer leaves (or the task is aborted)."""
        try:
            set_phase_status("Preparing World")
            self.open_world(exploration.world_path)
            exploration.publish("ready")
            while True:
                set_phase_status("Exploring World - Waiting for a Move")
                try:
                    request = exploration.next_request(0.25)
                except queue.Empty:
                    check_abort()
                    continue
                if request is None:
                    break
                if "render" in request:
                    set_phase_status("Exploring World - Rendering the Video")
                    exploration.publish("rendered", frames=self._render(self.session), fps=FPS, record=request["render"], moves=self.session.moves)
                    continue
                if "save" in request:
                    set_phase_status("Exploring World - Saving the World")
                    exploration.publish("saved", data=self._world_file(self._explored(self.session)), moves=self.session.moves)
                    continue
                set_phase_status(f"Exploring World - Move {self.session.moves + 1}")
                count = latent_count(request["seconds"])
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=count * STEPS)
                steps = itertools.count()  # no preview while exploring: nobody sees the progress panel
                on_step = None if callback is None else (lambda step, latent: callback(next(steps), None, False))
                torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                video = self.move(request["keys"], request["text"], request["seconds"], on_step)
                frames = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                exploration.publish("clip", frames=frames, fps=FPS, stats=self._move_stats(time.perf_counter() - started, len(frames), request))
                del video
        finally:
            exploration.publish("ended")
            self.close_session()
        return None

    @staticmethod
    def _explored(session):
        """The session as a new world: its segments and moves, a new identity."""
        session.manifest = {**session.manifest, "world_id": uuid.uuid4().hex}
        return session
