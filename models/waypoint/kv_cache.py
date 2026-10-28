# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Keys and values of the committed (clean) frames each Waypoint block attends, kept on the GPU in the q8 format of attention_kit (int8 per
token, head and 32 channels with FP16 scales: about half of BF16) or in 16 bits:
- local blocks: the 15 frames before the current one;
- global blocks (every fourth): the frames at multiples of 8 within the 128 before it (16 at most).
Each block keeps its frames as a compact array of slots (attention does not depend on the order of its keys): a frame leaving the window
frees its slot, the last slot moving into it, and a committed frame is written after the others. A commit writes the frame while the
block still attends the one it replaces: 16 slots per block.

A commit pass writes each block's keys / values into a staging buffer at fixed addresses (so that CUDA graphs can record it); end_commit
then quantizes the whole buffer at once and stores each block's part in the next slot of the blocks keeping the frame."""
import torch

from shared.attention_kit import GROUP, q8_prefix, quantize_q8
from .transformer import DEPTH, HEAD_DIM, KV_HEADS, is_global

LOCAL_FRAMES, GLOBAL_FRAMES, DILATION = 15, 128, 8
SLOTS = 16


def in_window(index, frame, current):
    """Whether block index attends the committed frame `frame` while generating frame `current`."""
    if is_global(index):
        return frame % DILATION == 0 and current - frame < GLOBAL_FRAMES
    return current - frame <= LOCAL_FRAMES


def kept(index, frame):
    """Whether block index keeps a committed frame at all."""
    return not is_global(index) or frame % DILATION == 0


class WindowCache:
    def __init__(self, tokens, device, dtype=torch.bfloat16, quantize=True):
        self.tokens, self.quantize = tokens, quantize
        shape = (DEPTH, SLOTS * tokens, KV_HEADS, HEAD_DIM)
        if quantize:
            make = lambda: (torch.empty(shape, dtype=torch.int8, device=device), torch.empty((*shape[:-1], HEAD_DIM // GROUP), dtype=torch.float16, device=device))
        else:
            make = lambda: (torch.empty(shape, dtype=dtype, device=device),)
        self.storage = (make(), make())  # (key tensors, value tensors), each (blocks, slots x tokens, heads, 64)
        self.staging = torch.empty((2, DEPTH, tokens, KV_HEADS, HEAD_DIM), dtype=dtype, device=device)  # keys / values of a commit
        self.frames = [[] for _ in range(DEPTH)]  # per block: the frame held by each slot
        self.committing = None

    def tensors(self):
        return [*self.storage[0], *self.storage[1], self.staging]

    def reset(self):
        """Empty, for another world (the same storage: CUDA graphs that read it stay valid)."""
        self.frames = [[] for _ in range(DEPTH)]
        self.committing = None

    def warm(self):
        """Whether the local windows are full: from then on the passes of the local blocks keep the same shapes (the global blocks
        change every 8 frames until they hold 16)."""
        return all(len(frames) == LOCAL_FRAMES for index, frames in enumerate(self.frames) if not is_global(index))

    def counts(self):
        """Frames held by each block (the slots its next pass attends)."""
        return [len(frames) for frames in self.frames]

    def _rows(self, slot):
        return slice(slot * self.tokens, (slot + 1) * self.tokens)

    def begin(self, current):
        """Before the passes of frame `current`: frees the slots of the frames that left each block's window."""
        for index, frames in enumerate(self.frames):
            for slot in reversed(range(len(frames))):
                if not in_window(index, frames[slot], current):
                    last = len(frames) - 1
                    if slot != last:
                        for tensors in self.storage:
                            for tensor in tensors:
                                tensor[index, self._rows(slot)] = tensor[index, self._rows(last)]
                        frames[slot] = frames[last]
                    frames.pop()

    def start_commit(self, frame):
        """The next pass commits `frame` (clean): its blocks write their keys / values (write), end_commit stores them."""
        self.committing = frame

    def write(self, index, which, tensor, heads):
        """Keys (which 0) or values (1) of the committed frame from block index for the heads `heads` (tensor: (1, tokens, heads, 64),
        normalized and rotated), into the staging buffer."""
        self.staging[which, index, :, heads.start:heads.stop].copy_(tensor[0])

    def end_commit(self):
        """Stores the staged frame in the next slot of the blocks keeping it."""
        frame, self.committing = self.committing, None
        layers = [index for index in range(DEPTH) if kept(index, frame)]
        staged = self.staging.view(2 * DEPTH, *self.staging.shape[2:])
        sources = quantize_q8(staged) if self.quantize else (staged,)
        for which, tensors in enumerate(self.storage):
            for target, source in zip(tensors, sources):
                for index in layers:
                    target[index, self._rows(len(self.frames[index]))] = source[which * DEPTH + index]
        for index in layers:
            self.frames[index].append(frame)

    def prefix(self, index, frames):
        """kv_prefix of block index: the keys / values of its first `frames` slots, None for none."""
        rows = frames * self.tokens
        if not rows:
            return None
        (k, *k_scale), (v, *v_scale) = ([tensor[index:index + 1, :rows] for tensor in tensors] for tensors in self.storage)
        return q8_prefix(k, k_scale[0], v, v_scale[0]) if self.quantize else (k, v)

    def state(self):
        return [list(frames) for frames in self.frames]
