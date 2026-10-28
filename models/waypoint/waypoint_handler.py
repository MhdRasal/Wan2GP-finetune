# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch

from .texts import WAYPOINT_DEEPY_INFOS, WAYPOINT_DEEPY_PROMPT_INFOS, WAYPOINT_INFOS, WAYPOINT_PROMPT_INFOS

ARCHITECTURES = {"waypoint_1_5": (512, 1024), "waypoint_1_5_360p": (256, 512)}  # (height, width) each was trained at


def _resolution(base_model_type):
    height, width = ARCHITECTURES[base_model_type]
    return f"{width}x{height}"


class family_handler:
    @staticmethod
    def query_supported_types():
        return list(ARCHITECTURES)

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "waypoint"

    @staticmethod
    def query_family_infos():
        return {"waypoint": (54, "Waypoint")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        resolution = _resolution(base_model_type)
        return {
            "device_explicit": True,
            "world_outputs": True,  # explorable worlds (.wgpworld in the Worlds gallery) rendered as videos
            "cuda_graphs": ["transformer"],  # every pass once the local windows are full (the first second of a world runs eagerly)
            "dtype": "bf16",
            "accelerated": "native",
            "size": "lighter" if base_model_type == "waypoint_1_5_360p" else "large",
            "specialities": [{"name": "real-time game worlds", "aliases": ["video game", "interactive", "real time"],
                              "description": "playable 60 fps game worlds steered with the keyboard, generated faster than playback"}],
            "fps": 60,
            "vae_block_size": 32,  # 16 pixels per latent, 2x2 latents per token
            "resolutions": [[f"{resolution} (2:1)", resolution]],
            "lock_resolutions": True,  # the size it was trained at (start images center-cropped to it)
            "frames_minimum": 4,
            "frames_steps": 4,
            "frames_selection_maximum": 60 * 120,
            "sliding_window": False,
            "image_prompt_types_allowed": "TS",  # a new world starts from an image; a continued world (custom guide) needs none
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            # the prompt only lists moves: Waypoint reads no text (its length overrides Video Length; without moves the world stands still)
            "prompt_description": "Moves, one per line, e.g. [Forward 2s], [Forward+Jump], [Look_Left 1s] (no description: Waypoint reads no text; without moves the world stands still for the length below)",
            "inference_steps": False,  # four fixed steps per latent frame
            "tiny_vae_architecture": "hunyuan_1_5_t2v",  # TAEHV 1.5 is its own VAE
            "custom_guide": {"id": "custom_guide", "name": "World", "label": "Continue a World (.wgpworld of the Worlds gallery)", "type": "file", "default": None,
                             "required": False, "file_types": [".wgpworld"]},
            "profiles_dir": [base_model_type],
            "infos": WAYPOINT_INFOS,
            "prompt_infos": WAYPOINT_PROMPT_INFOS,
            "deepy_infos": WAYPOINT_DEEPY_INFOS,
            "deepy_prompt_infos": WAYPOINT_DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_world_explorer(base_model_type, manifest, memory):  # fixed windows, whatever the profile
        from .exploration import explorer_menus
        return explorer_menus(manifest)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": "DeepBeepMeep/HunyuanVideo1.5", "sourceFolderList": ["preview_decoders"], "fileList": [["taehv1_5.safetensors"]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False, text_encoder_quantization=None, dtype=torch.bfloat16,
                   VAE_dtype=torch.float32, mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None, text_encoder_filename=None, profile=-1, **kwargs):
        from .waypoint_main import model_factory

        height, width = ARCHITECTURES[base_model_type]
        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized, height=height, width=width)
        # every latent frame runs the DiT five times and the small VAE once: both stay in VRAM together. Streamed (profiles 2 / 4), the 24
        # blocks cross PCIe five times per latent frame: 0.48 s per frame against ~0.06 s resident (RTX 4090, BF16), so the DiT (1.3 GB
        # INT8, 2.6 GB BF16) is loaded whole whatever the profile (budget 0)
        return model, {"pipe": {"transformer": model.transformer, "vae": model.vae}, "coTenantsMap": {"transformer": ["vae"], "vae": ["transformer"]},
                       "budgets": {"transformer": 0}}

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors("hunyuan", sub_family="hunyuan1.5")  # TAEHV 1.5 latents: HunyuanVideo 1.5's latent space

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if inputs["custom_guide"] is None and "S" not in inputs["image_prompt_type"]:
            return "A new Waypoint world starts from a Start Image (Start with Image); without one, choose a world of the Worlds gallery to continue"
        from shared.utils.world_instructions import split_instructions
        from .controls import EXPLORER_CONTROLS
        try:
            split_instructions(inputs["prompt"], EXPLORER_CONTROLS)
        except ValueError as error:
            return str(error)
        return None

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "video_length": 240,
            "resolution": _resolution(base_model_type),
            "image_prompt_type": "S",
            "multi_prompts_gen_type": "FG",  # the whole prompt is one list of moves, whatever its lines
        })
