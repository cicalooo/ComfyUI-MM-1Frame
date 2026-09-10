"""Native still-image helpers for MiniMax H3."""

import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import _empty_av_latent


CANVAS_MULTIPLE = 32
MAX_RESOLUTION = 16384
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
PROFILES = ["Quality | native 5 frames", "Balanced | native 5 frames", "Low VRAM | true T=1"]
FRAME_SELECTIONS = ["recommended", "middle", "first", "last"]


def _one_frame_av_latent(width, height, batch_size=1):
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, 1, height // 16, width // 16], device=device)
    audio_t = round(AUDIO_LATENT_FPS / FPS)
    audio = torch.zeros([batch_size, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _profile_latent(width, height, profile, batch_size=1):
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    if profile == PROFILES[-1]:
        return _one_frame_av_latent(width, height, batch_size)
    latent, _ = _empty_av_latent(width, height, 5, batch_size)
    return latent


def _resize(image, width, height):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _reference_size(image, width, height, mode):
    image_height, image_width = image.shape[1], image.shape[2]
    if mode == "match":
        scale = min(1.0, math.sqrt((width * height) / (image_width * image_height)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
    target_width = max(CANVAS_MULTIPLE, round(image_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(image_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return target_width, target_height


def _video_samples(samples):
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    return latent


def _flatten_decoded(images):
    if images.ndim == 4:
        return images, images.shape[0], 1
    if images.ndim != 5:
        raise ValueError(f"MiniMax H3 VAE returned unsupported image shape {tuple(images.shape)}")
    batch, frames = images.shape[:2]
    return images.reshape(batch * frames, *images.shape[-3:]), batch, frames


def _recommended_indices(images, batch, frames):
    grouped = images.reshape(batch, frames, *images.shape[-3:])
    gray = grouped[..., :3].mul(grouped.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    horizontal = (gray[..., 1:] - gray[..., :-1]).abs().mean(dim=(-1, -2))
    vertical = (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=(-1, -2))
    clipping = ((gray < 0.01) | (gray > 0.99)).float().mean(dim=(-1, -2))
    temporal = torch.zeros_like(horizontal)
    if frames > 1:
        differences = (grouped[:, 1:] - grouped[:, :-1]).abs().mean(dim=(-1, -2, -3))
        temporal[:, :-1] += differences
        temporal[:, 1:] += differences
        temporal[:, 1:-1] *= 0.5
    center = (frames - 1) / 2
    positions = torch.arange(frames, device=images.device, dtype=images.dtype)
    center_bias = 0.002 * (1.0 - (positions - center).abs() / max(center, 1.0))
    scores = horizontal + vertical - 0.20 * clipping - 0.05 * temporal + center_bias
    return scores.argmax(dim=1)


def _select_frames(images, batch, frames, selection):
    if frames == 1:
        indices = torch.zeros(batch, device=images.device, dtype=torch.long)
    elif selection == "recommended":
        indices = _recommended_indices(images, batch, frames)
    elif selection == "middle":
        indices = torch.full((batch,), frames // 2, device=images.device, dtype=torch.long)
    elif selection == "last":
        indices = torch.full((batch,), frames - 1, device=images.device, dtype=torch.long)
    else:
        indices = torch.zeros(batch, device=images.device, dtype=torch.long)
    offsets = torch.arange(batch, device=images.device) * frames
    return images[offsets + indices], int(indices[0].item())


class EmptyMiniMaxH3OneFrameLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MM1FrameEmptyMiniMaxH3LatentAV", display_name="Empty MiniMax H3 Still AV Latent",
            category="model/latent/minimax",
            description="Creates either the native five-frame H3 packet or the experimental true T=1 still latent.",
            inputs=[
                io.Combo.Input("profile", options=PROFILES, default=PROFILES[0]),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ], outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, profile, width, height, batch_size):
        return io.NodeOutput(_profile_latent(width, height, profile, batch_size))


class MiniMaxH3ReferenceToImageOneFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MM1FrameMiniMaxH3ReferenceToImage", display_name="MiniMax H3 Reference to Still",
            category="model/conditioning/minimax",
            description="Native Ref2VA conditioning with a quality-focused five-frame profile and a true T=1 fallback.",
            inputs=[
                io.Clip.Input("clip"), io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("profile", options=PROFILES, default=PROFILES[0]),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' limits references to the output pixel area; 'max' keeps a 2048px short edge at higher memory cost."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image; use <Picture i> in the prompt."),
                        prefix="ref_image_", min=0, max=9)),
            ], outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, profile, width, height, ref_image_size="match", ref_images=None):
        latent = _profile_latent(width, height, profile)
        ref_items, ref_blocks = [], []
        for image in (ref_images or {}).values():
            if image is None:
                continue
            target_width, target_height = _reference_size(image, width, height, ref_image_size)
            resized = _resize(image[:1], target_width, target_height)
            encoded = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": target_height // 16,
                               "latent_w": target_width // 16, "latent": encoded})
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
        return io.NodeOutput(conditioning, latent)


class MiniMaxH3StillDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MM1FrameMiniMaxH3StillDecode", display_name="MiniMax H3 Still Decode",
            category="model/latent/minimax",
            description="Decodes only the visual H3 latent and returns a selected still plus every candidate.",
            inputs=[io.Vae.Input("vae"), io.Latent.Input("samples"),
                    io.Combo.Input("selection", options=FRAME_SELECTIONS, default="recommended")],
            outputs=[io.Image.Output(display_name="selected image"),
                     io.Image.Output(display_name="all candidates"),
                     io.Int.Output(display_name="selected index")],
        )

    @classmethod
    def execute(cls, vae, samples, selection):
        decoded = vae.decode(_video_samples(samples))
        images, batch, frames = _flatten_decoded(decoded)
        selected, index = _select_frames(images, batch, frames, selection)
        return io.NodeOutput(selected, images, index)


NODES = [EmptyMiniMaxH3OneFrameLatent, MiniMaxH3ReferenceToImageOneFrame, MiniMaxH3StillDecode]

__all__ = ["NODES", "EmptyMiniMaxH3OneFrameLatent", "MiniMaxH3ReferenceToImageOneFrame", "MiniMaxH3StillDecode"]
