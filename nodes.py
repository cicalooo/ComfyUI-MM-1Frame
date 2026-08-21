"""True one-frame MiniMax H3 nodes."""

import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import io


CANVAS_MULTIPLE = 32
MAX_RESOLUTION = 16384
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

def _one_frame_av_latent(width, height, batch_size=1):
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")

    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [batch_size, 24, 1, height // 16, width // 16],
        device=device,
    )
    audio_t = round(AUDIO_LATENT_FPS / FPS)
    audio = torch.zeros([batch_size, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


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

    target_width = max(
        CANVAS_MULTIPLE,
        round(image_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    target_height = max(
        CANVAS_MULTIPLE,
        round(image_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return target_width, target_height


class EmptyMiniMaxH3OneFrameLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MM1FrameEmptyMiniMaxH3LatentAV",
            display_name="Empty MiniMax H3 One-Frame AV Latent",
            category="model/latent/minimax",
            description="Creates the true T=1 audiovisual latent required by the MiniMax H3 image VAE.",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, batch_size) -> io.NodeOutput:
        return io.NodeOutput(_one_frame_av_latent(width, height, batch_size))


class MiniMaxH3ReferenceToImageOneFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MM1FrameMiniMaxH3ReferenceToImage",
            display_name="MiniMax H3 Reference to Image (1 Frame)",
            category="model/conditioning/minimax",
            description="Builds MiniMax H3 reference-image conditioning and a true one-frame AV latent. Leave references empty for text-to-image.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="'match' limits each reference to the output pixel area; 'max' keeps up to a 2048px short edge for stronger identity fidelity at higher cost.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image; use <Picture i> in the prompt."),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        prompt,
        width,
        height,
        ref_image_size="match",
        ref_images=None,
    ) -> io.NodeOutput:
        latent = _one_frame_av_latent(width, height)
        ref_items = []
        ref_blocks = []

        for image in (ref_images or {}).values():
            if image is None:
                continue
            target_width, target_height = _reference_size(
                image, width, height, ref_image_size
            )
            resized = _resize(image[:1], target_width, target_height)
            encoded = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": target_height // 16,
                    "latent_w": target_width // 16,
                    "latent": encoded,
                }
            )

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": ref_blocks}
            )
        return io.NodeOutput(conditioning, latent)


NODES = [
    EmptyMiniMaxH3OneFrameLatent,
    MiniMaxH3ReferenceToImageOneFrame,
]

__all__ = [
    "NODES",
    "EmptyMiniMaxH3OneFrameLatent",
    "MiniMaxH3ReferenceToImageOneFrame",
]
