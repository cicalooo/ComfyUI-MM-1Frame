import asyncio

import torch

import nodes as mm_nodes


class FakeVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image)
        return torch.ones([1, 24, 1, image.shape[1] // 16, image.shape[2] // 16])


class FakeClip:
    def __init__(self):
        self.prompt = None
        self.ref_items = None

    def tokenize(self, prompt, minimax_ref_items):
        self.prompt = prompt
        self.ref_items = minimax_ref_items
        return {"tokens": True}

    def encode_from_tokens_scheduled(self, tokens):
        assert tokens == {"tokens": True}
        return [[torch.zeros([1]), {}]]


def test_one_frame_latent_shapes(monkeypatch):
    monkeypatch.setattr(
        mm_nodes.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
    )

    latent = mm_nodes._one_frame_av_latent(1344, 768, batch_size=2)
    video, audio = latent["samples"].unbind()

    assert video.shape == (2, 24, 1, 48, 84)
    assert audio.shape == (2, 32, 2, 2)


def test_one_frame_latent_rejects_unaligned_canvas():
    try:
        mm_nodes._one_frame_av_latent(1000, 768)
    except ValueError as exc:
        assert "divisible by 32" in str(exc)
    else:
        raise AssertionError("unaligned canvas was accepted")


def test_reference_node_builds_native_ref_payload(monkeypatch):
    monkeypatch.setattr(
        mm_nodes.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
    )
    clip = FakeClip()
    vae = FakeVAE()
    image = torch.zeros([1, 64, 96, 3])

    output = mm_nodes.MiniMaxH3ReferenceToImageOneFrame.execute(
        clip=clip,
        vae=vae,
        prompt="Keep <Picture 1> recognizable",
        width=128,
        height=96,
        ref_images={"ref_image_0": image},
    )

    conditioning, latent = output.result
    video, audio = latent["samples"].unbind()

    assert clip.prompt == "Keep <Picture 1> recognizable"
    assert len(clip.ref_items) == 1
    assert vae.images[0].shape == (1, 64, 96, 3)
    assert conditioning[0][1]["minimax_refs"][0]["kind"] == "image"
    assert conditioning[0][1]["minimax_refs"][0]["latent_h"] == 4
    assert conditioning[0][1]["minimax_refs"][0]["latent_w"] == 6
    assert video.shape == (1, 24, 1, 6, 8)
    assert audio.shape == (1, 32, 2, 2)


def test_extension_registers_both_nodes():
    import importlib.util
    import sys
    from pathlib import Path

    package_dir = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_mm_1frame",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    extension = asyncio.run(package.comfy_entrypoint())
    registered = asyncio.run(extension.get_node_list())

    assert [node.define_schema().node_id for node in registered] == [
        "MM1FrameEmptyMiniMaxH3LatentAV",
        "MM1FrameMiniMaxH3ReferenceToImage",
    ]
