import asyncio

import torch

import nodes as mm_nodes


class FakeVAE:
    def __init__(self, decoded=None):
        self.images = []
        self.decoded = decoded

    def encode(self, image):
        self.images.append(image)
        return torch.ones([1, 24, 1, image.shape[1] // 16, image.shape[2] // 16])

    def decode(self, latent):
        assert latent.shape[1] == 24
        return self.decoded


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


def _cpu_intermediate(monkeypatch):
    monkeypatch.setattr(mm_nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))


def test_low_vram_profile_is_true_t1(monkeypatch):
    _cpu_intermediate(monkeypatch)
    latent = mm_nodes._profile_latent(1344, 768, mm_nodes.PROFILES[-1], batch_size=2)
    video, audio = latent["samples"].unbind()
    assert video.shape == (2, 24, 1, 48, 84)
    assert audio.shape == (2, 32, 2, 2)


def test_quality_profile_uses_native_five_frame_helper(monkeypatch):
    called = {}

    def fake_native(width, height, length, batch_size):
        called["args"] = (width, height, length, batch_size)
        return {"samples": "native"}, 5

    monkeypatch.setattr(mm_nodes, "_empty_av_latent", fake_native)
    assert mm_nodes._profile_latent(1344, 768, mm_nodes.PROFILES[0], 3) == {"samples": "native"}
    assert called["args"] == (1344, 768, 5, 3)


def test_profile_latent_rejects_unaligned_canvas():
    try:
        mm_nodes._profile_latent(1000, 768, mm_nodes.PROFILES[0])
    except ValueError as exc:
        assert "divisible by 32" in str(exc)
    else:
        raise AssertionError("unaligned canvas was accepted")


def test_reference_node_builds_native_ref_payload(monkeypatch):
    _cpu_intermediate(monkeypatch)
    clip, vae = FakeClip(), FakeVAE()
    image = torch.zeros([1, 64, 96, 3])
    output = mm_nodes.MiniMaxH3ReferenceToImageOneFrame.execute(
        clip=clip, vae=vae, prompt="Keep <Picture 1> recognizable",
        profile=mm_nodes.PROFILES[-1], width=128, height=96,
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


def test_still_decoder_returns_selected_and_candidate_batch():
    decoded = torch.zeros([1, 5, 8, 8, 3])
    decoded[:, 3, :, 1::2] = 1
    latent = {"samples": torch.zeros([1, 24, 2, 1, 1])}
    selected, candidates, index = mm_nodes.MiniMaxH3StillDecode.execute(
        FakeVAE(decoded), latent, "recommended"
    ).result
    assert selected.shape == (1, 8, 8, 3)
    assert candidates.shape == (5, 8, 8, 3)
    assert index == 3


def test_node_schemas_and_registration():
    latent_schema = mm_nodes.EmptyMiniMaxH3OneFrameLatent.define_schema()
    assert [input_.id for input_ in latent_schema.inputs] == ["profile", "width", "height", "batch_size"]
    ref_schema = mm_nodes.MiniMaxH3ReferenceToImageOneFrame.define_schema()
    assert [input_.id for input_ in ref_schema.inputs[:7]] == [
        "clip", "vae", "prompt", "profile", "width", "height", "ref_image_size"
    ]
    decode_schema = mm_nodes.MiniMaxH3StillDecode.define_schema()
    assert [output.io_type for output in decode_schema.outputs] == ["IMAGE", "IMAGE", "INT"]

    import importlib.util
    import sys
    from pathlib import Path

    package_dir = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_mm_1frame", package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    extension = asyncio.run(package.comfy_entrypoint())
    registered = asyncio.run(extension.get_node_list())
    assert [node.define_schema().node_id for node in registered] == [
        "MM1FrameEmptyMiniMaxH3LatentAV",
        "MM1FrameMiniMaxH3ReferenceToImage",
        "MM1FrameMiniMaxH3StillDecode",
    ]
