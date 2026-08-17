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


def test_middle_frame_picker_returns_upper_middle_and_preserves_tensor_metadata():
    for frame_count, expected_index in ((1, 0), (5, 2), (124, 62)):
        images = torch.arange(frame_count * 2 * 3 * 3, dtype=torch.float16).reshape(frame_count, 2, 3, 3)

        selected = mm_nodes.MM1FramePickMiddleFrame.execute(images).result[0]

        assert selected.shape == (1, 2, 3, 3)
        assert torch.equal(selected, images[expected_index : expected_index + 1])
        assert selected.dtype == images.dtype
        assert selected.device == images.device


def test_middle_frame_picker_accepts_total_range_and_midpoint_sliders():
    images = torch.arange(10 * 2 * 3 * 3, dtype=torch.float32).reshape(10, 2, 3, 3)

    selected = mm_nodes.MM1FramePickMiddleFrame.execute(
        images,
        total_range=8,
        midpoint=5,
    ).result[0]

    assert torch.equal(selected, images[5:6])

    full_sequence = torch.arange(124 * 2 * 3 * 3, dtype=torch.float32).reshape(124, 2, 3, 3)
    selected = mm_nodes.MM1FramePickMiddleFrame.execute(
        full_sequence,
        total_range=124,
        midpoint=62,
    ).result[0]

    assert torch.equal(selected, full_sequence[62:63])


def test_middle_frame_picker_rejects_empty_or_invalid_input():
    invalid_inputs = (torch.empty(0, 2, 3, 3), torch.empty(2, 3, 3), None)

    for images in invalid_inputs:
        try:
            mm_nodes.MM1FramePickMiddleFrame.execute(images)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid IMAGE input was accepted")


def test_middle_frame_picker_rejects_invalid_slider_values():
    images = torch.zeros(5, 2, 3, 3)

    invalid_sliders = (
        {"total_range": 0, "midpoint": 0},
        {"total_range": 6, "midpoint": 2},
        {"total_range": 5, "midpoint": 5},
        {"total_range": 5, "midpoint": -1},
    )

    for sliders in invalid_sliders:
        try:
            mm_nodes.MM1FramePickMiddleFrame.execute(images, **sliders)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid slider values were accepted")


def test_middle_frame_picker_schema_is_image_to_one_image():
    schema = mm_nodes.MM1FramePickMiddleFrame.define_schema()

    assert schema.node_id == "MM1FramePickMiddleFrame"
    assert [input_.id for input_ in schema.inputs] == ["images", "total_range", "midpoint"]
    assert [input_.io_type for input_ in schema.inputs] == ["IMAGE", "INT", "INT"]
    assert schema.inputs[1].default == 124
    assert schema.inputs[2].default == 62
    assert [output.io_type for output in schema.outputs] == ["IMAGE"]
    assert "midpoint slider" in schema.description


def test_extension_registers_three_nodes():
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
        "MM1FramePickMiddleFrame",
    ]
