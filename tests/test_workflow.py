import json
from pathlib import Path


EXAMPLES = Path(__file__).parents[1] / "examples"
WORKFLOW = EXAMPLES / "minimax_h3_reference_edit_quality.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_consistent_links(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, source_id, source_slot, target_id, target_slot, link_type in links.values():
        source, target = nodes[source_id], nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert source["outputs"][source_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] in {link_type, "*"}


def _by_type(workflow):
    return {node["type"]: node for node in workflow["nodes"]}


def _source_node(workflow, target_type, target_input):
    by_type = _by_type(workflow)
    target = by_type[target_type]
    link_id = next(input_["link"] for input_ in target["inputs"] if input_["name"] == target_input)
    link = next(link for link in workflow["links"] if link[0] == link_id)
    return next(node for node in workflow["nodes"] if node["id"] == link[1])


def test_examples_contain_only_rebuilt_quality_workflow():
    assert sorted(path.name for path in EXAMPLES.glob("*.json")) == [WORKFLOW.name]


def test_workflow_has_consistent_links():
    _assert_consistent_links(_load(WORKFLOW))


def test_workflow_is_native_quality_path_without_cross_family_lora():
    workflow = _load(WORKFLOW)
    by_type = _by_type(workflow)
    assert {
        "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage",
        "MM1FrameMiniMaxH3ReferenceToImage", "BasicGuider", "BasicScheduler",
        "KSamplerSelect", "SamplerCustomAdvanced", "MM1FrameMiniMaxH3StillDecode",
        "SaveImage", "PreviewImage",
    } <= by_type.keys()
    assert "LoraLoaderModelOnly" not in by_type
    assert "ModelAttentionBackend" not in by_type
    assert "VAEDecode" not in by_type
    assert by_type["UNETLoader"]["widgets_values"][0] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert by_type["CLIPLoader"]["widgets_values"][0] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert by_type["VAELoader"]["widgets_values"] == ["minimax_h3_video_vae_fp16.safetensors"]


def test_workflow_uses_quality_profile_and_twenty_native_steps():
    workflow = _load(WORKFLOW)
    by_type = _by_type(workflow)
    prepare = by_type["MM1FrameMiniMaxH3ReferenceToImage"]
    assert prepare["widgets_values"][1:] == ["Quality | native 5 frames", 1344, 768, "match"]
    assert "<Picture 1>" in prepare["widgets_values"][0]
    assert by_type["KSamplerSelect"]["widgets_values"] == ["sa_solver"]
    assert by_type["BasicScheduler"]["widgets_values"] == ["simple", 20, 1.0]
    assert by_type["MM1FrameMiniMaxH3StillDecode"]["widgets_values"] == ["recommended"]


def test_workflow_decodes_sampler_output_with_same_native_video_vae():
    workflow = _load(WORKFLOW)
    assert _source_node(workflow, "MM1FrameMiniMaxH3ReferenceToImage", "vae")["type"] == "VAELoader"
    assert _source_node(workflow, "MM1FrameMiniMaxH3StillDecode", "vae")["type"] == "VAELoader"
    assert _source_node(workflow, "MM1FrameMiniMaxH3StillDecode", "samples")["type"] == "SamplerCustomAdvanced"
    assert _source_node(workflow, "SaveImage", "images")["type"] == "MM1FrameMiniMaxH3StillDecode"
