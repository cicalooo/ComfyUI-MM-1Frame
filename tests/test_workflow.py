import json
from pathlib import Path


EXAMPLES = Path(__file__).parents[1] / "examples"
WORKFLOW = EXAMPLES / "minimax_h3_reference_edit_showcase.json"
T1_VAE = "minimax_h3_t1_image_vae_step1597.safetensors"


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
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert source["outputs"][source_slot]["type"] == link_type
        assert target["inputs"][target_slot]["type"] in {link_type, "*"}

    for node in nodes.values():
        for output_slot, output in enumerate(node["outputs"]):
            for link_id in output["links"] or []:
                assert links[link_id][1:3] == [node["id"], output_slot]
        for input_slot, input_ in enumerate(node["inputs"]):
            if input_["link"] is not None:
                assert links[input_["link"]][3:5] == [node["id"], input_slot]


def _by_type(workflow):
    return {node["type"]: node for node in workflow["nodes"]}


def _source_node(workflow, target_type, target_input):
    by_type = _by_type(workflow)
    target = by_type[target_type]
    link_id = next(input_["link"] for input_ in target["inputs"] if input_["name"] == target_input)
    link = next(link for link in workflow["links"] if link[0] == link_id)
    return by_type[next(node["type"] for node in workflow["nodes"] if node["id"] == link[1])]


def test_examples_contain_only_the_final_workflow():
    assert sorted(path.name for path in EXAMPLES.glob("*.json")) == [WORKFLOW.name]


def test_final_workflow_has_consistent_links():
    _assert_consistent_links(_load(WORKFLOW))


def test_final_workflow_contains_original_method_nodes_and_filenames():
    workflow = _load(WORKFLOW)
    by_type = _by_type(workflow)

    assert {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MM1FrameMiniMaxH3ReferenceToImage",
        "LoraLoaderModelOnly",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "SaveImage",
    } <= by_type.keys()
    assert "MM1FramePatch500KDecoder" not in by_type
    assert "MM1FramePickMiddleFrame" not in by_type
    assert "MiniMaxH3ReferenceToVideo" not in by_type
    assert "VAEDecodeTiled" not in by_type

    assert by_type["UNETLoader"]["widgets_values"][0] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert by_type["CLIPLoader"]["widgets_values"][0] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert by_type["VAELoader"]["widgets_values"] == [T1_VAE]
    lora_files = [
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "LoraLoaderModelOnly"
    ]
    assert any("lightx2v-turbo8" in file for file in lora_files)


def test_final_workflow_preserves_reference_prompt_and_sampling_path():
    workflow = _load(WORKFLOW)
    by_type = _by_type(workflow)

    reference_node = by_type["MM1FrameMiniMaxH3ReferenceToImage"]
    assert reference_node["widgets_values"][1:4] == [1024, 1536, "max"]
    assert "<Picture 1>" in reference_node["widgets_values"][0]
    assert any(input_["name"] == "ref_images.ref_image_0" for input_ in reference_node["inputs"])
    assert by_type["KSamplerSelect"]["widgets_values"] == ["sa_solver"]
    assert by_type["BasicScheduler"]["widgets_values"] == ["simple", 8, 1.0]


def test_final_workflow_decodes_directly_with_original_t1_vae():
    workflow = _load(WORKFLOW)
    by_type = _by_type(workflow)

    assert _source_node(workflow, "MM1FrameMiniMaxH3ReferenceToImage", "vae")["type"] == "VAELoader"
    assert _source_node(workflow, "VAEDecode", "vae")["type"] == "VAELoader"
    assert _source_node(workflow, "VAEDecode", "samples")["type"] == "SamplerCustomAdvanced"
    assert by_type["VAELoader"]["outputs"][0]["links"] == [7, 16]
    assert by_type["VAEDecode"]["inputs"][1]["link"] == 16
    assert by_type["VAEDecode"]["inputs"][0]["link"] == 15
    assert by_type["SaveImage"]["inputs"][0]["link"] == 17
