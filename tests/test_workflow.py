import json
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "examples" / "minimax_h3_reference_edit_showcase.json"


def test_showcase_workflow_has_consistent_links():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert source["outputs"][source_slot]["type"] == link_type


def test_showcase_uses_true_one_frame_reference_node():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert "MM1FrameMiniMaxH3ReferenceToImage" in by_type
    assert "MiniMaxH3ReferenceToVideo" not in by_type

    reference_node = by_type["MM1FrameMiniMaxH3ReferenceToImage"]
    assert reference_node["widgets_values"][1:4] == [1024, 1536, "max"]
    assert "<Picture 1>" in reference_node["widgets_values"][0]
    assert any(input_["name"] == "ref_images.ref_image_0" for input_ in reference_node["inputs"])


def test_showcase_uses_expected_sampling_path():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert by_type["KSamplerSelect"]["widgets_values"] == ["sa_solver"]
    assert by_type["BasicScheduler"]["widgets_values"] == ["simple", 8, 1.0]
    assert by_type["VAELoader"]["widgets_values"] == [
        "minimax_h3_t1_image_vae_step1597.safetensors"
    ]
