import json
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "examples" / "minimax_h3_reference_edit_showcase.json"
BENCHMARK_WORKFLOW = Path(__file__).parents[1] / "examples" / "minimax_h3_t1_quality_benchmark.json"
BENCHMARK_MANIFEST = Path(__file__).parents[1] / "benchmarks" / "minimax_h3_t1_ab_manifest.json"
VIDEO_WORKFLOW = Path(__file__).parents[1] / "examples" / "minimax_h3_reference_video_middle_frame.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_consistent_links(workflow):
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


def test_showcase_workflow_has_consistent_links():
    _assert_consistent_links(_load(WORKFLOW))


def test_showcase_uses_true_one_frame_reference_node():
    workflow = _load(WORKFLOW)
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert "MM1FrameMiniMaxH3ReferenceToImage" in by_type
    assert "MiniMaxH3ReferenceToVideo" not in by_type

    reference_node = by_type["MM1FrameMiniMaxH3ReferenceToImage"]
    assert reference_node["widgets_values"][1:4] == [1024, 1536, "max"]
    assert "<Picture 1>" in reference_node["widgets_values"][0]
    assert any(input_["name"] == "ref_images.ref_image_0" for input_ in reference_node["inputs"])


def test_showcase_uses_expected_sampling_path():
    workflow = _load(WORKFLOW)
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert by_type["KSamplerSelect"]["widgets_values"] == ["sa_solver"]
    assert by_type["BasicScheduler"]["widgets_values"] == ["simple", 8, 1.0]
    assert by_type["VAELoader"]["widgets_values"] == [
        "minimax_h3_t1_image_vae_step1597.safetensors"
    ]


def test_video_workflow_uses_native_h3_video_path_and_picks_middle_frame():
    workflow = _load(VIDEO_WORKFLOW)
    _assert_consistent_links(workflow)

    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert "MiniMaxH3ReferenceToVideo" in by_type
    assert "MiniMaxH3SigmaShift" in by_type
    assert "SamplerCustomAdvanced" in by_type
    assert "VAEDecodeTiled" in by_type
    assert "MM1FramePickMiddleFrame" in by_type
    assert "SaveImage" in by_type
    assert "MM1FrameMiniMaxH3ReferenceToImage" not in by_type
    assert "VAEDecode" not in by_type

    reference_node = by_type["MiniMaxH3ReferenceToVideo"]
    assert reference_node["widgets_values"][3] == 124
    assert "<Picture 1>" in reference_node["widgets_values"][0]

    vae_names = {
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "VAELoader"
    }
    assert "minimax_h3_video_vae_fp16.safetensors" in vae_names
    assert "minimax_h3_audio_vae_fp32.safetensors" in vae_names

    picker = by_type["MM1FramePickMiddleFrame"]
    assert picker["inputs"][0]["link"] == 17
    assert [input_["name"] for input_ in picker["inputs"]] == [
        "images",
        "total_range",
        "midpoint",
    ]
    assert picker["widgets_values"] == [124, 62]
    assert picker["outputs"][0]["links"] == [18]
    assert "Range 124" in picker["title"]
    assert "Midpoint 62" in picker["title"]


def test_quality_benchmark_workflow_has_consistent_links_and_native_controls():
    workflow = _load(BENCHMARK_WORKFLOW)
    _assert_consistent_links(workflow)

    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert by_type["MM1FrameMiniMaxH3ReferenceToImage"]["widgets_values"][1:4] == [
        1024,
        1536,
        "max",
    ]
    assert by_type["RandomNoise"]["widgets_values"] == [20260817, "fixed"]
    assert by_type["KSamplerSelect"]["widgets_values"] == ["sa_solver"]
    assert by_type["BasicScheduler"]["widgets_values"] == ["simple", 8, 1.0]
    assert by_type["MiniMaxH3SigmaShift"]["widgets_values"] == [12.0, 3.0]
    assert by_type["CFGGuider"]["widgets_values"] == [1.0]
    assert "ConditioningZeroOut" in by_type
    assert "ExperimentalH3TeaCache" not in by_type


def test_quality_benchmark_manifest_covers_requested_matrix():
    manifest = _load(BENCHMARK_MANIFEST)
    matrix = manifest["matrix"]

    assert Path(__file__).parents[1].joinpath(manifest["workflow"]).exists()
    assert matrix["samplers"] == [
        "sa_solver",
        "sa_solver_pece",
        "dpmpp_2m_cfg_pp",
        "res_multistep_cfg_pp",
        "euler_cfg_pp",
    ]
    assert matrix["schedulers"][0] == "simple"
    assert matrix["steps"] == {"turbo": [8], "non_turbo": [12, 16]}
    assert matrix["cfg"] == [1.0, 1.25, 1.5]
    assert all(pair["audio"] == pair["video"] / 4 for pair in matrix["sigma_shift_pairs"])
    assert {item["label"] for item in matrix["model_precision"]} == {"int8", "w4a8"}
    assert matrix["teacache"]["default"] == "disabled"
    assert matrix["teacache"]["minimum_steps"] == 6
    vae_investigation = manifest["vae_investigation"]
    assert Path(__file__).parents[1].joinpath(vae_investigation["baseline_artifact"]).exists()
    assert vae_investigation["compatible_w4a8_checkpoint"] is None

    required = set(manifest["record_fields"]["required"])
    template = manifest["record_template"]
    statuses = set(manifest["record_fields"]["status_values"])
    assert required <= template.keys()
    assert template["status"] in statuses
    assert all(check["status"] in statuses for check in template["quality"].values())
