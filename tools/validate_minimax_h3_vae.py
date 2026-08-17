"""Validate a MiniMax H3 VAE through ComfyUI's native VAE interface."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

_COMFY_ROOT = Path(__file__).resolve().parents[3]
if (_COMFY_ROOT / "comfy").is_dir() and str(_COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFY_ROOT))

import comfy.model_management
import comfy.sd
import comfy.utils


def _checkpoint_structure(path: Path) -> dict[str, object]:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        keys = list(file.keys())
        markers = [key for key in keys if key.endswith(".comfy_quant")]
        formats: dict[str, int] = {}
        for key in markers:
            marker = json.loads(bytes(file.get_tensor(key).tolist()).decode("utf-8"))
            fmt = str(marker.get("format", "missing"))
            formats[fmt] = formats.get(fmt, 0) + 1
        metadata = file.metadata() or {}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "keys": len(keys),
        "quant_markers": len(markers),
        "quant_formats": formats,
        "has_t1_metadata": metadata.get("h3_t1_format") == "full_decoder_v1",
        "metadata_keys": sorted(metadata),
    }


def _make_input(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, dtype=torch.float32).view(height, 1)
    x = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, width)
    red = x.expand(height, width)
    green = y.expand(height, width)
    blue = (0.25 + 0.5 * x + 0.25 * y).expand(height, width)
    return torch.stack((red, green, blue), dim=-1).unsqueeze(0).contiguous()


def _clear_model(vae) -> None:
    if vae is not None:
        try:
            vae.patcher.unpatch_model(unload_weights=True)
        except Exception:
            pass
    comfy.model_management.cleanup_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(path: Path, image: torch.Tensor) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    structure = _checkpoint_structure(path)
    state, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, device=torch.device("cpu"), return_metadata=True
    )
    vae = None
    try:
        load_started = time.perf_counter()
        vae = comfy.sd.VAE(
            sd=state,
            device=torch.device("cuda"),
            dtype=torch.float16,
            metadata=metadata,
        )
        load_seconds = time.perf_counter() - load_started

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            encode_started = time.perf_counter()
            latent = vae.encode(image)
            encode_seconds = time.perf_counter() - encode_started
            decode_started = time.perf_counter()
            decoded = vae.decode(latent)
            decode_seconds = time.perf_counter() - decode_started

        latent = latent.detach().float().cpu().contiguous()
        decoded = decoded.detach().float().cpu().contiguous()
        result = {
            **structure,
            "native_loader": "comfy.sd.VAE -> MiniMaxH3VideoVAE",
            "load_seconds": round(load_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "latent_shape": list(latent.shape),
            "decoded_shape": list(decoded.shape),
            "latent_finite": bool(torch.isfinite(latent).all()),
            "decoded_finite": bool(torch.isfinite(decoded).all()),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
        }
        return result, latent, decoded
    finally:
        del state
        _clear_model(vae)


def _diff(a: torch.Tensor, b: torch.Tensor, prefix: str) -> dict[str, float]:
    delta = (a - b).abs()
    return {
        f"{prefix}_mae": float(delta.mean()),
        f"{prefix}_max_abs": float(delta.max()),
        f"{prefix}_rmse": float(torch.sqrt(torch.mean((a - b) ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="unquantized T=1 source checkpoint")
    parser.add_argument("candidate", type=Path, help="quantized candidate checkpoint")
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native VAE inference validation")
    reference = args.reference.resolve()
    candidate = args.candidate.resolve()
    if not reference.is_file() or not candidate.is_file():
        raise FileNotFoundError(f"missing validation checkpoint: {reference} or {candidate}")

    image = _make_input(args.height, args.width)
    ref_result, ref_latent, ref_decoded = _run(reference, image)
    candidate_result, candidate_latent, candidate_decoded = _run(candidate, image)

    comparison: dict[str, object] = {
        "input_shape": list(image.shape),
        "latent_same_shape": list(ref_latent.shape) == list(candidate_latent.shape),
        "decoded_same_shape": list(ref_decoded.shape) == list(candidate_decoded.shape),
    }
    if ref_latent.shape == candidate_latent.shape:
        comparison.update(_diff(ref_latent, candidate_latent, "latent"))
    if ref_decoded.shape == candidate_decoded.shape:
        comparison.update(_diff(ref_decoded, candidate_decoded, "decoded"))

    print(json.dumps({"reference": ref_result, "candidate": candidate_result, "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
