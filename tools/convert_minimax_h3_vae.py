"""Convert the native MiniMax H3 T=1 VAE decoder Linear layers.

This deliberately uses ComfyUI's installed quantization layouts and the layer
set evidenced by the local native H3 VAE int8 checkpoint. The source file is
never modified. The converter is intentionally separate from ComfyUI runtime
code so a failed or experimental conversion cannot change model loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Running a tool by absolute path does not put the ComfyUI root on sys.path.
# Discover the local checkout so the converter always imports the same native
# quantization layouts that the installed ComfyUI loader will use.
_COMFY_ROOT = Path(__file__).resolve().parents[3]
if (_COMFY_ROOT / "comfy").is_dir() and str(_COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMFY_ROOT))

from comfy.quant_ops import AsymW4A8Int8Layout, TensorWiseINT8Layout


QUANT_MARKER_SUFFIX = ".comfy_quant"
DEFAULT_REFERENCE = Path(
    r"C:\diffusioncomfy\models\vae\minimax_h3_video_vae_int8_convrot.safetensors"
)
W4A8_GROUP_SIZE = 16
CONVROT_GROUP_SIZE = 256


def _marker_config(fmt: str) -> dict[str, object]:
    if fmt == "int8":
        return {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": CONVROT_GROUP_SIZE,
        }
    if fmt == "w4a8":
        return {
            "format": "asym_w4a8_int8",
            "group_size": W4A8_GROUP_SIZE,
            "convrot": True,
            "convrot_groupsize": CONVROT_GROUP_SIZE,
        }
    raise ValueError(f"unsupported format: {fmt}")


def _marker_tensor(config: dict[str, object]) -> torch.Tensor:
    encoded = json.dumps(config).encode("utf-8")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _reference_modules(reference: Path) -> list[str]:
    if not reference.is_file():
        raise FileNotFoundError(f"native H3 VAE int8 reference not found: {reference}")

    with safe_open(str(reference), framework="pt", device="cpu") as file:
        modules: list[str] = []
        for key in file.keys():
            if not key.endswith(QUANT_MARKER_SUFFIX):
                continue
            config = json.loads(bytes(file.get_tensor(key).tolist()).decode("utf-8"))
            if config.get("format") != "int8_tensorwise":
                raise ValueError(f"reference marker {key} is not int8_tensorwise: {config}")
            modules.append(key[: -len(QUANT_MARKER_SUFFIX)])

    if not modules:
        raise ValueError(f"reference has no native quantization markers: {reference}")
    return sorted(modules)


def _target_keys(source: Path, reference: Path) -> list[str]:
    modules = _reference_modules(reference)
    with safe_open(str(source), framework="pt", device="cpu") as file:
        source_keys = set(file.keys())

    missing = [f"{module}.weight" for module in modules if f"{module}.weight" not in source_keys]
    if missing:
        raise ValueError(
            "source does not contain every decoder Linear weight from the native H3 "
            f"reference; missing {missing[:5]}" + (" ..." if len(missing) > 5 else "")
        )
    return [f"{module}.weight" for module in modules]


def _validate_shapes(
    source: Path,
    target_keys: list[str],
    fmt: str,
) -> list[tuple[str, tuple[int, ...]]]:
    invalid: list[tuple[str, tuple[int, ...]]] = []
    with safe_open(str(source), framework="pt", device="cpu") as file:
        for key in target_keys:
            shape = tuple(file.get_tensor(key).shape)
            if len(shape) != 2:
                invalid.append((key, shape))
                continue
            if fmt == "w4a8":
                k = shape[1]
                if (
                    k % 16 != 0
                    or k % W4A8_GROUP_SIZE != 0
                    or k % CONVROT_GROUP_SIZE != 0
                ):
                    invalid.append((key, shape))
    if invalid:
        details = ", ".join(f"{key}={shape}" for key, shape in invalid[:8])
        suffix = " ..." if len(invalid) > 8 else ""
        raise ValueError(f"{fmt} layout rejects selected VAE matrices: {details}{suffix}")

    return [(key, shape) for key, shape in _source_shapes(source, target_keys)]


def _source_shapes(source: Path, target_keys: list[str]):
    with safe_open(str(source), framework="pt", device="cpu") as file:
        for key in target_keys:
            yield key, tuple(file.get_tensor(key).shape)


def _quantize_weight(weight: torch.Tensor, fmt: str):
    gpu_weight = weight.to(device="cuda")
    if fmt == "int8":
        return TensorWiseINT8Layout.quantize(
            gpu_weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=CONVROT_GROUP_SIZE,
        )
    return AsymW4A8Int8Layout.quantize(
        gpu_weight,
        group_size=W4A8_GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUP_SIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        codebook=True,
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _add_quantized(
    output: dict[str, torch.Tensor],
    key: str,
    qdata: torch.Tensor,
    params,
    fmt: str,
) -> None:
    module = key[: -len(".weight")]
    output[key] = _cpu_tensor(qdata)
    if fmt == "int8":
        output[f"{key}_scale"] = _cpu_tensor(params.scale)
    else:
        output[f"{key}_s_rel"] = _cpu_tensor(params.scale)
        output[f"{key}_s_channel"] = _cpu_tensor(params.s_channel)
        if params.correction is not None:
            output[f"{key}_correction"] = _cpu_tensor(params.correction)
        if params.codebook is not None:
            output[f"{key}_codebook"] = _cpu_tensor(params.codebook)
    output[f"{module}{QUANT_MARKER_SUFFIX}"] = _marker_tensor(_marker_config(fmt))


def convert(source: Path, output: Path, fmt: str, reference: Path, overwrite: bool) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    reference = reference.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if source == output:
        raise ValueError("output must be a separate file; the source is never overwritten")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite explicitly: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the installed native H3 quantization backend")

    target_keys = _target_keys(source, reference)
    shapes = _validate_shapes(source, target_keys, fmt)
    marker = _marker_config(fmt)
    temp = output.with_name(output.name + ".tmp")
    if temp.exists() and not overwrite:
        raise FileExistsError(f"temporary output already exists; remove it or pass --overwrite: {temp}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    target_set = set(target_keys)
    quantized = 0
    started = time.perf_counter()

    try:
        with safe_open(str(source), framework="pt", device="cpu") as source_file:
            metadata = dict(source_file.metadata() or {})
            for key in source_file.keys():
                if key not in target_set:
                    tensors[key] = _cpu_tensor(source_file.get_tensor(key))
                    continue

                weight = source_file.get_tensor(key)
                qdata, params = _quantize_weight(weight, fmt)
                _add_quantized(tensors, key, qdata, params, fmt)
                quantized += 1
                del qdata, params, weight
                torch.cuda.empty_cache()
                print(f"quantized {quantized}/{len(target_keys)} {key}", flush=True)

        if temp.exists():
            if not overwrite:
                raise FileExistsError(f"temporary output already exists: {temp}")
            temp.unlink()
        save_file(tensors, str(temp), metadata=metadata)
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"output already exists: {output}")
            output.unlink()
        os.replace(str(temp), str(output))
    except Exception:
        if temp.exists() and not output.exists():
            print(f"conversion failed; partial temporary file retained for inspection: {temp}")
        raise
    finally:
        tensors.clear()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    return {
        "format": fmt,
        "source": str(source),
        "output": str(output),
        "reference": str(reference),
        "quantized_layers": quantized,
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "marker": marker,
        "layers": len(shapes),
        "elapsed_seconds": round(elapsed, 3),
        "cuda": torch.cuda.get_device_name(0),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("int8", "w4a8"), required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    targets = _target_keys(args.source.resolve(), args.reference.resolve())
    shapes = _validate_shapes(args.source.resolve(), targets, args.format)
    print(
        json.dumps(
            {
                "format": args.format,
                "source": str(args.source.resolve()),
                "reference": str(args.reference.resolve()),
                "target_layers": len(targets),
                "shapes": {key: list(shape) for key, shape in shapes[:4]},
                "cuda_available": torch.cuda.is_available(),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    result = convert(args.source, args.output, args.format, args.reference, args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
