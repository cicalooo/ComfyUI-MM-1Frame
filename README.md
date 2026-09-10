# ComfyUI-MM-1Frame

Native ComfyUI helpers for using MiniMax H3 as a still-image generator and multi-reference editor. The pack prioritizes quality and predictable VRAM use without replacing ComfyUI's H3 model, conditioning, sampler, or VAE implementations.

## Profiles

| Profile | H3 visual latent | Intended VAE | Use |
| --- | --- | --- | --- |
| **Quality \| native 5 frames** | Native `T=2`, decoded to five frames | `minimax_h3_video_vae_fp16.safetensors` | Recommended |
| **Balanced \| native 5 frames** | Native `T=2`, decoded to five frames | `minimax_h3_video_vae_int8_convrot.safetensors` | Lower VAE storage/VRAM |
| **Low VRAM \| true T=1** | Experimental `T=1` | `minimax_h3_t1_image_vae_step1597_int8_convrot.safetensors` | Fastest and lightest |

Quality and Balanced use ComfyUI core's native five-frame MiniMax timing helper. Low VRAM keeps the original true-one-frame path. The selected VAE must match the chosen profile.

## Nodes

- **MiniMax H3 Reference to Still** builds native `<Picture i>` Ref2VA conditioning and the selected still latent. With no references it also supports text-only generation.
- **Empty MiniMax H3 Still AV Latent** creates the selected profile's standalone audiovisual latent.
- **MiniMax H3 Still Decode** decodes only the visual latent. It returns a selected image, every decoded candidate, and the selected index. `recommended` balances edge detail, clipping, temporal disagreement, and a small middle-frame preference.

The decoder deliberately does not load or decode the audio VAE.

## Recommended native stack

| Component | Recommended file |
| --- | --- |
| Ref2VA diffusion model | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Text/vision encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Quality VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Balanced VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Low-VRAM VAE | `minimax_h3_t1_image_vae_step1597_int8_convrot.safetensors` |

The rebuilt example uses the Ref2VA base model directly with 20 `sa_solver` / `simple` steps and CFG 1. It intentionally does **not** apply an FL2VA Turbo LoRA to the Ref2VA checkpoint. Cross-family acceleration is not the quality default.

## Example workflow

Import `examples/minimax_h3_reference_edit_quality.json` and choose a reference image. It is a clean native graph:

```text
Native loaders -> MiniMax H3 Reference to Still -> native sampler
               -> MiniMax H3 Still Decode -> Save Image
```

The workflow defaults to the FP16 video VAE and Quality profile. For Balanced, change both the profile and VAE to the INT8 ConvRot video VAE. For Low VRAM, change both to the T=1 INT8 ConvRot image VAE.

Reference resizing defaults to `match`, which limits each reference to approximately the output pixel area. `max` preserves up to a 2048-pixel short edge but can materially increase memory use.

## Requirements

- A recent ComfyUI build with native MiniMax H3 support and `comfy_api.latest`.
- Matching H3 diffusion model, Qwen3-VL encoder, and VAE files in the normal ComfyUI model folders.

## Test

```text
python -m pytest custom_nodes/ComfyUI-MM-1Frame/tests
```
