# ComfyUI-MM-1Frame

Native ComfyUI nodes for direct one-frame MiniMax H3 reference-image editing. The pack keeps the original distilled-guidance workflow: one true T=1 latent, the T=1 image VAE, and the working 8-step Turbo LoRA path.

## Nodes

- **MiniMax H3 Reference to Image (1 Frame)** — builds native `<Picture i>` reference conditioning and the true one-frame AV latent. With no references, it also supports text-to-image.
- **Empty MiniMax H3 One-Frame AV Latent** — creates a standalone T=1 AV latent for hand-built workflows.

No experimental decoder patcher, middle-frame picker, iterative video workflow, or VAE conversion tool is included.

## Requirements

- A recent [ComfyUI](https://github.com/Comfy-Org/ComfyUI) build with native MiniMax H3 support and `comfy_api.latest`.
- The model files below in the standard ComfyUI folders.

| Asset | Download | Install in |
| --- | --- | --- |
| MiniMax H3 Ref2VA checkpoint | [`minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `models/diffusion_models/` |
| Qwen3-VL MiniMax text encoder | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `models/text_encoders/` |
| T=1 image VAE | [`minimax_h3_t1_image_vae_step1597.safetensors`](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/blob/main/minimax_h3_t1_image_vae_step1597.safetensors) | `models/vae/` |
| 8-step Turbo LoRA | [`minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors`](https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors) | `models/loras/h3/` |
| ThisIsFine detail LoRA (optional) | [`MaxiMin-HHH-R2V-ThisIsFine_LoRA_V0_1.safetensors`](https://huggingface.co/Mamad8/MaxiMin-HHH-R2V-ThisIsFine/blob/main/MaxiMin-HHH-R2V-ThisIsFine_LoRA_V0_1.safetensors) | `models/loras/h3/` |

The working showcase uses the installed `turb\\unknown__fl2v__lightx2v-turbo8__v1.0__rank24-resized__bf16__unpruned.safetensors` Turbo LoRA at strength `0.75`, the optional detail LoRA at `1.0`, `sa_solver`, `simple`, 8 steps, and CFG `1.0`.

## Use

1. Restart ComfyUI after installing this folder under `custom_nodes`.
2. Import [`examples/minimax_h3_reference_edit_showcase.json`](examples/minimax_h3_reference_edit_showcase.json).
3. Select a reference image and keep `<Picture 1>` in the prompt where the reference should be used.
4. Queue the workflow. The sampled one-frame latent is decoded directly with the T=1 image VAE and saved under `output/MM1Frame/`.

If filenames differ, select equivalent Ref2VA, Qwen3-VL MiniMax, T=1 image VAE, and LoRA files in the loader nodes.

## Scope

This extension supports direct one-frame generation/reference editing only. It does not decode multi-frame video, extract a middle frame, replace native H3 nodes, modify ComfyUI core files, or alter global VAE behavior.

## Test

From the ComfyUI environment:

```text
python -m pytest custom_nodes/ComfyUI-MM-1Frame/tests
```
