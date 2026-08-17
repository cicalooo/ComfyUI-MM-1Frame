# ComfyUI-MM-1Frame

Native ComfyUI nodes for true one-frame MiniMax H3 generation and reference-image editing. The package reproduces the T=1 behavior from the community patch without modifying `comfy_extras/nodes_minimax_h3.py` or affecting normal H3 video nodes.

## Nodes

- **MiniMax H3 Reference to Image (1 Frame)** — builds native `<Picture i>` reference conditioning plus the true one-frame AV latent. With no references it also works as text-to-image conditioning.
- **Empty MiniMax H3 One-Frame AV Latent** — standalone T=1 latent for hand-built workflows.

Both nodes are under the native `model/.../minimax` categories.

## Requirements

- A recent [ComfyUI](https://github.com/Comfy-Org/ComfyUI) build with native MiniMax H3 support and `comfy_api.latest`
- The model files listed below

No additional Python packages are required.

### Showcase model downloads

Paths are relative to the ComfyUI folder. Keep the listed filenames, or reselect your alternatives in the workflow loader nodes.

| Asset | Download | Approx. size | Install in |
| --- | --- | ---: | --- |
| MiniMax H3 Ref2VA checkpoint | [`minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | 19.5 GiB | `models/diffusion_models/` |
| Qwen3-VL MiniMax text encoder | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 14.6 GiB | `models/text_encoders/` |
| T=1 single-image VAE | [`minimax_h3_t1_image_vae_step1597.safetensors`](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/blob/main/minimax_h3_t1_image_vae_step1597.safetensors) | 4.9 GiB | `models/vae/` |
| 8-step Turbo LoRA (optional) | [`minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors`](https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors) | 348 MiB | `models/loras/h3/` |
| ThisIsFine detail LoRA (optional) | [`MaxiMin-HHH-R2V-ThisIsFine_LoRA_V0_1.safetensors`](https://huggingface.co/Mamad8/MaxiMin-HHH-R2V-ThisIsFine/blob/main/MaxiMin-HHH-R2V-ThisIsFine_LoRA_V0_1.safetensors) | 569 MiB | `models/loras/h3/` |

The showcase has both optional LoRAs enabled. If you omit either one, bypass or remove its loader node and adjust the step count if you remove the Turbo LoRA.

## Usage

1. Restart ComfyUI after installing this folder under `custom_nodes`.
2. Load the MiniMax H3 model, text encoder, and single-image VAE as usual.
3. Add **MiniMax H3 Reference to Image (1 Frame)**.
4. Connect the CLIP and single-image VAE.
5. Add reference images and address them in the prompt as `<Picture 1>`, `<Picture 2>`, and so on.
6. Connect `positive` and `latent` to the normal MiniMax H3 sampling path, then decode with the single-image VAE.

Suggested starting settings from the referenced experiments: 8 steps, CFG 1, `sa_solver` sampler with the `simple` scheduler, and the MiniMax H3 Turbo LoRA where appropriate.

## Showcase workflow

Import [`examples/minimax_h3_reference_edit_showcase.json`](examples/minimax_h3_reference_edit_showcase.json) into ComfyUI for a complete one-reference image-edit graph.

The showcase is preconfigured for the model files currently used by this installation, including the 8-step Turbo and ThisIsFine LoRAs. After importing it:

1. Select an image in **Load Reference Image**.
2. Edit the prompt in **MiniMax H3 Reference to Image (1 Frame)**, keeping `<Picture 1>` where the reference should be addressed.
3. Queue the workflow. Outputs are saved under `output/MM1Frame/`.

If your model filenames differ, select equivalent MiniMax H3 Ref2VA/hybrid, Qwen3-VL MiniMax, T=1 image VAE, and LoRA files in the loader nodes.

## Why the latent is different

The normal native node clamps generation to at least five frames. True image generation needs:

- Video latent: `[B, 24, 1, H/16, W/16]`
- Audio latent: `[B, 32, 2, 2]`

The two audio steps are `round(40 / 24)`, matching H3's 40 Hz audio-latent rate to one video frame at 24 fps.

## Scope

This project supports text and image references for one-frame output. Reference video and audio inputs are intentionally omitted because the native H3 reference-video path requires at least five frames and does not form a coherent T=1 workflow.

The project does not download models, patch ComfyUI, replace the native video nodes, or alter global VAE behavior.

## Test

From the ComfyUI environment:

```text
python -m pytest custom_nodes/ComfyUI-MM-1Frame/tests
```
