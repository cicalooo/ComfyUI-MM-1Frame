# MiniMax H3 one-frame quality benchmark

Import [`examples/minimax_h3_t1_quality_benchmark.json`](../examples/minimax_h3_t1_quality_benchmark.json) into ComfyUI. The fixed reference, prompt, dimensions, seed, model path, and T=1 VAE are deliberately visible in the graph. The complete matrix and a record template live in [`minimax_h3_t1_ab_manifest.json`](minimax_h3_t1_ab_manifest.json).

This is a controlled A/B workflow, not an automatic quality claim. Queue one candidate at a time, save it with a unique `record_id`, and compare several prompts before accepting a sampler, shift pair, or acceleration setting.

## Baseline

Run this first and keep it unchanged as the comparison anchor:

| Field | Baseline |
| --- | --- |
| DiT | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| VAE | `minimax_h3_t1_image_vae_step1597.safetensors` |
| LoRAs | Turbo `0.75` + ThisIsFine detail `1.0` |
| Sampler / scheduler | `sa_solver` / `simple` |
| Steps / CFG | `8` / `1.0` |
| Sigma shifts | video `12.0`, audio `3.0` |
| TeaCache | disabled |

Select the reference image once after import. Keep the prompt, reference, dimensions, seed `20260817`, and model files fixed for every A/B pair. Set the SaveImage prefix to include the record ID, for example `MM1Frame/benchmark/example-int8-turbo-8-sa_solver-simple-cfg1-shift12-3`.

## Candidate order

Change one axis at a time from the baseline. Use `simple` first; only test another schedule after the candidate sampler runs successfully and its image is comparable.

1. Compare the native samplers: `sa_solver`, `sa_solver_pece`, `dpmpp_2m_cfg_pp`, `res_multistep_cfg_pp`, and `euler_cfg_pp`.
2. Try compatible native schedules such as `normal`, `sgm_uniform`, `karras`, and `beta` after `simple`. Record a scheduler as `not_applicable` if the ComfyUI build or model rejects it.
3. With the Turbo LoRA enabled, compare 8 steps. Then bypass the Turbo LoRA and compare the matched detail-only and no-LoRA variants at 12 and 16 steps.
4. Keep CFG at `1.0` for the first pass. If the baseline is stable, try the small low-CFG sweep `1.25` and `1.5`.
5. Test sigma shifts as pairs: `(12.0, 3.0)`, `(10.0, 2.5)`, and `(14.0, 3.5)`. Never change the audio shift independently; the native video/audio relationship is part of the candidate.
6. Run the int8 and w4a8 DiT checkpoints independently. Re-check the full sampler/step choice for w4a8; do not transfer a winner from one quantization level by assumption. The local w4a8 filename is a candidate, not proof that every ComfyUI build can load it.

For each candidate, fill the `record_template` in the manifest (JSONL is convenient for append-only results). Record all of the following even when a measurement is unavailable: fixed seed, prompt, reference image, dimensions, diffusion model, text encoder, VAE, LoRAs and strengths, sampler, scheduler, steps, CFG, both sigma shifts, TeaCache mode, runtime, peak VRAM, output filename, and quality notes.

Use these quality checks for every prompt set:

- identity/reference adherence;
- fine detail and texture;
- anatomy and geometry;
- color stability;
- repeatability when the same record is queued again.

Each check must be `passed`, `failed`, `skipped`, `unavailable`, or `not_applicable`. Accept a change only when it preserves or improves the result across several prompts and does not regress the runtime/VRAM objective.

## Optional TeaCache comparison

The installed `ComfyUI-Experimental-H3-TeaCache` plugin is outside this extension. If it is installed, insert `Experimental H3 TeaCache` after the H3 model/LoRA path and before sampling:

1. Run the baseline with TeaCache disabled.
2. Compare `balanced` first, then `quality`, keeping the rest of the record identical.
3. Leave TeaCache disabled for runs under six steps; the plugin intentionally does not cache those schedules.
4. Keep the node optional and disabled by default until visual comparisons show acceptable identity, texture, and geometry drift. Record the plugin mode and detected precision for every enabled run.

TeaCache can save DiT compute but is not a VAE optimization. Its cache buffers can also affect VRAM, particularly with block swapping, so measure peak VRAM rather than assuming a gain.

## Separate T=1 VAE investigation

First measure the current `minimax_h3_t1_image_vae_step1597.safetensors` encode and decode time and peak VRAM at the benchmark dimensions. Use the same reference image and sampled latent shape, and record reconstruction error/perceptual artifacts when comparing VAE implementations.

Then inspect the actual ComfyUI loader and installed checkpoints. In the current environment, native VAE loading has an int8-convrot path for the H3 video VAE, but no compatible W4A8 T=1 image-VAE checkpoint is evidenced. Do not invent a W4A8 VAE format, add a quantization kernel, or patch `comfy/` for this experiment. A future VAE candidate is acceptable only when it has a real checkpoint, a native or explicitly compatible loader path, numerical reconstruction comparison, and visual artifact review against the current T=1 VAE.

The current warmed baseline is recorded in [`vae_baseline_20260817.json`](vae_baseline_20260817.json): median encode `0.461 s`, median decode `1.538 s`, peak allocated `5108.2 MiB`, and peak reserved `5224.0 MiB` on an RTX 3090 at 1024x1536. These numbers include the resident VAE and use a deterministic synthetic input; they are a baseline for later comparisons, not a quality result.

## Result statuses

The repository tests validate the manifest and workflow structure only. Full checkpoint inference, visual comparisons, encode/decode timings, and peak-VRAM measurements are GPU-run evidence and must be reported as `unavailable` until actually collected. This keeps a passing test suite from being mistaken for an image-quality result.
