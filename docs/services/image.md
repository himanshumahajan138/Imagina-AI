# Image Service — Cinematic Stills

Generates one cinematic still per scene from a script-line + image-prompt + video-prompt combination. The stills become seed images for per-scene video generation downstream.

## Files

```
services/image/
  __init__.py
  service.py
  backends/
    __init__.py
    coreml_local.py    SDXL-Turbo via diffusers + MPS  (DEFAULT local)
    zimage_local.py    Z-Image-Turbo (Tongyi-MAI) via diffusers + MPS
    replicate.py       FLUX.1-dev / SD 3.5 / SDXL (any Replicate image model)
    gemini.py          Imagen via google-genai
    openai.py          gpt-image-1 via OpenAI Responses API
```

## `ImageService`

Standard facade pattern. The protocol method:

```python
def generate_image(
    self,
    prompt: str,
    out_path: Path,
    dimension: str,
    reference_images: list[Path] | None = None,
    **kwargs: Any,
) -> MediaAsset:
```

Universal kwargs the cinematic pipeline passes:

- `script` — the spoken line (used in `IMAGE_PROMPT` template).
- `video_scene` — the video prompt (also folded in for continuity).
- `reference_text` — optional user-supplied "make it look like X" instruction.
- `old_image` — when regenerating with a custom instruction, the previous image's path is passed so the OpenAI backend can use it as input_image.
- `previous_response_id` — OpenAI-only, for refinement chaining (continuity across scenes).

Other backends silently ignore the kwargs they don't use.

## How the prompt is built

`services/llm/prompts.py:IMAGE_PROMPT` is the template. Backends format it with `script`, `scene`, and `video_scene`, then append any `reference_text` instruction:

```python
prompt = IMAGE_PROMPT.format(scene=item["scene"],
                             script=item["script"],
                             video_scene=item["video_scene"])
if reference_text:
    prompt += f"\n** IMPORTANT CUSTOM INSTRUCTIONS ... {reference_text}"
```

The Replicate backend uses this directly. The OpenAI backend wraps the prompt as `input_text` plus optional `input_image` items for the reference images and the previous-response link. The Gemini backend feeds it to `Imagen`'s `prompt` parameter.

## Backends

### `coreml_local.py` — SDXL-Turbo (default local)

```yaml
sdxl-turbo:
  tier: local
  backend: coreml_local
  hf_id: stabilityai/sdxl-turbo
  ram_gb: 8
  num_inference_steps: 4
  guidance_scale: 0.0
  # vae_id: madebyollin/sdxl-vae-fp16-fix    # optional
  supported_dimensions: ["1024x1024"]
```

**Filename note**: the file is called `coreml_local.py` because Apple's compiled-Core-ML pipeline was the original aspiration. Current implementation is `diffusers` + MPS (~2-3× slower than ANE-compiled SDXL would be, but doesn't require a custom model conversion step). Keep the filename for now — yaml refers to it.

#### Quality knobs

The cinematic stills look noticeably better at 4 inference steps than the official 1-step recipe. The yaml defaults are tuned for quality:

| Knob | Default | Effect | Trade-off |
| --- | --- | --- | --- |
| `num_inference_steps` | `4` | More denoising → more detail | ~25 s/image vs ~8 s at 1 step |
| `guidance_scale` | `0.0` | Stays on Turbo's distillation rails (Turbo was trained without CFG) | Turning up to ~1.0 sometimes oversharpens |
| `vae_id` | unset | When set (e.g. `madebyollin/sdxl-vae-fp16-fix`), swaps in a more accurate VAE for fp16 inference | Extra ~325 MB download |

`is_turbo` is autodetected from the model_id (`"turbo" in self.model_id.lower()`) and used to choose defaults when yaml doesn't specify (`4 / 0.0` for Turbo, `25 / 7.5` for full SDXL).

#### Pipeline construction

```python
def _load_pipeline(model_id, vae_id):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    if vae_id:
        pipe.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe.to("mps")
    return pipe
```

Cache key: `f"diffusers::{model_id}{'|vae='+vae_id if vae_id else ''}"`. The VAE id is in the key so swapping it doesn't reuse a stale pipeline.

#### Reference images

SDXL-Turbo via `AutoPipelineForText2Image` is text-to-image only. Reference images would need IP-Adapter weights and a different pipeline class. Currently silently ignored; image still generates from the text prompt only.

### `zimage_local.py` — Z-Image-Turbo (alternative local)

```yaml
z-image-turbo:
  tier: local
  backend: zimage_local
  hf_id: Tongyi-MAI/Z-Image-Turbo
  ram_gb: 12
  num_inference_steps: 9
  cpu_offload: false
  supported_dimensions: ["1024x1024", "1024x1536", "1536x1024"]
```

Tongyi-MAI's 6B S3-DiT model, 8-step distilled. Uses `ZImagePipeline` directly (does **not** route through `AutoPipelineForText2Image` — Z-Image is its own pipeline class).

```python
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
if cpu_offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)
```

Quality is the best of the local options; cost is disk (~33 GB on disk, ~12 GB resident). Listed second under `image:` in yaml so SDXL-Turbo is the auto-pick default — Z-Image is opt-in via the sidebar tier picker.

The repo ships fp32 master weights only; `torch_dtype=torch.bfloat16` casts at load time, so disk is fp32 (~33 GB) but memory is bf16 (~12 GB).

`num_inference_steps=9` produces 8 DiT forward passes (per the Z-Image team's recommended config). `guidance_scale=0.0` is hard-coded — Turbo was trained CFG-free.

### `replicate.py` — FLUX.1-dev and other Replicate image models

```yaml
flux-dev:
  tier: cloud_oss
  backend: replicate
  replicate_id: black-forest-labs/flux-dev
  requires_env: REPLICATE_API_TOKEN
```

Generic Replicate-image backend. Maps our `dimension` ("1536x1024") to FLUX's `aspect_ratio` ("16:9") via `core.config.ASPECT_RATIOS`. Uses `core.replicate_client.run` + `download` for the actual generation.

```python
output = run(self.replicate_id, input={
    "prompt": full_prompt,
    "aspect_ratio": _aspect_ratio_for(dimension),
    "output_format": "png",
    "output_quality": 90,
    "num_outputs": 1,
})
url = first_url(output)
download(url, out_path)
```

Drop-in for any Replicate image model that takes (prompt, aspect_ratio, output_format) — FLUX, SD3, SDXL, Playground v2.5, etc. Just add a yaml entry.

### `gemini.py` — Imagen 4

```yaml
imagen-3:
  tier: api
  backend: gemini
  requires_env: GOOGLE_GENAI_API_KEY
```

Uses the `google.genai` SDK's `generate_images` with `Imagen 4`. Translates our dimension to Imagen's `aspect_ratio` field.

```python
response = get_client().models.generate_images(
    model="imagen-4.0-generate-preview-06-06",
    prompt=prompt,
    config=genai_types.GenerateImagesConfig(
        output_mime_type="image/png",
        aspect_ratio=ASPECT_RATIOS[dimension],
        number_of_images=1,
    ),
)
```

3-attempt retry loop because Imagen occasionally returns 0 images for ambiguous prompts. Retries log progress and return None on total failure (the service-layer wraps that as `GenerationFailed`).

### `openai.py` — gpt-image-1 with refinement chaining

```yaml
gpt-image-1:
  tier: api
  backend: openai
  requires_env: OPENAI_API_KEY
```

Uses OpenAI's Responses API with the `image_generation` tool. The interesting bit is **response chaining** for continuity across scenes:

```python
response = get_client().responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": content}],     # text + optional input_images
    tools=[{"type": "image_generation",
            "background": "opaque",
            "quality": "high" if has_refs else "medium",
            "size": dimension}],
    previous_response_id=previous_response_id or None,   # ← continuity
)
```

When the cinematic pipeline runs in "Image Refinement Mode" (`st.session_state.image_refinement_mode = True`), it passes the previous scene's `response_id` so OpenAI can keep the visual style consistent. The backend stores the new `response_id` in `MediaAsset.meta["response_id"]` so the next call has it.

```python
return MediaAsset(
    path=Path(path), kind="image",
    meta={"response_id": response_id},
)
```

Other backends ignore `previous_response_id` and the `meta` field comes back without one — refinement chaining gracefully no-ops for non-OpenAI picks.

#### Reference images

The OpenAI backend supports up to ~3 reference images, sent as `input_image` content blocks alongside the prompt. The backend accepts both file-like (Streamlit `UploadedFile`) and path-string inputs:

```python
for img_file in custom_images_list:
    if hasattr(img_file, "read"):
        raw = img_file.read()
        if hasattr(img_file, "seek"): img_file.seek(0)
    else:
        with open(img_file, "rb") as f: raw = f.read()
    content.append({"type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"})
```

Path-string support matters because the worker has no Streamlit context — `pipelines/cinematic.py:_materialize_reference_images` persists uploaded bytes to `/tmp` first, then passes paths.

## Async at the worker

`POST /images/generate` is **async** (`202 → job_id`). First-run weight downloads (SDXL ~7 GB, Z-Image ~33 GB) take many minutes; sync would either need an absurd timeout or cliff-fail.

`WorkerClient.generate_image` polls every 2 s (tighter than the 5 s default for video, since cached image gen lands in ~10 s and you want responsive UI ticks).

See [worker.md](../worker.md) for the async lifecycle.

## Refinement mode UX

The "Image Refinement Mode" toggle in the sidebar controls whether response-id chaining is active. In code:

```python
# pipelines/cinematic.py:_generate_storyboard_images
refinement_mode = bool(st.session_state.get("image_refinement_mode"))
previous_response_id = None
for i, item in enumerate(scene_script_pairs):
    ...
    kwargs = {"script": ..., "video_scene": ..., ...}
    if refinement_mode:
        kwargs["previous_response_id"] = previous_response_id
    asset = worker.generate_image(prompt=item["scene"], ..., **kwargs)
    if refinement_mode and asset.meta.get("response_id"):
        previous_response_id = asset.meta["response_id"]
```

If the user picks a non-OpenAI backend with refinement mode on, the kwarg is passed and silently ignored. The toggle's label could communicate this better — see future improvements.

## Storyboard regeneration

`ui/components/storyboard_gallery.py` lets the user regenerate a single scene's image with a custom instruction. Under the hood, it calls back into `pipelines.cinematic._generate_storyboard_images` with a single-element list and the `old_image` / `reference_text` kwargs set. Used by the OpenAI backend to feed the previous image as `input_image` so the regen "edits" rather than starts from scratch.

## Memory bookkeeping

Local image backends share the `diffusers::` cache prefix. So `evict_modality("image")` drops both SDXL and Z-Image. They never both fit in 16 GB anyway — the LRU would evict one to make room for the other. The pipeline calls `evict_models("image")` at the end of `generate_audio_images` to free SDXL/Z-Image before video gen needs the slot.

## Failure modes

| Failure | What's raised |
| --- | --- |
| `torch` / `diffusers` not installed | `BackendUnavailable` with `pip install ...` hint |
| `ZImagePipeline` not in your diffusers version | `BackendUnavailable("...PRs #12703 + #12715, in mainline...")` |
| OpenAI 401 / quota | bubbles as exception → 500 Internal Server Error in worker |
| Imagen returns no images after 3 retries | `GenerationFailed("Gemini image generation returned no image")` |
| Replicate URL download fails | `GenerationFailed(...)` from `core.replicate_client` |
| Backend's runtime exception (CUDA/MPS OOM, etc.) | Wrapped as `GenerationFailed(f"... generation failed: {e}")` |

## Future improvements

- **IP-Adapter for SDXL** — would unlock reference-image conditioning on the local backend. ~200 MB extra weights.
- **Refinement-mode UX** — currently the toggle is on regardless of backend; could disable or rename ("OpenAI: keep style consistent") when the picked backend doesn't support it.
- **Multi-image batching** — generate multiple scenes in one call where the backend supports `num_outputs > 1`. Diffusers does; OpenAI Responses doesn't.
- **Quality preset profiles** — bundle (steps, guidance, scheduler) tuples into named presets (`"draft" / "standard" / "high"`) instead of three separate yaml fields.
- **Apple Core ML SDXL** — actual ANE-compiled inference. ~2-3× speedup on M2. Requires per-resolution compiled model packages.
- **Scheduler selection** — DPMSolver++, Euler-A, DEIS, etc. Currently uses the diffusers default for each pipeline class.
- **Cache-key prefix per backend module** instead of shared `diffusers::` — would allow holding both SDXL and Z-Image resident if RAM ever permits (does not on M2 16 GB).
- **NSFW / safety filter hooks** — currently disabled by default; surface as a yaml flag.
