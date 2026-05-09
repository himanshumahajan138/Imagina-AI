# LLM Service — Script Generation

Generates the structured cinematic script (a list of beats with timing, dialogue, and image/video prompts) from a theme + duration + language.

## Files

```
services/llm/
  __init__.py
  service.py           LLMService facade
  prompts.py           SCRIPT_PROMPT and IMAGE_PROMPT templates
  parser.py            row_to_block, validate_script_data, parse_script_scene_content
  backends/
    __init__.py
    openai.py          OpenAI GPT script gen
    gemini.py          Google Gemini script gen
    replicate.py       Replicate-hosted OSS LLMs (DeepSeek V3, Llama 3.x, etc.)
    mlx_local.py       MLX-LM Qwen 2.5 7B Q4 (Apple Silicon)
```

## What's a "script"?

A `Script` is a list of `ScriptBlock`s, one per scene:

```python
@dataclass
class ScriptBlock:
    script:      str   # spoken line in the chosen language, ~scene_duration seconds
    scene:       str   # English image-prompt paragraph for the still
    video_scene: str   # English video-prompt paragraph (camera moves, motion, mood)
    start_time:  str   # "HH:MM:SS,mmm"
    end_time:    str
```

The cinematic pipeline editor displays the script as a `pd.DataFrame`. The user can edit cells directly before kicking off image/video gen.

The LLM is told to emit valid JSON — no prose, no markdown — by the system prompt and the `SCRIPT_PROMPT` template. Parsing uses `json_repair` to tolerate near-miss output.

## `LLMService`

```python
class LLMService:
    def __init__(self, model_id: str | None = None) -> None:
        preferred = model_id or session_preferred("llm")
        self.model_id, self.cfg = pick_model("llm", preferred=preferred)
        self._backend: LLMBackend = self._load_backend()

    def generate_script(self, theme, duration, language, **kwargs) -> pd.DataFrame:
        blocks = self._backend.generate_script(theme=theme, duration=duration,
                                               language=language, **kwargs).blocks
        return pd.json_normalize([asdict(b) for b in blocks])
```

The facade returns a DataFrame because the UI's data-editor expects one. The worker-side route uses `_backend.generate_script` directly (`worker/routes/scripts.py`) so it can return the typed `Script` without the pandas conversion.

## Prompts (`prompts.py`)

`SCRIPT_PROMPT` is the user-facing template with placeholders for `{theme}`, `{duration}`, `{language}`, `{seconds}`. It instructs the model to:

- Emit a JSON array of ~`{seconds}`-long beats covering the full `{duration}`.
- Write `script` lines in `{language}`, scene/video_scene prompts in English.
- Compute timestamps in HH:MM:SS,mmm (last beat ends exactly at `duration`).
- Return strict JSON only, no markdown fences or commentary.

`IMAGE_PROMPT` is used by the image backends to generate the still — it composes the script line + scene + video_scene into a single image-gen prompt. Lives here (rather than in `services/image/`) because the LLM also sometimes uses it for system-prompt context.

## Parser (`parser.py`)

Two parsers:

- **`row_to_block(row: dict) -> ScriptBlock`** — coerces an LLM-emitted dict into the dataclass. Used by every LLM backend after `json_repair.loads`.
- **`parse_script_scene_content(text: str) -> list[dict]`** — parses an SRT-style block-formatted text file the user uploaded via the "Upload Existing Script File" sidebar expander. Format:

  ```
  00:00:00,000 --> 00:00:08,000
  [script]: "Spoken line."
  [scene]: "Image prompt paragraph."
  [video_scene]: "Video direction paragraph."

  00:00:08,000 --> 00:00:16,000
  [script]: "Next line."
  ...
  ```

  Block-by-block regex parse, lenient on whitespace, logs warnings on malformed blocks rather than raising.

- **`validate_script_data(rows, global_duration) -> list[str]`** — runs after the user edits the DataFrame in the UI. Checks:
  - `start_time` / `end_time` are SRT-format and `start < end`.
  - Adjacent rows' timestamps are contiguous (no gaps or overlaps).
  - Required text fields aren't empty.
  - Max `end_time` doesn't exceed the global duration slider.

  Returns a list of human-readable error strings the UI surfaces with `st.error` per line.

## Backends

### `mlx_local.py` — Apple Silicon, Qwen 2.5 7B Q4

```yaml
qwen-2.5-7b:
  tier: local
  backend: mlx_local
  ram_gb: 6
  hf_id: mlx-community/Qwen2.5-7B-Instruct-4bit
```

Lazy-imports `mlx_lm`. Loads via `core.model_manager` under cache key `mlx::{hf_id}`. Cold load ~30 s; subsequent calls ~3-10 s for a typical 6-beat script.

```python
def generate_script(self, theme, duration, language, **kwargs) -> Script:
    seconds = _seconds_for(kwargs.get("model_type", "local"))
    prompt = SCRIPT_PROMPT.format(theme=theme, language=language,
                                  duration=duration, seconds=seconds)
    model, tokenizer = self._model_and_tokenizer()
    messages = [
        {"role": "system", "content": "You output strict JSON only. No prose, no markdown."},
        {"role": "user", "content": prompt},
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=templated, max_tokens=4096, verbose=False)
    structured = json_repair.loads(raw.strip())
    # ... validate then build Script
```

`warmup()` calls `_model_and_tokenizer()` which triggers the cache load. Used by worker startup preload.

**M2 16 GB note**: 7B Q4 weights ≈ 4 GB on disk, ~6 GB resident (KV cache + activations). Don't swap to 14B or larger without expecting OOM during image gen unless you also evict the LLM (which the pipeline already does).

### `replicate.py` — DeepSeek V3 et al.

```yaml
deepseek-v3:
  tier: cloud_oss
  backend: replicate
  replicate_id: deepseek-ai/deepseek-v3
  requires_env: REPLICATE_API_TOKEN
```

Uses `core.replicate_client.run` + `join_text` to handle the token-iterator response shape Replicate uses for LLMs. ~50 lines total.

```python
output = run(self.replicate_id, input={
    "prompt": prompt,
    "max_tokens": 4096,
    "temperature": 0.7,
    "system_prompt": "You output strict JSON only. No prose, no markdown.",
})
raw_text = join_text(output).strip()
```

Any Replicate-hosted model with a similar input shape works — DeepSeek, Llama 3.3 70B, Qwen 2.5 72B. Just add a yaml entry; no code change.

### `openai.py` — GPT

```yaml
gpt-5.4-mini:
  tier: api
  backend: openai
  requires_env: OPENAI_API_KEY
```

Standard `chat.completions.create` call. Module-level `_client` cached after first construction. Public function `openai_script_generator(...)` is preserved for `core/utils.py` back-compat.

```python
response = get_client().chat.completions.create(
    model=model,                   # default "gpt-4o-mini"; overridable from cfg
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
raw_output = response.choices[0].message.content.strip()
```

The model name (cfg `model` field) is independent of the yaml `model_id`. e.g. yaml `gpt-5.4-mini` mapped to actual `model: "gpt-4o-mini"` until you flip the underlying API model.

### `gemini.py` — Gemini

```yaml
gemini-3.1-flash:
  tier: api
  backend: gemini
  requires_env: GOOGLE_GENAI_API_KEY
```

Uses the `google-genai` SDK with `response_mime_type="application/json"` for native JSON-mode output (skips the JSON-repair fallback most of the time). System instruction enforces strict-JSON.

```python
response = get_client().models.generate_content(
    model=self.model,
    contents=prompt,
    config=genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction="You output strict JSON only. No prose, no markdown.",
        temperature=0.7,
    ),
)
```

## How `model_type` ends up in the prompt

Backwards-compat hack: the cinematic pipeline passes a legacy `model_type` kwarg that gets used by `_seconds_for(model_type)` to compute `seconds` for the prompt template:

- `model_type == "gemini"` → 8 (VEO 3 produces 8s clips)
- `model_type == "openai"` → 12 (Sora produces 12s clips)
- otherwise → 10

This affects the `seconds` placeholder in `SCRIPT_PROMPT`, telling the LLM how long each beat should be when read aloud.

A cleaner approach would be to pass the actual `scene_duration` from the picked video model directly. Filed under "future improvements".

## Error mapping

| Failure | What's raised | Caller surface |
| --- | --- | --- |
| `mlx-lm` not installed | `BackendUnavailable("mlx-lm not installed...")` | Worker 502 → client `BackendUnavailable` |
| MLX OOM | `GenerationFailed(f"MLX LLM returned ...")` (after retries) | Worker 502 → client `GenerationFailed` |
| OpenAI 401 / rate limit | re-raised as is | Worker 500; surfaces in body |
| LLM emits non-JSON | `GenerationFailed("... returned invalid script JSON")` | Worker 502 → client `GenerationFailed` |
| Empty or malformed JSON array | `assert` failure inside backend → bubbles as exception | Worker 500 |

## Editing the DataFrame

After `LLMService.generate_script` returns, the cinematic tab puts the DataFrame into `st.session_state.script_df` and exposes it via `st.data_editor`. The user can:

- Edit script lines (text)
- Edit scene / video_scene prompts (text)
- Adjust speaker / speed columns (added with `.assign(speed=, speaker=, custom_image=None)`)
- Upload a custom image per scene (stored as base64 in the `custom_image` column)
- Add or delete rows (rows ≥ length are appended; deleted rows shift later timestamps)

`validate_script_data` runs before image/video gen kicks off. Errors block progression.

## Future improvements

- **Pass `scene_duration` directly** instead of `model_type`. Cleaner and removes the magic-string coupling.
- **Streaming script generation** — yield blocks as they're parsed instead of returning the full list. Useful for long scripts where the user wants to start editing early.
- **Per-scene prompts as separate calls** — currently we ask the LLM for the entire script in one shot. Per-scene calls would let us regenerate one block without re-rolling the rest.
- **Few-shot examples** for non-English languages — quality drops noticeably outside English; a few language-specific examples in the system prompt would help.
- **JSON-schema validation** before `row_to_block` — catches shape errors with better messages than `KeyError`.
- **`model_type` removal** once nothing reads it. Currently a sidebar mirror sets `st.session_state.model_type` for legacy callers.
