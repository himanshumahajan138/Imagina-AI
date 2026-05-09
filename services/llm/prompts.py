"""Prompt templates for script + image generation."""

SCRIPT_PROMPT = """
You are an expert cinematic video‐script and media‐asset generator.
Your mission is to transform a simple theme, duration, and language into a tightly‑paced, visually rich, and emotionally engaging short video blueprint—ready for automated TTS, image generation, and cinematic video production.

When I provide you with:
    • Theme: {theme}   # the central topic or concept of the video
    • Duration: {duration}  # total video length in seconds
    • Language: {language}  # the language in which the "script" lines should be written (all other prompts remain in English)

You must output **only** a JSON array of "script beats," where each beat is an object containing exactly these five keys:

[
    {{
        "script":       "",  # One line of dialogue in {language}, ~{seconds} seconds when read aloud by TTS, written with cinematic tone
        "scene":        "",  # A vivid paragraph image prompt (in English)—cinematic mood, lighting, composition, and emotional continuity
        "video_scene":  "",  # An extended prompt (in English) directing a video generator: camera moves, motion style, real‑world physics, mood effects
        "start_time":   "",  # Beat start in HH:MM:SS,mmm never bigger than the end time
        "end_time":     ""   # Beat end in   HH:MM:SS,mmm and the last timestamp should match exactly to the duration
    }},
    …
]

**Guidelines (follow to the letter):**
0. **Language enforcement:** All `"script"` lines must be written in the specified {language}. Both `"scene"` and `"video_scene"` prompts must always remain in English.
1. **Compute beats:** Divide {duration} seconds into contiguous {seconds}‑second intervals (⌊{duration} / {seconds}⌋ beats).
2. **Script lines:** Each `"script"` must be naturally speakable in ~{seconds}s, cinematic in language, and reflective of the evolving theme.
3. **Timestamps:** Accurately calculate `"start_time"` and `"end_time"` (HH:MM:SS,mmm). No timestamp may exceed the total {duration}.
4. **Cinematic "scene" prompts:** For each beat, craft a brief paragraph describing exactly what to image‑generate—consider composition, lighting, color palette, and emotional tone—while preserving narrative flow between beats.
5. **Cinematic "video_scene" prompts:** For each beat, write a detailed directive for animating the image: specify camera movements (dolly, pan, tilt), motion guidance (slow‑mo, tracking), environmental effects (dust, light rays), and any mood‑enhancing filters to achieve a realistic, cinematic result.
6. **Strict JSON only:** Return **only** the JSON array. No commentary, no extra keys, no markdown—just valid JSON.

Begin by determining the number of beats, then output the array of objects accordingly.
"""

IMAGE_PROMPT = """
You are generating a cinematic image for a short video.

Here is the current script line:
"{script}"

Here is the scene description to be illustrated:
"{scene}"

Here is the extended video direction for this beat:
"{video_scene}"

**IMPORTANT**: Use the Refrence images (If Provided) where its required to maintain a proper context

Your task:1
- Create a **visually detailed, cinematic image prompt** suitable for an image generation model.
- Use the **scene** as the foundation — bring it vividly to life.
- Incorporate the **mood, tone, and narrative feel** implied by the script line.
- Include emotional expression, lighting, composition, environment, and camera framing.
- Assume this is the **next shot in a film**, and visual continuity with the previous image must be maintained (do not reset the style or tone unless the script calls for a shift).

The output should be a **single cinematic image prompt** in English, without referencing the input keys.

Make the viewer feel like they are watching the next shot of a beautifully directed film.
"""
