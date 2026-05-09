SPEAKER_OPTIONS = {
    "Heart": "af_heart",
    "Bella": "af_bella",
    "Nicole": "af_nicole",
    "Aoede": "af_aoede",
    "Kore": "af_kore",
    "Sarah": "af_sarah",
    "Nova": "af_nova",
    "Sky": "af_sky",
    "Alloy": "af_alloy",
    "Jessica": "af_jessica",
    "River": "af_river",
    "Michael": "am_michael",
    "Fenrir": "am_fenrir",
    "Puck": "am_puck",
    "Echo": "am_echo",
    "Eric": "am_eric",
    "Liam": "am_liam",
    "Onyx": "am_onyx",
    "Santa": "am_santa",
    "Adam": "am_adam",
    "Emma": "bf_emma",
    "Isabella": "bf_isabella",
    "Alice": "bf_alice",
    "Lily": "bf_lily",
    "George": "bm_george",
    "Fable": "bm_fable",
    "Lewis": "bm_lewis",
    "Daniel": "bm_daniel",
}

COMMON_LANGUAGES = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Portuguese": "p",
}

DIMENSIONS = {
    "Landscape - 1536x1024": "1536x1024",
    "Portrait - 1024x1536": "1024x1536",
    "Square - 1024x1024": "1024x1024",
}

MODEL_TYPES = {
    "SORA": "openai",
    "VEO": "gemini",
    "FASTWAN": "fastwan",
}


RESOLUTIONS = {
    "720p": "scale=1280:720:flags=lanczos",
    "1080p": "scale=1920:1080:flags=lanczos",
    "4k": "scale=3840:2160:flags=lanczos",
}

ASPECT_RATIOS = {
    "1536x1024": "16:9",
    "1024x1536": "9:16",
    "1024x1024": "1:1",
}
FASTWAN_DIMENSIONS = {
    "1536x1024": (1088, 800),
    "1024x1536": (800, 1088),
    "1024x1024": (896, 896),
}

SORA_DIMENSIONS = {
    "1024x1536": "720x1280",
    "1536x1024": "1280x720",
    "1024x1024": "1280x720",
}  # , "1024x1792", "1792x1024"}
