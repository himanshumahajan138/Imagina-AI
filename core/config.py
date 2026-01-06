COMMON_LANGUAGES = [
    "english",
    "chinese",
]
SPEAKER_OPTIONS = {
    "Emma": COMMON_LANGUAGES,
    "Frank": COMMON_LANGUAGES,
    "Carter": COMMON_LANGUAGES,
    "Davis": COMMON_LANGUAGES,
    "Mike": COMMON_LANGUAGES,
    "Samuel": COMMON_LANGUAGES,
    "Wayne": COMMON_LANGUAGES,
}
DIMENSIONS = {
    "Landscape - 1536x1024": "1536x1024",
    "Portrait - 1024x1536": "1024x1536",
    "Square - 1024x1024": "1024x1024",
}

MODEL_TYPES = {
    "SORA": "openai",
    "VEO": "gemini",
    # "FASTWAN": "fastwan",
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
