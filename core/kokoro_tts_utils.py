"""
Kokoro Audio Generation Pipeline
A clean, reusable pipeline for text-to-speech conversion using Kokoro TTS
"""

from kokoro import KPipeline
import soundfile as sf
import numpy as np
import torch
from typing import Optional, Union


class KokoroAudioPipeline:
    """
    A clean pipeline for generating audio from text using Kokoro TTS.

    Supported Languages:
    - 'a': American English (default)
    - 'b': British English
    - 'e': Spanish
    - 'f': French
    - 'h': Hindi
    - 'i': Italian
    - 'j': Japanese (requires: pip install misaki[ja])
    - 'p': Brazilian Portuguese
    - 'z': Mandarin Chinese (requires: pip install misaki[zh])
    """

    def __init__(self, lang_code: str = "a"):
        """
        Initialize the Kokoro pipeline.

        Args:
            lang_code: Language code for the TTS model
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self.sample_rate = 24000

    def text_to_audio(
        self,
        text: str,
        voice: Union[str, torch.Tensor] = "af_heart",
        speed: float = 1.0,
        split_pattern: str = r"\n+",
        output_file: Optional[str] = None,
        combine_segments: bool = True,
    ) -> np.ndarray:
        """
        Convert text to audio.

        Args:
            text: Input text to convert to speech
            voice: Voice identifier (e.g., 'af_heart', 'am_adam') or voice tensor
            speed: Speech speed multiplier (1.0 = normal)
            split_pattern: Regex pattern for splitting text into segments
            output_file: Optional path to save the audio file
            combine_segments: If True, combines all segments into one audio array

        Returns:
            numpy array containing the audio data
        """
        generator = self.pipeline(
            text, voice=voice, speed=speed, split_pattern=split_pattern
        )

        audio_segments = []

        for i, (graphemes, phonemes, audio) in enumerate(generator):
            audio_segments.append(audio)

        if combine_segments:
            combined_audio = np.concatenate(audio_segments)
        else:
            combined_audio = audio_segments

        if output_file:
            self.save_audio(combined_audio, output_file)

        return combined_audio

    def save_audio(self, audio: np.ndarray, output_file: str):
        """
        Save audio data to a WAV file.

        Args:
            audio: Audio data as numpy array
            output_file: Path to save the audio file
        """
        sf.write(output_file, audio, self.sample_rate)
        print(f"✓ Audio saved to: {output_file}")

    def load_voice_tensor(self, voice_path: str) -> torch.Tensor:
        """
        Load a custom voice tensor from file.

        Args:
            voice_path: Path to the voice tensor file (.pt)

        Returns:
            Voice tensor
        """
        return torch.load(voice_path, weights_only=True)


# # Example usage
# if __name__ == "__main__":
#     pipeline = KokoroAudioPipeline(lang_code='a')

#     longer_text = """
#     The Kokoro TTS model is an open-weight text-to-speech system.
#     It delivers high-quality audio generation with excellent performance.
#     Perfect for both production and personal projects.
#     """

#     audio = pipeline.text_to_audio(
#         text=longer_text,
#         voice='af_heart',
#         speed=1.0,
#         output_file="example_output.wav"
#     )

#     print(f"✓ Pipeline complete! Generated {len(audio)} audio samples.")
