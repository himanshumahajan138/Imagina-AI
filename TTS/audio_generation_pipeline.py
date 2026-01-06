import os
import copy
import glob
import time
import torch
from core.logger_utils import logger
from typing import Optional, Dict, Any
from TTS.vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from TTS.vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")
        
        if not os.path.exists(voices_dir):
            logger.warning(f"Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        self.voice_presets = {}
        pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
        
        for pt_file in pt_files:
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            full_path = os.path.abspath(pt_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        logger.info(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        logger.info(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        speaker_name = speaker_name.lower()
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        matched_path = None
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_name or speaker_name in preset_name.lower():
                if matched_path is not None:
                    raise ValueError(f"Multiple voice presets match '{speaker_name}', please be more specific.")
                matched_path = path
        if matched_path is not None:
            return matched_path
        
        default_voice = list(self.voice_presets.values())[0]
        logger.warning(f"No voice preset found for '{speaker_name}', using default: {default_voice}")
        return default_voice


class VibeVoiceTTS:
    """Main TTS class for VibeVoice generation"""
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: Optional[str] = None,
    ):
        """
        Initialize VibeVoice TTS system
        
        Args:
            model_path: Path to the HuggingFace model directory
            device: Device for inference (cuda/mps/cpu). Auto-detected if None.
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Normalize mpx typo to mps
        if device.lower() == "mpx":
            logger.info("Device 'mpx' detected, treating as 'mps'")
            device = "mps"
        
        # Validate mps availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available. Falling back to CPU.")
            device = "cpu"
        
        self.device = device
        self.model_path = model_path
        logger.info(f"Using device: {self.device}")
        
        # Initialize voice mapper
        self.voice_mapper = VoiceMapper()
        
        # Load processor and model
        self._load_model()
    
    def _load_model(self):
        """Load the processor and model"""
        logger.info(f"Loading processor & model from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
        
        # Determine dtype and attention implementation
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:  # cpu
            load_dtype = torch.float32
            attn_impl = "sdpa"
        
        logger.info(f"Loading with torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")
        
        # Load model with device-specific logic
        try:
            if self.device == "mps":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:  # cpu
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            if attn_impl == 'flash_attention_2':
                logger.error(f"Error loading model: {e}")
                logger.warning("Retrying with SDPA. Note: only flash_attention_2 fully tested, quality may be lower.")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation='sdpa'
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)
        
        if hasattr(self.model.model, 'language_model'):
            logger.info(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def generate_speech(
        self,
        text: str,
        speaker_name: str = "Wayne",
        output_path: Optional[str] = None,
        cfg_scale: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Generate speech from text
        
        Args:
            text: Input text to convert to speech
            speaker_name: Name of the speaker voice to use
            output_path: Path to save output audio. If None, saves to ./outputs/
            cfg_scale: Classifier-Free Guidance scale (default: 1.5)
        
        Returns:
            Dictionary containing:
                - audio: Generated audio tensor
                - output_path: Path where audio was saved
                - generation_time: Time taken to generate
                - audio_duration: Duration of generated audio
                - rtf: Real Time Factor
                - metrics: Additional generation metrics
        """
        # Normalize quotes
        full_script = text.replace("'", "'").replace('"', '"').replace('"', '"')
        
        # Get voice preset
        target_device = self.device if self.device != "cpu" else "cpu"
        voice_sample = self.voice_mapper.get_voice_path(speaker_name)
        logger.info(f"Using voice preset for {speaker_name}: {voice_sample}")
        all_prefilled_outputs = torch.load(voice_sample, map_location=target_device, weights_only=False)
        
        # Prepare inputs
        inputs = self.processor.process_input_with_cached_prompt(
            text=full_script,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)
        
        logger.info(f"Starting generation with cfg_scale: {cfg_scale}")
        
        # Generate audio
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=True,
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
        )
        generation_time = time.time() - start_time
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        
        # Calculate metrics
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        input_tokens = inputs['tts_text_ids'].shape[1]
        output_tokens = outputs.sequences.shape[1]
        generated_tokens = output_tokens - input_tokens - all_prefilled_outputs['tts_lm']['last_hidden_state'].size(1)
        
        logger.info(f"Generated audio duration: {audio_duration:.2f} seconds")
        logger.info(f"RTF (Real Time Factor): {rtf:.2f}x")
        logger.info(f"Prefilling text tokens: {input_tokens}")
        logger.info(f"Generated speech tokens: {generated_tokens}")
        
        # Save output
        if output_path is None:
            output_dir = "./outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"generated_{int(time.time())}.wav")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
        )
        logger.info(f"Saved output to {output_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("GENERATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Output file: {output_path}")
        logger.info(f"Speaker: {speaker_name}")
        logger.info(f"Generation time: {generation_time:.2f}s")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"RTF: {rtf:.2f}x")
        logger.info("="*50)
        
        return {
            "audio": outputs.speech_outputs[0],
            "output_path": output_path,
            "generation_time": generation_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "metrics": {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": output_tokens,
            }
        }


def generate_speech_from_file(
    txt_path: str,
    speaker_name: str = "Wayne",
    output_dir: str = "./outputs",
    model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
    device: Optional[str] = None,
    cfg_scale: float = 1.5,
) -> Dict[str, Any]:
    """
    Convenience function to generate speech from a text file
    
    Args:
        txt_path: Path to text file
        speaker_name: Speaker voice name
        output_dir: Directory to save output
        model_path: Path to model
        device: Device to use (auto-detected if None)
        cfg_scale: CFG scale
    
    Returns:
        Generation results dictionary
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file not found: {txt_path}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    if not text:
        raise ValueError("No text found in file")
    
    tts = VibeVoiceTTS(model_path=model_path, device=device)
    
    txt_filename = os.path.splitext(os.path.basename(txt_path))[0]
    output_path = os.path.join(output_dir, f"{txt_filename}_generated.wav")
    
    return tts.generate_speech(
        text=text,
        speaker_name=speaker_name,
        output_path=output_path,
        cfg_scale=cfg_scale,
    )


# # Example usage
# if __name__ == "__main__":
#     # Method 1: Using the class directly
#     tts = VibeVoiceTTS(
#         model_path="microsoft/VibeVoice-Realtime-0.5B",
#         device="cuda"  # or "mps", "cpu", None for auto-detect
#     )
    
#     result = tts.generate_speech(
#         text="Hello world, this is a test of the VibeVoice system.",
#         speaker_name="Wayne",
#         output_path="./outputs/test_output.wav",
#         cfg_scale=1.5
#     )
    
#     # Method 2: Using the convenience function for file input
#     result = generate_speech_from_file(
#         txt_path="demo/text_examples/1p_vibevoice.txt",
#         speaker_name="Wayne",
#         output_dir="./outputs",
#         cfg_scale=1.5
#     )