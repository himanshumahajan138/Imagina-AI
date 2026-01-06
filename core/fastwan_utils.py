# import base64
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from core.logger_utils import logger
# from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

# # Initialize the generator once (can be done at module level or passed as parameter)
# _generator = None


# def get_generator():
#     """Initialize and cache the video generator"""
#     global _generator
#     if _generator is None:
#         model_name = "Wan-AI/FastWan2.2-TI2V-5B"
#         config = PipelineConfig.from_pretrained(model_name)
#         config.vae_precision = "fp16"
#         config.dit_cpu_offload = True

#         _generator = VideoGenerator.from_pretrained(
#             model_name, num_gpus=1, pipeline_config=config
#         )
#     return _generator


# def fastwan_video_generation(
#     prompt: str,
#     height: int,
#     width: int,
#     duration: int,
#     image: str,
#     output_file: Path,
#     attempt: int = 1,
# ):
#     if attempt > 3:
#         logger.error("Maximum retry attempts reached. Aborting.")
#         return ""

#     try:
#         logger.info(f"Attempt {attempt}: Generating video with FastWan.")

#         # Get the generator
#         generator = get_generator()

#         # Create sampling parameters
#         sampling_param = SamplingParam.from_pretrained("Wan-AI/FastWan2.2-TI2V-5B")

#         # Configure parameters based on input
#         sampling_param.width = width
#         sampling_param.height = height

#         # Convert duration (seconds) to frames (assuming ~30fps)
#         # Adjust fps based on your needs
#         fps = 30
#         sampling_param.num_frames = int(duration * fps)

#         # Set other parameters matching your original config
#         sampling_param.guidance_scale = (
#             0 if hasattr(sampling_param, "guidance_scale") else 7.5
#         )
#         sampling_param.num_inference_steps = 5  # matches 'steps' in original
#         sampling_param.seed = 42

#         # Handle the input image
#         # Decode base64 image if it's a base64 string
#         init_image = None
#         if image:
#             if image.startswith("data:image") or len(image) > 500:  # Likely base64
#                 # Remove data URI prefix if present
#                 if "," in image:
#                     image = image.split(",", 1)[1]
#                 image_bytes = base64.b64decode(image)
#                 # Convert to PIL Image
#                 from io import BytesIO

#                 init_image = Image.open(BytesIO(image_bytes))
#             else:
#                 # Assume it's a file path
#                 init_image = Image.open(image)  # type: ignore

#         # Generate video
#         # Note: fastvideo might not support init_image directly in generate_video
#         # You may need to check the library documentation for image-to-video support
#         video = generator.generate_video(
#             prompt,
#             sampling_param=sampling_param,
#             output_path=str(output_file.parent),  # Directory path
#             return_frames=False,
#             save_video=True,
#         )  # type: ignore

#         # If the library doesn't save with your exact filename, rename it
#         # This depends on how the library names files
#         logger.info(f"Video saved to {output_file}")
#         return str(output_file)

#     except Exception as e:
#         logger.exception(f"Error on attempt {attempt}: {e}")
#         if attempt < 3:
#             logger.info(f"Retrying... (attempt {attempt + 1})")
#             return fastwan_video_generation(
#                 prompt, height, width, duration, image, output_file, attempt + 1
#             )
#         else:
#             return ""
