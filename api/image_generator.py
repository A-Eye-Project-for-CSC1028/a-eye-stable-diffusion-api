import torch
from schemas.depth2img import Depth2Img
from diffusers import StableDiffusionDepth2ImgPipeline


class ImageGenerator:
    @staticmethod
    def depth2img(params: Depth2Img):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        pipe.to("cuda")

        result = pipe(
            prompt=params.prompt if params.prompt else '',
            image=params.image,
            depth_map=params.depth_map,
            strength=params.strength if params.strength else 0.8,
            num_inference_steps=params.num_inference_steps if params.num_inference_steps else 50,
            guidance_scale=params.guidance_scale if params.guidance_scale else 7.5,
            negative_prompt=params.negative_prompt if params.negative_prompt else '',
            num_images_per_prompt=params.num_images_per_prompt if params.num_images_per_prompt else 1,
            eta=params.eta if params.eta else 0.0,
            generator=params.generator,
            prompt_embeds=params.prompt_embeds,
            negative_prompt_embeds=params.negative_prompt_embeds,
            output_type=params.output_type if params.output_type else 'pil',
            return_dict=params.return_dict if params.return_dict else True,
            clip_skip=params.clip_skip
        )

        return result.images[0]
