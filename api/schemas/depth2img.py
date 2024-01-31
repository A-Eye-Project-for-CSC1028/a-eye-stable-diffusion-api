# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/depth2img
from marshmallow import Schema, fields
from PIL import Image
from torch import ByteTensor, FloatTensor, Generator


class FloatTensorField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return value.tolist()

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        return FloatTensor(value)


class GeneratorField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return value.get_state().tolist()

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        generator = Generator()
        generator.set_state(ByteTensor(value))
        return generator


class Depth2Img():
    def __init__(self, prompt: str | None, image: Image.Image, depth_map: FloatTensor | None,
                 strength: float | None, num_inference_steps: int | None, guidance_scale: float | None,
                 negative_prompt: str | None, num_images_per_prompt: int | None, eta: float | None,
                 generator: Generator | None, prompt_embeds: FloatTensor | None, negative_prompt_embeds: FloatTensor | None,
                 output_type: str | None, return_dict: bool | None, clip_skip: int | None) -> None:
        self.prompt = prompt
        self.image = image
        self.depth_map = depth_map
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.num_images_per_prompt = num_images_per_prompt
        self.eta = eta
        self.generator = generator
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.output_type = output_type
        self.return_dict = return_dict
        self.clip_skip = clip_skip


class Depth2ImgSchema(Schema):
    prompt = fields.String(missing='')
    image = fields.Raw(type='file')
    depth_map = FloatTensorField(missing=None)
    strength = fields.Float(missing=0.8)
    num_inference_steps = fields.Integer(missing=50)
    guidance_scale = fields.Float(missing=7.5)
    negative_prompt = fields.String(missing='')
    num_images_per_prompt = fields.Integer(missing=1)
    # Only considered when using DDIMScheduler.
    eta = fields.Float(missing=0.0)
    generator = GeneratorField(missing=None)
    prompt_embeds = FloatTensorField(missing=None)
    negative_prompt_embeds = FloatTensorField(missing=None)
    output_type = fields.String(missing='pil')
    return_dict = fields.Boolean(missing=True)
    clip_skip = fields.Integer(missing=None)
