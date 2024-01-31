from datetime import time
import io

from flask import Flask, request, send_file, abort
from marshmallow import ValidationError
from PIL import Image
from torch import FloatTensor, Generator


from image_generator import ImageGenerator
from schemas.depth2img import Depth2Img


app = Flask(__name__)


@app.post('/depth2img')
def generate_via_depth2img(
    prompt: str = None,
    depth_map: FloatTensor = None,
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Generator = None,
    prompt_embeds: FloatTensor = None,
    negative_prompt_embeds: FloatTensor = None,
    output_type: str = 'pil',
    return_dict: bool = True,
    clip_skip: int = None
):
    try:
        if 'image' not in request.files:
            abort(400, 'No image file provided.')

        file = request.files['image']
        if file:
            image = Image.open(io.BytesIO(file.read()))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            params = Depth2Img(
                prompt=prompt,
                image=image,
                depth_map=depth_map,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                clip_skip=clip_skip
            )

            generated_image = ImageGenerator.depth2img(params)

            img_byte_arr = io.BytesIO()
            generated_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            return send_file(
                img_byte_arr,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f'${time.time()}.jpg'
            )
    except ValidationError as e:
        abort(400, str(e))
    except Exception as e:
        abort(500, str(e))


if __name__ == '__main__':
    app.run(debug=True)
