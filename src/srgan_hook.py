import io
import logging

import numpy as np
from PIL import Image


LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')


def preprocess(inputs, ctx):
    image = inputs.get('images')
    if image is None:
        raise RuntimeError('Missing "images" key in inputs. Provide an image in "images" key')

    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    width = image.size[0]
    height = image.size[1]
    down_width = width//4
    down_heigh = height//4
    image = image.resize((down_width,down_heigh),Image.LANCZOS)
    np_image = np.array(image).astype(np.float32)/127.5-1
    ctx.width  = width
    ctx.height = height
    return {'images': [np_image]}


def postprocess(outputs, ctx):
    image = outputs['result'][0]
    image = (image+1)*127.5
    image = np.clip(image,0,255)
    image = Image.fromarray(np.uint8(image))
    image = image.resize((ctx.width,ctx.height),Image.LANCZOS)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs
