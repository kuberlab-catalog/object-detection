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
    np_image = np.array(image).astype(np.float32)/127.5-1
    return {'images': [np_image]}


def postprocess(outputs, ctx):
    image = outputs['result'][0]
    image = (image+1)*127.5
    image = np.clip(image,0,255)
    image = Image.fromarray(np.uint8(image))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs
