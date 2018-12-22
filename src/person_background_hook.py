import io
import logging

import numpy as np
from PIL import Image
import PIL.ImageColor as ImageColor

LOG = logging.getLogger(__name__)
PARAMS = {
    'threshold': 0.5,
    'class': [1]
}


def init_hook(**params):
    # PARAMS['threshold'] = params.get('threshold',0.5)
    # PARAMS['class'] = params.get('class',1)
    LOG.info('Loaded.')
    LOG.info('initialized with params: %s', PARAMS)


def preprocess(inputs, ctx):
    image = inputs.get('inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    np_image = np.array(image)
    ctx.image = image
    return {'image_tensor': [np_image]}


def postprocess(outputs, ctx):
    num_detection = int(outputs['num_detections'][0])
    logging.info('num_detection: {}'.format(num_detection))
    if num_detection < 1:
        image_bytes = io.BytesIO()
        ctx.image.save(image_bytes, format='PNG')
        outputs['output'] = image_bytes.getvalue()
        return outputs

    width = ctx.image.size[0]
    height = ctx.image.size[1]

    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    logging.info('detection_boxes: {}'.format(detection_boxes))
    detection_boxes = detection_boxes * [height, width, height, width]
    detection_boxes = detection_boxes.astype(np.int32)
    # detection_scores = outputs["detection_scores"][0][:num_detection]
    detection_classes = outputs["detection_classes"][0][:num_detection]
    logging.info('detection_classes: {}'.format(detection_classes))
    detection_masks = outputs["detection_masks"][0][:num_detection]

    total_mask = np.zeros((height, width), np.float32)
    for i in range(num_detection):
        if int(detection_classes[i]) not in PARAMS['class']:
            continue
        mask_image = Image.fromarray(detection_masks[i])
        box = detection_boxes[i]
        mask_image = mask_image.resize((box[3] - box[1], box[2] - box[0]), Image.LANCZOS)
        box_mask = np.array(mask_image)
        box_mask = np.pad(box_mask, ((box[0], height - box[2]), (box[1], width - box[3])), 'constant')
        total_mask += box_mask
    mask = np.less(total_mask, 0.5).astype(np.int32)
    color = 'white'
    alpha = 1
    rgb = ImageColor.getrgb(color)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, ctx.image, pil_mask)
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs
