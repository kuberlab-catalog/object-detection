import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageFilter

LOG = logging.getLogger(__name__)


def init_hook(**params):
    # PARAMS['threshold'] = params.get('threshold',0.5)
    # PARAMS['class'] = params.get('class',1)
    LOG.info('Loaded.')


def preprocess(inputs, ctx):
    image = inputs.get('inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    np_image = np.array(image)
    ctx.image = image
    ctx.area_threshold = int(inputs.get('area_threshold', 0))
    ctx.max_objects = int(inputs.get('max_objects', 100))
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.object_classes = [int(inputs.get('object_class', 1))]
    ctx.image_filter = int(inputs.get('image_filter', 1))
    ctx.blur_radius = int(inputs.get('blur_radius', 2))
    return {'inputs': [np_image]}


def postprocess(outputs, ctx):
    num_detection = int(outputs['num_detections'][0])

    def return_original():
        image_bytes = io.BytesIO()
        ctx.image.save(image_bytes, format='PNG')
        outputs['output'] = image_bytes.getvalue()
        return outputs

    if num_detection < 1:
        return return_original()

    width = ctx.image.size[0]
    height = ctx.image.size[1]
    image_area = width * height
    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    detection_boxes = detection_boxes * [height, width, height, width]
    detection_boxes = detection_boxes.astype(np.int32)
    # detection_scores = outputs["detection_scores"][0][:num_detection]
    detection_classes = outputs["detection_classes"][0][:num_detection]
    detection_masks = outputs["detection_masks"][0][:num_detection]

    masks = []
    for i in range(num_detection):
        if int(detection_classes[i]) not in ctx.object_classes:
            continue
        mask_image = Image.fromarray(detection_masks[i])
        box = detection_boxes[i]
        mask_image = mask_image.resize((box[3] - box[1], box[2] - box[0]), Image.NEAREST)
        box_mask = np.array(mask_image)
        box_mask = np.pad(box_mask, ((box[0], height - box[2]), (box[1], width - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, ctx.pixel_threshold).astype(np.int32)))
        if area * 100 / image_area < ctx.area_threshold:
            continue
        masks.append((area, box_mask))

    if len(masks) < 1:
        return return_original()
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((height, width), np.float32)
    for i in range(min(len(masks), ctx.max_objects)):
        total_mask = np.maximum(total_mask,masks[i][1])
    if ctx.image_filter == 0:
        mask = np.less(total_mask, ctx.pixel_threshold)
        image = np.array(ctx.image)
        image = np.dstack((image, np.ones((height, width)) * 255))
        image[mask] = 0
        image = Image.fromarray(np.uint8(image))
    elif ctx.image_filter == 5:
        #mask = np.less(total_mask, ctx.pixel_threshold)
        #total_mask[mask] = 0
        total_mask = total_mask*255
        image = Image.fromarray(np.uint8(total_mask))
    else:
        mask = np.greater_equal(total_mask, ctx.pixel_threshold)
        image = np.array(ctx.image)
        objects = image[mask]
        radius = min(max(ctx.blur_radius,2),10)
        image = ctx.image.filter(ImageFilter.GaussianBlur(radius=radius))
        image = np.array(image)
        image[mask] = objects
        image = Image.fromarray(np.uint8(image))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs
