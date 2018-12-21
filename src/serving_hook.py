import io
import logging

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import label_map_util
import visualization_utils as vis_utils


LOG = logging.getLogger(__name__)
PARAMS = {
    'threshold': 0.7,
    'skip_labels': False,
    'skip_scores': False,
    'line_thickness': 3,
    'max_boxes': 20,
}
category_index = None


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def init_hook(**params):
    threshold = params.get('threshold')
    skip_scores = params.get('skip_scores')
    skip_labels = params.get('skip_labels')
    thickness = params.get('line_thickness')
    max_boxes = params.get('max_boxes')
    if skip_labels:
        PARAMS['skip_labels'] = boolean_string(skip_labels)

    if skip_scores:
        PARAMS['skip_scores'] = boolean_string(skip_labels)

    if threshold:
        PARAMS['threshold'] = float(threshold)

    if thickness:
        PARAMS['line_thickness'] = int(thickness)

    if max_boxes:
        PARAMS['max_boxes'] = int(max_boxes)

    label_map_path = params.get('label_map')
    if not label_map_path:
        raise RuntimeError(
            'Label map required. Provide path to label_map via'
            ' -o label_map=<label_map.pbtxt>'
        )

    LOG.info('Loading label map from %s...' % label_map_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes
    )
    global category_index
    category_index = label_map_util.create_category_index(categories)
    LOG.info('Loaded.')
    LOG.info('Initialized with params: %s', PARAMS)


def preprocess(inputs, ctx):
    image = inputs.get('inputs')
    if image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    threshold = inputs.get('threshold')
    ctx.threshold = float(threshold)\
        if (threshold is not None and float(threshold) > 0)\
        else PARAMS['threshold']

    line_thickness = inputs.get('line_thickness')
    ctx.line_thickness = int(line_thickness)\
        if (line_thickness is not None and int(line_thickness) > 0)\
        else PARAMS['line_thickness']

    max_boxes = inputs.get('max_boxes')
    ctx.max_boxes = int(max_boxes)\
        if (max_boxes is not None and int(max_boxes) > 0)\
        else PARAMS['max_boxes']

    skip_labels = inputs.get('skip_labels')
    ctx.skip_labels = skip_labels[0] if skip_labels is not None else PARAMS['skip_labels']
    # ctx.skip_labels = boolean_string(skip_labels[0].decode('utf-8'))\
    #     if (skip_labels is not None and len(skip_labels) > 0)\
    #     else PARAMS['skip_labels']

    skip_scores = inputs.get('skip_scores')
    ctx.skip_scores = skip_scores[0] if skip_scores is not None else PARAMS['skip_scores']
    # ctx.skip_scores = boolean_string(skip_scores[0].decode('utf-8'))\
    #     if (skip_scores is not None and len(skip_scores) > 0)\
    #     else PARAMS['skip_scores']

    LOG.info('Visualization params: threshold %f, line_thickness %d, max_boxes %d, skip_labels %r, skip_scores %r' %
             (ctx.threshold, ctx.line_thickness, ctx.max_boxes, ctx.skip_labels, ctx.skip_scores))

    ctx.image = Image.open(io.BytesIO(image[0]))
    return {'inputs': image}


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def add_overlays(frame, boxes, labels=None):
    draw = ImageDraw.Draw(frame)
    if boxes is not None:
        for i, face in enumerate(boxes):
            face_bb = face.astype(int)
            draw_rectangle(
                draw,
                [(face_bb[0], face_bb[1]), (face_bb[2], face_bb[3])],
                (0, 255, 0), width=2
            )

            if labels:
                draw.text(
                    (face_bb[0] + 4, face_bb[1] + 5),
                    labels[i], font=ImageFont.load_default(),
                )


def postprocess(outputs, ctx):
    # keys
    # "detection_boxes"
    # "detection_classes"
    # "detection_scores"
    # "num_detections"
    detection_boxes = outputs["detection_boxes"].reshape([-1, 4])
    detection_scores = outputs["detection_scores"].reshape([-1])
    detection_classes = np.int32((outputs["detection_classes"])).reshape([-1])
    width = ctx.image.size[0]
    height = ctx.image.size[1]
    ctx.image = ctx.image.convert('RGB')
    image_arr = np.array(ctx.image)

    LOG.info('Visualization params: threshold %f, line_thickness %d, max_boxes %d, skip_labels %r, skip_scores %r' %
             (ctx.threshold, ctx.line_thickness, ctx.max_boxes, ctx.skip_labels, ctx.skip_scores))

    vis_utils.visualize_boxes_and_labels_on_image_array(
        image_arr,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=ctx.max_boxes,
        min_score_thresh=ctx.threshold,
        agnostic_mode=False,
        line_thickness=ctx.line_thickness,
        skip_labels=ctx.skip_labels,
        skip_scores=ctx.skip_scores,
    )
    from_arr = Image.fromarray(image_arr)
    image_bytes = io.BytesIO()
    from_arr.save(image_bytes, format='PNG')

    outputs['output'] = image_bytes.getvalue()
    return outputs
