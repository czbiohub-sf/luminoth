"""Provides various visualization-specific functions."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
LINE_TYPE = 2
BB_COLOR = (0, 255, 0)
BB_LINE_WIDTH = 2


def draw_label(im_rgb, bbox, label, prob, color=FONT_COLOR, scale=FONT_SCALE):
    """Visualize objects as returned by `Detector`.

    Arguments:
        im_rgb (numpy.ndarray): Color Image to draw the label,prob on.
        bbox (List of 4 int/float): Indicates the bounding box boundaries
        label (str or int): Label of the bbox
        prob(float): Probability of the label for the given bbox
        color (list of 3-tuples): List of 2 rgb tuples indicating the color for
            bounding box and font
        scale (float): Scale factor for the font

    Returns:
        numpy.ndarray: Color image with bounding box and labels drawn on
    """

    label = str(label) + '({:.2f})'.format(prob)  # Turn `prob` into a string.

    left_corner_of_text = (int(bbox[0]), int(bbox[1]))
    cv2.putText(
        im_rgb,
        label,
        left_corner_of_text,
        FONT,
        FONT_SCALE,
        FONT_COLOR,
        LINE_TYPE)


def vis_objects(
        image, objects, color=[BB_COLOR, FONT_COLOR], labels=True,
        scale=FONT_SCALE, line_width=BB_LINE_WIDTH):
    """Visualize objects as returned by `Detector`.

    Arguments:
        image (numpy.ndarray): Image to draw the bounding boxes on.
        objects (list of dicts or dict): List of objects as returned by a
            `Detector` instance.
        color (list of 3-tuples): List of 2 rgb tuples indicating the color for
            bounding box and font
        labels (boolean): If true, labels are plotted on the bbounding box with
            probabilities
        scale (float): Scale factor for the font
        line_width (int): width of the bounding box

    Returns:
        numpy.ndarray: Color image with bounding box and labels drawn on
    """
    image = image.astype(np.uint8)

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            im_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 2:
        im_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        bbox = obj['bbox']
        left_corner_of_text = (int(bbox[0]), int(bbox[1]))
        right_bottom_corner = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(
            im_rgb,
            left_corner_of_text,
            right_bottom_corner,
            color[0],
            line_width,
        )
        if labels:
            draw_label(
                im_rgb, bbox, obj['label'], obj['prob'], color[1], scale)

    return image
