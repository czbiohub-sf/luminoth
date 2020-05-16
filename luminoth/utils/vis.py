"""Provides various visualization-specific functions."""
import os

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FONT = os.path.join(CURRENT_DIR, "arial.ttf")
FONT_SCALE = 10000000
FONT_COLOR = (255, 219, 140, 0)
LINE_TYPE = 2
BB_COLOR = (224, 189, 182)
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

    left_corner_of_text = (
        int(bbox[0] + BB_LINE_WIDTH), int(bbox[1] + BB_LINE_WIDTH))

    # Convert the image to RGB (OpenCV uses BGR)
    cv2_im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)

    # Pass the image to PIL
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    font = ImageFont.truetype(FONT, 28, encoding="unic")

    # Draw the text
    draw.text(left_corner_of_text, label, font=font, fill=color)

    # Get back the image to OpenCV
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return cv2_im_processed


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
    im_rgb = image.astype(np.uint8)

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 2:
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)

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
            im_rgb = draw_label(
                im_rgb, bbox, obj['label'], obj['prob'], color[1], scale)

    return im_rgb
