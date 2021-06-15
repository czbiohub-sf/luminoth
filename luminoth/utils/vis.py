"""Provides various visualization-specific functions."""
import os
import sys

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FONT_SCALE = 2
FONT_COLOR = (255, 219, 140, 0)
LINE_TYPE = 2
BB_COLOR = (182, 189, 224)
BB_LINE_WIDTH = 2


def get_font():
    """Attempts to retrieve a reasonably-looking TTF font from the system.
    We don't make much of an effort, but it's what we can reasonably do without
    incorporating additional dependencies for this task.
    """
    if sys.platform == "win32":
        font_names = ["Arial"]
    elif sys.platform in ["linux", "linux2"]:
        font_names = ["DejaVuSans-Bold", "DroidSans-Bold"]
    elif sys.platform == "darwin":
        font_names = ["Menlo", "Helvetica"]

    font = None
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name)
            break
        except IOError:
            continue

    return font


SYSTEM_FONT = get_font()


def draw_label(im_rgb, coords, label, prob, color, scale=1):
    """Draw a box with the label and probability."""
    # Attempt to get a native TTF font. If not, use the default bitmap font.
    # Rescale unit16 or other images to uint8
    im_rgb = ((im_rgb - im_rgb.min()) / (im_rgb.max() - im_rgb.min()) * 255).astype(
        np.uint8
    )
    global SYSTEM_FONT
    if SYSTEM_FONT:
        label_font = SYSTEM_FONT.font_variant(size=int(round(16 * scale)))
        prob_font = SYSTEM_FONT.font_variant(size=int(round(12 * scale)))
    else:
        label_font = ImageFont.load_default()
        prob_font = ImageFont.load_default()

    label = str(label)  # `label` may not be a string.

    # We want the probability font to be smaller, so we'll write the label in
    # two steps.
    label_w, label_h = label_font.getsize(label)

    # Get margins to manually adjust the spacing. The margin goes between each
    # segment (i.e. margin, label, margin, prob, margin).
    margin_w, margin_h = label_font.getsize("M")
    margin_w *= 0.2

    # Convert the image to RGB (OpenCV uses BGR)
    # Pass the image to PIL
    pil_im = Image.fromarray(im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # Then write the two pieces of text.
    draw.text(
        [
            coords[0] + margin_w,
            coords[1],
        ],
        label,
        font=label_font,
        fill=color,
    )

    if prob is not None:
        prob = "({:.3f})".format(prob)  # Turn `prob` into a string.
        prob_w, prob_h = prob_font.getsize(prob)
        draw.text(
            [
                coords[0] + label_w + 2 * margin_w,
                coords[1] + (margin_h - prob_h),
            ],
            prob,
            font=prob_font,
        )
        # Get back the image to OpenCV

    return np.array(pil_im)


def vis_objects(
    im_rgb,
    objects,
    color=[BB_COLOR, FONT_COLOR],
    labels=True,
    scale=FONT_SCALE,
    line_width=BB_LINE_WIDTH,
):
    """Visualize objects as returned by `Detector`.

    Arguments:
        im_rgb (numpy.ndarray): Image to draw the bounding boxes on.
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
    # Rescale unit16 or other images to uint8
    im_rgb = ((im_rgb - im_rgb.min()) / (im_rgb.max() - im_rgb.min()) * 255).astype(
        np.uint8
    )
    if len(im_rgb.shape) == 3:
        if im_rgb.shape[2] == 1:
            im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    elif len(im_rgb.shape) == 2:
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        bbox = obj["bbox"]
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
                im_rgb, bbox, obj["label"], obj["prob"], color[1], scale
            )

    return im_rgb
