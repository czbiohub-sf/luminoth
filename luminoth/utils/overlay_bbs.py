"""Overlay bounding boxes with labels on an image in a given directory"""
import glob
import os

import click
import cv2 as cv
import pandas as pd

FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 0, 255)
LINE_TYPE = 2
BB_COLOR = (0, 255, 0)
BB_LINE_WIDTH = 2


def overlay_bb_labels(
        im_path, df, color=BB_COLOR, line_width=BB_LINE_WIDTH):
    """
    Given a image directory, load given position and overlay
    with bounding boxes

    Args:
        im_dir: str Directory with images to overlay on
        df: pandas.DataFrame df with image_path,xmin,xmax,ymin,ymax
        pos: int Position index (FOV)
        color: tuple RGB color tuple for bounding box
            Default green (0, 255, 0)
        line_width: int Bounding box line width, Default 2
    """
    im_rgb = cv.imread(im_path, cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)

    basename = os.path.basename(im_path).replace("pred_", "")
    tmp_df = df[df.base_path == basename]

    # Plot bounding boxes

    for index, row in tmp_df.iterrows():
        label = row.label
        left_corner_of_text = (int(row.xmin), int(row.ymin))
        right_bottom_corner = (int(row.xmax), int(row.ymax))
        cv.putText(
            im_rgb,
            label,
            left_corner_of_text,
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            LINE_TYPE)

        cv.rectangle(
            im_rgb,
            left_corner_of_text,
            right_bottom_corner,
            color,
            line_width,
        )
    return im_rgb


@click.command(help="Save the bounding boxes and their annotations overlaid on the input image in the given dir as a png image")  # noqa
@click.argument("im_dir", nargs=1) # noqa
@click.option("--csv_path", help="Absolute path to csv file containing rois xmin,xmax,ymin,ymax,label", required=True) # noqa
@click.option("--output_dir", help="Absolute path to folder name to save the roi overlaid images to", required=True) # noqa
@click.option("--fmt", help="Format of images in input directory", required=True) # noqa
@click.option('--display', help="Display overlaid images, Default False")  # noqa
def overlay_bbs(im_dir, csv_path, output_dir, fmt, display):

    os.makedirs(output_dir, exist_ok=True)
    images_in_path = glob.glob(im_dir + "*." + fmt)
    df = pd.read_csv(csv_path)
    base_names = [
        os.path.basename(row.image_id) for index, row in df.iterrows()]
    df["base_path"] = pd.Series(base_names)

    for im_path in images_in_path:
        im_rgb = overlay_bb_labels(im_path, df, display=display)
        png = os.path.basename(im_path).replace(fmt, "png")
        cv.imwrite(os.path.join(output_dir, png), im_rgb)

    print("Overlaid bounding box labeled images are at {}".format(output_dir))


if __name__ == '__main__':
    overlay_bbs()
