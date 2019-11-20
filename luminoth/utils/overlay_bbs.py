"""Overlay bounding boxes with labels on an image in a given directory"""
import glob
import os

import click
import cv2
import pandas as pd

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 0, 255)
LINE_TYPE = 2
BB_COLOR = (0, 255, 0)
BB_LINE_WIDTH = 2


def add_base_path(csv_path, input_image_format, image_path_column):
    """
    Returns a dataframe with all the bounding boxes & labels
    from all the image paths in csv_path. Also adds a column base_path which
    is just the basename of a fullpath, e.g /data/bla/img.tif's
    base_path would be img

    Args:
        csv_path: Path to comma separated txt/csv file with
            image_path,x1,y1,x2,y2,class_name or
            image_id,xmin,ymin,xmax,ymax,label
        input_image_format: remove this format tag from the basename of the
            image_path while adding base_path column
        image_path_column: str name of the image_path_column

    Returns:
        bb_labels_df: Returns a dataframe with all the bounding boxes & labels
            from all the paths in filenames. Also adds a column base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be img
    """
    dfs = []
    dfs.append(pd.read_csv(csv_path))
    bb_labels_df = pd.concat(dfs, ignore_index=True)
    base_names = [
        os.path.basename(row[image_path_column]).replace(
            input_image_format, "") for index, row in bb_labels_df.iterrows()]
    bb_labels_df["base_path"] = pd.Series(base_names)
    return bb_labels_df


def overlay_bb_labels(
        im_path,
        input_image_format,
        df,
        color=BB_COLOR,
        line_width=BB_LINE_WIDTH):
    """
    Given a image directory, load given position and overlay
    with bounding boxes

    Args:
        im_path: str Path of an image to overlay on
        input_image_format: str Format of the input images
        df: pandas.DataFrame df with image_path,x1,y1,x2,y2,class_name or
            image_path,xmin,ymin,xmax,ymax,label
        pos: int Position index (FOV)
        color: tuple RGB color tuple for bounding box
            Default green (0, 255, 0)
        line_width: int Bounding box line width, Default 2

    Returns:
        im_rgb: overlaid_image with same shape and 3 channels
    """
    im_rgb = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    shape = im_rgb.shape
    if len(shape) == 2:
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)
    elif len(shape) == 3:
        if shape[2] != 3:
            im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)

    basename = os.path.basename(im_path).replace(input_image_format, "")
    tmp_df = df[
        [True if row['base_path'].endswith(
            basename) else False for index, row in df.iterrows()]]

    # Plot bounding boxes, annotation label
    for index, row in tmp_df.iterrows():
        label = str(row.label)
        left_corner_of_text = (int(row.xmin), int(row.ymin))
        right_bottom_corner = (int(row.xmax), int(row.ymax))
        cv2.putText(
            im_rgb,
            label,
            left_corner_of_text,
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            LINE_TYPE)

        cv2.rectangle(
            im_rgb,
            left_corner_of_text,
            right_bottom_corner,
            color,
            line_width,
        )
    return im_rgb


def overlay_bbs_on_all_images(
        im_dir, csv_path, image_path_column, output_dir, input_image_format):
    """
    Save the bounding boxes and their annotations overlaid
     on the input image in the given dir as a png image
     in the output_dir

    Args:
        im_dir: str Directory with images to overlay on
        csv_path: Path to comma separated txt/csv file with
            image_path,x1,y1,x2,y2,class_name or
            image_idh,xmin,ymin,xmax,ymax,label
        image_path_column: str name of the image_path_column
        output_dir: str Directory to save overlaid images to
        input_image_format: str Format of the input images

    Returns:
        Writes overlaid images to output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(
            "Path {} already exists, might be overwriting data".format(
                output_dir))
    images_in_path = glob.glob(
        os.path.join(im_dir, "*" + input_image_format))
    df = add_base_path(csv_path, input_image_format, image_path_column)

    for im_path in images_in_path:
        im_rgb = overlay_bb_labels(im_path, input_image_format, df)
        png = os.path.basename(im_path).replace(input_image_format, ".png")
        cv2.imwrite(os.path.join(output_dir, png), im_rgb)
    print("Overlaid bounding box labeled images are at: {}".format(output_dir))


@click.command(help="Save the bounding boxes and their annotations overlaid on the input images in the given dir as png images in the output_dir")  # noqa
@click.option("--im_dir", help="Path to directory containing images to overlay on", type=str, required=True) # noqa
@click.option("--csv_path", help="Absolute path to csv file containing rois xmin,xmax,ymin,ymax,label,image_path", required=True, type=str) # noqa
@click.option("--image_path_column", help="Name of the image path column, it is often image_id for lumi output, or could be image_path if saved outside of lumi", required=True, type=str) # noqa
@click.option("--output_dir", help="Absolute path to folder name to save the roi overlaid images to", required=True, type=str) # noqa
@click.option("--input_image_format", help="Format of images in input directory", required=True, type=str) # noqa
def overlay_bbs(
        im_dir, csv_path, image_path_column, output_dir, input_image_format):

    overlay_bbs_on_all_images(
        im_dir,
        csv_path,
        image_path_column,
        output_dir,
        input_image_format)


if __name__ == '__main__':
    overlay_bbs()
