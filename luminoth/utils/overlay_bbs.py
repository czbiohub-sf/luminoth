import glob
import os

import click
import cv2
import pandas as pd

# Constants for bounding box and label overlays
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)
LINE_TYPE = 2
BB_COLOR = (0, 255, 0)
BB_LINE_WIDTH = 2


def add_base_path(csv_path, input_image_format):
    """
    Returns a dataframe with all the bounding boxes & labels
    from all the image paths reading from csv_path.
    Also adds a column base_path which is just the basename of a fullpath,
    e.g /data/bla/img.tif's base_path would be img

    Args:
        csv_path: str Path to comma separated txt/csv file with
            [image_id,xmin,ymin,xmax,ymax,label]
        input_image_format: str remove this format tag from the basename of the
            image_path while adding base_path column

    Returns:
        bb_labels_df: pandas.DataFrame df with all the bounding boxes & labels
            from all the paths in filenames. Also adds a column base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be img
    """
    dfs = []
    dfs.append(pd.read_csv(csv_path))
    bb_labels_df = pd.concat(dfs, ignore_index=True)
    base_names = [
        os.path.basename(row['image_id']).replace(
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
    Return an image overlaid with bounding box rectangles and the annotations

    Args:
        im_path: str Path of an image to overlay on
        input_image_format: str Format of the input image
        df: pandas.DataFrame df with image_path,xmin,ymin,xmax,ymax,label
        color: tuple RGB color tuple for rectangle drawn around an object in
            bounding box, Default green (0, 255, 0)
        line_width: int Bounding box line width, Default 2

    Returns:
        im_rgb: np.array overlaid_image with same shape and 3 channels
    """
    im_rgb = cv2.imread(im_path, cv2.IMREAD_COLOR)

    basename = os.path.basename(im_path).replace(input_image_format, "")
    tmp_df = df[df.base_path == basename]

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
        im_dir, csv_path, output_dir, input_image_format):
    """
    Save the bounding boxes and their annotations overlaid
     on all the input images in the given dir as a png image
     in the output_dir

    Args:
        im_dir: str Directory with images to overlay on
        csv_path: str Path to comma separated txt/csv file with
            image_id,xmin,ymin,xmax,ymax,label
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
    df = add_base_path(csv_path, input_image_format)

    for im_path in images_in_path:
        im_rgb = overlay_bb_labels(im_path, input_image_format, df)
        png = os.path.basename(im_path).replace(input_image_format, ".png")
        cv2.imwrite(os.path.join(output_dir, png), im_rgb)
    print("Overlaid bounding box labeled images are at: {}".format(output_dir))


@click.command(help="Save the bounding boxes and their annotations overlaid on the input images in the given dir as png images in the output_dir")  # noqa
@click.option("--im_dir", help="Path to directory containing images to overlay on", type=str, required=True) # noqa
@click.option("--csv_path", help="Absolute path to csv file containing rois xmin,xmax,ymin,ymax,label,image_path", required=True, type=str) # noqa
@click.option("--output_dir", help="Absolute path to folder name to save the roi overlaid images to", required=True, type=str) # noqa
@click.option("--input_image_format", help="Format of images in input directory", required=True, type=str) # noqa
def overlay_bbs(im_dir, csv_path, output_dir, input_image_format):

    overlay_bbs_on_all_images(
        im_dir,
        csv_path,
        output_dir,
        input_image_format)


if __name__ == '__main__':
    overlay_bbs()
