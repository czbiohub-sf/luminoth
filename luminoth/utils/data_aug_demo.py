import tempfile

import click
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from luminoth.utils.mosaic import assemble_mosaic
from luminoth.utils.overlay_bbs import overlay_bb_labels
from luminoth.utils.image import (
    resize_image, flip_image, random_patch, random_resize, random_distortion,
    patch_image, rot90, random_patch_gaussian, equalize_histogram
)


def get_data_aug_images(image, bboxes):
    # Convert to tensorflow
    # Open a session, run all the data augmentation
    # Get the numpy arrays within the session, overlay_bb_labels
    # and then save the images in tempfolder with delete=False
    # return the list of images within the session to mosaic_data_aug
    augmented_images = []
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    rotate_dict = sess.run(rot90(image, bboxes=bboxes))
    resize_dict = sess.run(resize_image(image, bboxes=bboxes))
    random_patch_dict = sess.run(random_patch(image, bboxes=bboxes))
    random_resize_dict = sess.run(random_resize(image, bboxes=bboxes))
    random_distortion_dict = sess.run(random_distortion(image, bboxes=bboxes))
    patch_dict = sess.run(patch_image(image, bboxes=bboxes))
    flip_dict = sess.run(flip_image(image, bboxes=bboxes))
    gaussian_dict = sess.run(random_patch_gaussian(image, bboxes=bboxes))
    equalize_dict = sess.run(equalize_histogram(image, bboxes=bboxes))

    return augmented_images


def mosaic_data_aug(input_png, csv_path, tile_size, fill_value, output_png):
    """
    Mosaic data augmented images after resizing them to tile_size.
    all the pixels that don't fit in the stitched image shape
    by the resized tiles are filled with fill_value

    Args:
        input_png: str Input png to augment
        csv_path: csv containing image_id,xmin,xmax,ymin,ymax,label
            for the input_png. Bounding boxes in the csv are augmented
        tile_size: tuple that each image in the mosaic is resized to
        fill_value: fill the tiles that couldn't be filled with the images
        output_png: write the stitched mosaic image to
        fmt: format of input images in im_dir

    Returns:
        Write stitched image of shape
        tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(
        len(images_in_path)).
    """
    image = cv2.imread(input_png, cv2.IMREAD_GRAYSCALE)
    df = pd.read_csv(csv_path)
    bboxes = []
    for index, row in df.iterrows():
        bboxes.append([
            row['xmin'],
            row['xmax'],
            row['ymin'],
            row['ymax'],
            row['label']])
    augmented_images = get_data_aug_images(image, bboxes)
    mosaiced_image = assemble_mosaic(
        augmented_images, tile_size, fill_value)
    shape = mosaiced_image.shape
    cv2.imwrite(output_png, mosaiced_image)

    print("Mosaiced image is at: {} of shape {}".format(output_png, shape))


@click.command(help="Save one assembled mosaic filled with data augmented images for png")  # noqa
@click.option("--input_png", help="Input png to augment", required=True, type=str) # noqa
@click.option("--csv_path", help="Csv containing image_id,xmin,xmax,ymin,ymax,label.Bounding boxes in the input png to augment", required=True, type=str) # noqa
@click.option("--tile_size", help="[x,y] list of tile size in x, y", required=False, multiple=True) # noqa
@click.option("--fill_value", help="fill the image with zeros or the first intensity at [0,0] in the image", required=False, type=str) # noqa
@click.option("--output_png", help="Absolute path to folder name to save the data aug mosaiced images to", required=True, type=str) # noqa
def data_aug_demp(input_png, tile_size, fill_value, output_png):
    mosaic_data_aug(input_png, tile_size, fill_value, output_png)


if __name__ == '__main__':
    data_aug_demp()
