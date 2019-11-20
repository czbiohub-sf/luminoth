import glob
import itertools
import math
import os

import click
import cv2
import natsort
import numpy as np


def assemble_mosaic(images_in_path, tile_size, fill_value):
    """
    Returns image after stitiching image in "images_in_path"
    after resizing them to tile_size. Returned image is of
    shape
    tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(
        len(images_in_path)).
    All the pixels that are not filled in the symmetrical mosaic image
    by the resized tiles are filled with fill_value

    Args:
        images_in_path: str List of images to mosaic
        tile_size: tuple that each image in the mosaic is resized to
        fill_value: fill the tiles that couldn't be filled with the images in
        im_dir for the symmetrical mosaic image.

    Returns:
        Returned stitched image of shape
        tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(
        len(images_in_path)).
    """
    x_tiles = y_tiles = math.ceil(np.sqrt(len(images_in_path)))
    shape = cv2.imread(
        images_in_path[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).shape
    if len(shape) > 2:
        channels = shape[2]
    else:
        channels = 1
    if tile_size is not None:
        tile_size_x = tile_size[0]
        tile_size_y = tile_size[1]
    else:
        tile_size_x = shape[0]
        tile_size_y = shape[1]
    mosaiced_im = np.ones(
        (x_tiles * tile_size_x, y_tiles * tile_size_y, channels),
        dtype=np.uint8)
    mosaiced_im = mosaiced_im * fill_value

    indices = list(itertools.product(range(x_tiles), range(y_tiles)))
    for index, im_path in enumerate(images_in_path):
        image = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if tile_size is not None:
            resized = cv2.resize(image, (tile_size_y, tile_size_x))
        else:
            resized = image
        if channels == 1:
            resized = resized[:, :, np.newaxis]
        x, y = indices[index]
        mosaiced_im[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1),
            :] = resized
    return mosaiced_im


def _set_fill_value(image, fill_value):
    if fill_value == "first":
        fill_value = image[0, 0]
    elif fill_value is None:
        fill_value = 128
    else:
        fill_value = int(fill_value)
    return fill_value


def mosaic_images(im_dir, tile_size, fill_value, output_png, fmt):
    """
    Mosaic images in im_dir after resizing them to tile_size.
    all the pixels that don't fit in the stitched image shape
    by the resized tiles
    are filled with fill_value

    Args:
        im_dir: str Directory with images to mosaic with
        tile_size: tuple that each image in the mosaic is resized to
        fill_value: fill the tiles that couldn't be filled with the images
        output_png: write the stitched mosaic image to
        fmt: format of input images in im_dir

    Returns:
        Write stitched image of shape
        tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(
        len(images_in_path)).
    """
    images_in_path = natsort.natsorted(
        glob.glob(os.path.join(im_dir, "*" + fmt)))
    image = cv2.imread(images_in_path[0], cv2.IMREAD_GRAYSCALE)
    fill_value = _set_fill_value(image, fill_value)
    mosaiced_image = assemble_mosaic(
        images_in_path, tile_size, fill_value)
    shape = mosaiced_image.shape
    cv2.imwrite(output_png, mosaiced_image)

    print("Mosaiced image is at: {} of shape {}".format(output_png, shape))


@click.command(help="Save one assembled mosaic from images in a directory")  # noqa
@click.option("--im_dir", help="Directory containing images to mosaic", required=True, type=str) # noqa
@click.option("--tile_size", help="[x,y] list of tile size in x, y", required=False, multiple=True) # noqa
@click.option("--fill_value", help="fill the image with zeros or the first intensity at [0,0] in the image", required=False, type=str) # noqa
@click.option("--output_png", help="Absolute path to folder name to save the roi overlaid images to", required=True, type=str) # noqa
@click.option("--fmt", help="Format of images in input directory", required=True, type=str) # noqa
def mosaic(im_dir, tile_size, fill_value, output_png, fmt):

    mosaic_images(im_dir, tile_size, fill_value, output_png, fmt)


if __name__ == '__main__':
    mosaic()
