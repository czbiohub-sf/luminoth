import click
import cv2 as cv
import glob
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt


def assemble_mosaic(images_in_path, tile_size, fill_value=0, display=True):
    """
    Mosaic each image in "images_in_path" after resizing them to tile_size,
    all the pixels that don't fit in by the resized tiles
    are filled with fill_value

    Args:
        images_in_path: str Directory with images to mosaic with
        tile_size: tuple that each image in the mosaic is resized to
        fill_value: int Bounding box line width
        display: boolean display the overlaid roi before returning the array

    Returns:
        Returned stitched image
    """
    x_tiles = y_tiles = math.ceil(np.sqrt(len(images_in_path)))
    if tile_size is not None:
        tile_size_x = tile_size[0]
        tile_size_y = tile_size[1]
    else:
        shape = cv.imread(
            images_in_path[0], cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR).shape
        tile_size_x = shape[0]
        tile_size_y = shape[1]
    mosaiced_im = np.ones(
        (x_tiles * tile_size_x, y_tiles * tile_size_y, 3), dtype=np.uint8)
    mosaiced_im = mosaiced_im * fill_value

    indices = list(itertools.product(range(x_tiles), range(y_tiles)))
    for index, im_path in enumerate(images_in_path):
        im_rgb = cv.imread(im_path, cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)
        if tile_size is not None:
            resized = cv.resize(im_rgb, (tile_size_x, tile_size_y))
        else:
            resized = im_rgb
        x, y = indices[index]
        mosaiced_im[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1),
            :] = resized

    plt.imshow(mosaiced_im)
    plt.axis('off')
    if display:
        plt.show()
    return mosaiced_im

@click.command(help="Save the assembled mosaic from image in a directory")  # noqa
@click.argument("im_dir", nargs=1) # noqa
@click.option("--tile_size", help="[x,y] list of tile size in x, y", required=False, type=int, multiple=True) # noqa
@click.option("--fill_value", help="fill the image with zeros or the first intensity at [0,0] in the image", required=False, type=str) # noqa
@click.option("--output_png", help="Absolute path to folder name to save the roi overlaid images to", required=True) # noqa
@click.option("--fmt", help="Format of images in input directory", required=True, type=str) # noqa
@click.option('--display', help="Display overlaid images, Default False")  # noqa
def mosaic(im_dir, tile_size, fill_value, output_png, fmt, display):

    images_in_path = glob.glob(im_dir + "*." + fmt)
    if fill_value == "first":
        fill_value = cv.imread(images_in_path[0], cv.IMREAD_GRAYSCALE)[0, 0]
    elif fill_value is None:
        fill_value = 128
    else:
        fill_value = int(fill_value)
    im_rgb = assemble_mosaic(
        images_in_path, tile_size, fill_value, display=display)
    cv.imwrite(output_png, im_rgb)


if __name__ == '__main__':
    mosaic()
