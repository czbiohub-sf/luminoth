import glob
import itertools
import os

import click
import cv2
import natsort


def split_mosaic(image, tile_size):
    """
    Returns image after splitting image to multiple tiles of
    of tile_size.

    Args:
        image: str an image array to split
        tile_size: tuple size each image in the split is resized to

    Returns:
        Returns list of images
    """
    # Set number of tiles
    shape = image.shape
    x_tiles = shape[0] // tile_size[0]
    y_tiles = shape[1] // tile_size[1]

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]

    images = []
    # Set each of the tiles with an image in images_in_path
    indices = list(itertools.product(range(x_tiles), range(y_tiles)))
    for index in range(len(indices)):
        x, y = indices[index]
        images.append(image[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1)])
    return images


def _set_tile_size(image, tile_size):
    # Set the tile size
    shape = image.shape
    if tile_size is not None and tile_size != ():
        if type(tile_size[0]) is str:
            tile_size = [int(
                tile_size[0].split(",")[0]), int(tile_size[0].split(",")[1])]
    else:
        tile_size = [shape[0], shape[1]]
    return tile_size


def disassemble_images(input_dir, fmt, tile_size, output_dir):
    """
    Disasemble input_img to tiles of size to tile_size.

    Args:
        input_dir: str Directory containing input images
        fmt: str format of input images
        tile_size: tuple that each disassembled/split image size
        output_dir: str directory to save the disassembled images to

    Returns:
        Save split/disassembled images to
    """
    # Create a folder to save images to
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Path {} already exists, might be overwriting data".format(
            output_dir))

    images = natsort.natsorted(
        glob.glob(os.path.join(input_dir, "*" + fmt)))
    result_images = []
    for input_img in images:
        image = cv2.imread(
            input_img, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        tile_size = _set_tile_size(image, tile_size)
        split_images = split_mosaic(image, tile_size)
        for index, image in enumerate(split_images):
            path = os.path.join(
                output_dir,
                "{}_{}.{}".format(
                    os.path.basename(input_img).split(".")[0],
                    index, jpg))
            cv2.imwrite(path, image)
            result_images.append(path)

    print("Split images are at: {}".format(output_dir))


@click.command(help="Save disassembled mosaic images in a directory")  # noqa
@click.option("--input_dir", help="Directory containing input images", required=True, type=str) # noqa
@click.option("--fmt", help="Format of input images", required=True, type=str) # noqa
@click.option("--tile_size", help="[x,y] list of tile size in x, y", required=False, multiple=True) # noqa
@click.option("--output_dir", help="Absolute path to name to save the disassembled images to", required=True, type=str) # noqa
def disassemble(input_dir, fmt, tile_size, output_dir):
    disassemble_images(input_dir, fmt, tile_size, output_dir)


if __name__ == '__main__':
    disassemble()
