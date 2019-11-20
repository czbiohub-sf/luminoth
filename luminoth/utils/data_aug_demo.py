import os
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
from luminoth.utils.split_train_val import LUMI_CSV_COLUMNS

TILE_SIZE = [256, 256]
FILL_VALUE = 128


def update_augmentation(
        augmented_dict, labels, location, augmentation, augmented_images):

    base_path = "input_{}_image.png".format(augmentation)
    im_filename = os.path.join(location, base_path)
    cv2.imwrite(im_filename, augmented_dict['image'])
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS + ['base_path'])

    for bboxes in augmented_dict['bboxes']:
        df = df.append(
            {'xmin': bboxes[0],
             'xmax': bboxes[1],
             'ymin': bboxes[2],
             'ymax': bboxes[3],
             'label': labels[bboxes[4]]}, ignore_index=True)
    df.base_path = base_path
    df.image_path = im_filename

    overlaid_rotated_image = overlay_bb_labels(im_filename, ".png", df)
    im_filename = os.path.join(location, augmentation + "bb_labels.png")
    cv2.imwrite(im_filename, overlaid_rotated_image)
    augmented_images.append(im_filename)


def get_data_aug_images(image_array, bboxes_array, labels):
    # Convert to tensorflow
    # Open a session, run all the data augmentation
    # Get the numpy arrays within the session, overlay_bb_labels
    # and then save the images in tempfolder with delete=False
    # return the list of images within the session to mosaic_data_aug
    augmented_images = []
    location = tempfile.mkdtemp()
    image = tf.placeholder(tf.float32, image_array.shape)
    feed_dict = {
        image: image_array,
    }
    if bboxes_array is not None:
        bboxes = tf.placeholder(tf.int32, bboxes_array.shape)
        feed_dict[bboxes] = bboxes_array
    else:
        bboxes = None

    with tf.Session() as sess:
        rotated = rot90(image, bboxes=bboxes)
        rotate_dict = sess.run(rotated, feed_dict=feed_dict)
        update_augmentation(
            rotate_dict, labels, location, "rotation", augmented_images)

        resized = resize_image(image, bboxes=bboxes)
        resize_dict = sess.run(resized, feed_dict=feed_dict)
        update_augmentation(
            resize_dict, labels, location, "resize", augmented_images)

        rand_patch = random_patch(image, bboxes=bboxes)
        random_patch_dict = sess.run(rand_patch, feed_dict=feed_dict)
        update_augmentation(
            random_patch_dict, labels,
            location, "random_patch", augmented_images)

        rand_resize = random_resize(image, bboxes=bboxes)
        random_resize_dict = sess.run(rand_resize, feed_dict=feed_dict)
        update_augmentation(
            random_resize_dict, labels,
            location, "random_resize", augmented_images)

        rand_distortion = random_distortion(image, bboxes=bboxes)
        random_distortion_dict = sess.run(rand_distortion, feed_dict=feed_dict)
        update_augmentation(
            random_distortion_dict, labels,
            location, "random_distortion", augmented_images)

        patch = patch_image(image, bboxes=bboxes)
        patch_dict = sess.run(patch, feed_dict=feed_dict)
        update_augmentation(
            patch_dict, labels, location, "patch", augmented_images)

        flip = flip_image(image, bboxes=bboxes)
        flip_dict = sess.run(flip, feed_dict=feed_dict)
        update_augmentation(
            flip_dict, labels, location, "flip", augmented_images)

        gaussian = random_patch_gaussian(image, bboxes=bboxes)
        gaussian_dict = sess.run(gaussian, feed_dict=feed_dict)
        update_augmentation(
            gaussian_dict, labels,
            location, "random_patch_gaussian", augmented_images)

        equalized = equalize_histogram(image, bboxes=bboxes)
        equalize_dict = sess.run(equalized, feed_dict=feed_dict)
        update_augmentation(
            equalize_dict, labels, location, "equalize", augmented_images)

    return augmented_images


def mosaic_data_aug(
        input_image, input_image_format,
        csv_path, image_path_column,
        tile_size, fill_value, output_png):
    """
    Mosaic data augmented images after resizing them to tile_size.
    all the pixels that don't fit in the stitched image shape
    by the resized tiles are filled with fill_value

    Args:
        input_image: str Input png to augment
        input_image_format: str Format of the input images
        csv_path: csv containing image_id,xmin,xmax,ymin,ymax,label
            for the input_png. Bounding boxes in the csv are augmented
        image_path_column: str name of the image_path_column
        tile_size: tuple that each image in the mosaic is resized to
        fill_value: fill the tiles that couldn't be filled with the images
        output_png: write the stitched mosaic image to
        fmt: format of input images in im_dir

    Returns:
        Write stitched image of shape
        tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(
        len(images_in_path)).
    """
    image = cv2.imread(input_image, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    df = pd.read_csv(csv_path)
    basename = os.path.basename(input_image).replace(input_image_format, "")
    tmp_df = df[
        [True if os.path.basename(row[image_path_column]).replace(
            input_image_format, "").endswith(
            basename) else False for index, row in df.iterrows()]]
    bboxes = []
    labels = df['label'].unique().tolist()
    for index, row in tmp_df.iterrows():
        bboxes.append([
            np.int32(row['xmin']),
            np.int32(row['xmax']),
            np.int32(row['ymin']),
            np.int32(row['ymax']),
            np.int32(labels.index(row['label']))])
    bboxes = np.array(bboxes, dtype=np.int32)
    augmented_images = get_data_aug_images(image, bboxes, labels)
    mosaiced_image = assemble_mosaic(
        augmented_images, tile_size, int(fill_value))
    shape = mosaiced_image.shape
    cv2.imwrite(output_png, mosaiced_image)

    print("Mosaiced image is at: {} of shape {}".format(output_png, shape))


@click.command(help="Save one assembled mosaic filled with data augmented images for png")  # noqa
@click.option("--input_image", help="Input data to augment", required=True, type=str) # noqa
@click.option("--input_image_format", help="Input image format", required=True, type=str) # noqa
@click.option("--csv_path", help="Csv containing image_id,xmin,xmax,ymin,ymax,label.Bounding boxes in the input png to augment", required=True, type=str) # noqa
@click.option("--image_path_column", help="Name of the image path column, it is often image_id for lumi output, or could be image_path if saved outside of lumi", required=True, type=str) # noqa
@click.option("--tile_size", help="[x,y] list of tile size in x, y", required=False, multiple=True, default=TILE_SIZE) # noqa
@click.option("--fill_value", help="fill the tiles in mosaic image that are not filled by the small tiles with this value", required=False, type=int, default=FILL_VALUE) # noqa
@click.option("--output_png", help="Absolute path to folder name to save the data aug mosaiced images to", required=True, type=str) # noqa
def data_aug_demo(
        input_image, input_image_format, csv_path,
        image_path_column, tile_size, fill_value, output_png):
    mosaic_data_aug(
        input_image, input_image_format,
        csv_path, image_path_column,
        tile_size, fill_value, output_png)


if __name__ == '__main__':
    data_aug_demo()
