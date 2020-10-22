# -*- coding: utf-8 -*-
# Added above statement to avoid encoding errors resulting from line 18
import glob
import math
import os
import random

import cv2
import click
import natsort
import numpy as np
import pandas as pd

"""
Split data in bb_labels .txt or .csv files to a lumi CSV dataset with
images,csv directory structure as follows::

        .
        ├── train
        │   ├── image_1.jpg
        │   ├── image_2.jpg
        │   └── image_3.jpg
        ├── val
        │   ├── image_4.jpg
        │   ├── image_5.jpg
        │   └── image_6.jpg
        ├── train.csv
        └── val.csv

    The CSV file itself must have the following format::

        image_id,xmin,ymin,xmax,ymax,label
        image_1.jpg,26,594,86,617,cat
        image_1.jpg,599,528,612,541,car
        image_2.jpg,393,477,430,552,dog

NB: All functions in this module are very specific to collecting and setting
train, val data from
different folders containing similarly named images. For example
sample_1/ch1_p1_t1.png, sample_2/ch1_p1_t1.png, sample_3/ch1_p1_t1.png
So, please do not import any functions from here to another module
unless you explicitly know the behaviour and need it
"""

# Constants for dataframe lumi csv headers
LUMI_CSV_COLUMNS = ['image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
# Constant for luminoth accepted images for training/validation
OUTPUT_IMAGE_FORMAT = ".jpg"


def add_basename_gather_df(filenames, input_image_format):
    """
    Returns a dataframe with all the bounding boxes & labels
    from all the paths in filenames. Also adds a column base_path which
    is just the base_path of a fullpath, e.g /data/bla/img.tif's
    base_path would be _data_bla_img

    Args:
        filenames: list of paths to comma separated txt/csv file with
            image_id, xmin, xmax, ymin, ymax, label
        input_image_format: str image format in the path

    Returns:
        bb_labels_df: pandas.DataFrame with all the bounding boxes & labels
            from all the paths in filenames. Also adds a
            column base_path which
            is just the base_path of a fullpath, e.g /data/bla/img.tif's
            base_path would be _data_bla_img
    """
    # Collect all the dataframes from list of all filenames
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename))
    bb_labels_df = pd.concat(dfs, ignore_index=True)

    # Add base_path columns
    base_names = [
        os.path.dirname(row["image_id"]).replace(os.sep, "_") + "_" +
        os.path.basename(row["image_id"]).replace(input_image_format, "")
        for index, row in bb_labels_df.iterrows()]
    bb_labels_df["base_path"] = pd.Series(base_names)

    label_name_type = type(bb_labels_df['label'].iloc[0])

    # Cast required columns to integers
    if label_name_type is str:
        cols = ['xmin', 'xmax', 'ymin', 'ymax']
        bb_labels_df[cols] = bb_labels_df[cols].applymap(np.int64)
    elif label_name_type is np.float64:
        cols = ['xmin', 'xmax', 'ymin', 'ymax', 'label']
        bb_labels_df[cols] = bb_labels_df[cols].applymap(np.int64)

    bb_labels_df.reset_index(drop=True, inplace=True)

    return bb_labels_df


def get_image_paths_per_class(bb_labels_df):
    """
    Returns dict containing label as the key and the
    list of image paths containing the label annotation as values

    Args:
        bb_labels: pandas.DataFrame with all the bounding boxes and labels
            from all the paths

    Returns:
        image_paths_per_class: dict containing label as the key and the
            list of image paths containing the label annotation as values
    """
    # Print meta for each unique label
    class_labels = np.unique(bb_labels_df['label'])
    image_paths_per_class = {}
    for label in class_labels:
        # This dict collects all the unique images that contains a label
        filtered_df = bb_labels_df[bb_labels_df['label'] == label]
        images = np.unique(filtered_df['image_id']).tolist()
        image_paths_per_class[label] = images
        print(
            'There are {} images with {} {} labeled classes in dataset'.format(
                len(image_paths_per_class[label]),
                len(filtered_df),
                label))
    return image_paths_per_class


def get_lumi_csv_df(bb_labels, images, output_image_format):
    """
    Return csv as required by luminoth with image_id,xmin,xmax,ymin,ymax,label

    Args:
        bb_labels: Dataframe with image_id, xmin, xmax, ymin, ymax, label
        images: List of images to filter by
        output_image_format: Defaults to jpg

    Returns:
        df: pandas.DataFrame Filters out the list of images given from bb_label
            and formats it to a csv dataformat required by luminoth
    """
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    label_name = ""

    # Find boxes in each image and put them in a dataframe
    for img_name in images:
        # Filter out the df for all the bounding boxes in one image
        basename = os.path.basename(img_name).replace(output_image_format, "")
        tmp_df = bb_labels[bb_labels.base_path == basename]

        # Add all the bounding boxes for the images to the dataframe
        count = 0
        for index, row in tmp_df.iterrows():
            label_name = row['label']

            if count == 0:
                label_name_type = type(label_name)
            assert label_name_type == type(label_name)
            if label_name_type is float or label_name_type is int:
                label_name = np.int64(label_name)

            df = df.append({'image_id': img_name,
                            'xmin': np.int64(row['xmin']),
                            'xmax': np.int64(row['xmax']),
                            'ymin': np.int64(row['ymin']),
                            'ymax': np.int64(row['ymax']),
                            'label': label_name},
                           ignore_index=True)
            count += 1

    if type(label_name) is str:
        cols = ['xmin', 'xmax', 'ymin', 'ymax']
    else:
        cols = ['xmin', 'xmax', 'ymin', 'ymax', 'label']

    df[cols] = df[cols].applymap(np.int64)
    df.reset_index(drop=True, inplace=True)
    return df


def write_lumi_images_csv(
        images,
        path,
        input_image_format,
        output_image_format,
        bb_labels,
        output_csv_path):
    """
    Write images to 'path' with 'output_image_format'
    and writes a csv containing LUMI_CSV_COLUMNS
    listing out the images and annotations for the same images

    Args:
        images: list of images to copy to path
        path: str Directory to output the scaled uint8 file to
        input_image_format: str input image data format
        output_image_format: str output image format
        bb_labels: pandas.DataFrame with all the bounding boxes and labels
            from all the paths in filenames. Also adds a column
            base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be _data_bla_img

    Returns:
        Writes images to path and writes a csv file containing LUMI_CSV_COLUMNS
        to output_csv_path
    """
    # Create a folder to save images to
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("Path {} already exists, might be overwriting data".format(path))

    # Iterate through images, rename them to include whole absolute path in
    # base_path. For example original image /data/bar/image.tif would be
    # saved in path + _data_bar_image.jpg. This is to prevent images with
    # similar basenames wouldn't be clobbered. Example if there is another
    # image at /data/foo/image.tif
    for index, original_path in enumerate(images):

        base_path = os.path.dirname(original_path).replace(os.sep, "_")
        basename = os.path.basename(original_path).replace(
            input_image_format, output_image_format)
        basename = base_path + "_" + basename
        new_path = os.path.join(path, basename)

        image = cv2.imread(
            original_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        cv2.imwrite(new_path, image)

    images = natsort.natsorted(
        glob.glob(os.path.join(path, "*" + output_image_format)))
    print(
        'number of images in path {} : {}'.format(
            path, len(images)))

    # Write to csv
    df = get_lumi_csv_df(bb_labels, images, output_image_format)
    df.to_csv(output_csv_path, index=False)


def filter_dense_annotation(image_paths_per_class):
    """
    Returns dict containing label as the key and the
    list of image paths containing the class annotation as values
    after deleting the key:value pair, for the label that contains
    the highest number of annotations

    Args:
        image_paths_per_class: dict containing label as the key and the
        list of image paths containing the class annotation as values

    Returns:
        image_paths_per_class: dict containing label as the key and the
        list of image paths containing the class annotation as values
        after removing the key,value pair with dense annotation class
    """
    # Get highest number of classes/labels/annotations in images
    max_class_count = max(
        [len(value) for key, value in image_paths_per_class.items()])
    max_class_count_name = [
        key for key, value in image_paths_per_class.items() if len(
            value) == max_class_count][0]

    # Remove the dense annotation class
    image_paths_per_class.pop(max_class_count_name)
    return image_paths_per_class


def split_data_to_train_val(
        filenames,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir,
        output_image_format):
    """
    Writes to output_dir two folders train, val images and two csv files,
    train.csv, val.csv

    Args:
        filenames: list of paths to comma separated txt file with
             image_id, xmin, xmax, ymin, ymax, label
        random_seed: int Randomize the images so no
            two continuous slices go to the train or validation directory
        percentage: float Percentage of data for training
            and 1 - percentage images are copied to validation directory
        filter_dense_anns: bool If a dataset
            consists of one densely annotated class and sparsely
            annotated other class labels.
            So, images with just dense classes are filtered out if this flag
            is set to True
        input_image_format: str input image data format
        output_dir: str Full path to save the train,val folders and
            their csv files
        output_image_format: str output image format,
    """
    random.seed(random_seed)
    bb_labels = add_basename_gather_df(filenames, input_image_format)

    # Get unique images per classes
    image_paths_per_class = get_image_paths_per_class(bb_labels)

    # Balancing class count by taking into account only images with
    # classes that are sparsely present these images are
    # expected to contain the other class that has lots of annotations
    if filter_dense_anns:
        image_paths_per_class = filter_dense_annotation(image_paths_per_class)

    all_imgs = []
    for class_name, images in image_paths_per_class.items():
        for im_path in images:
            all_imgs.append(im_path)

    # Get all the unique images and randomize them
    all_imgs = np.unique(all_imgs).tolist()
    all_imgs_length = len(all_imgs)
    print("total unique images are {}".format(all_imgs_length))
    random.shuffle(all_imgs)

    # Get unique images until training_image_index to train and rest to val
    training_image_index = int(math.floor(percentage * all_imgs_length))
    write_lumi_images_csv(
        all_imgs[0:training_image_index],
        os.path.join(output_dir, "train"),
        input_image_format,
        output_image_format,
        bb_labels,
        os.path.join(output_dir, 'train.csv'))

    write_lumi_images_csv(
        all_imgs[training_image_index:all_imgs_length],
        os.path.join(output_dir, "val"),
        input_image_format,
        output_image_format,
        bb_labels,
        os.path.join(output_dir, 'val.csv'))


@click.command(help="Split and arrange images into 2 folders for grayscale jpgs for train, and validation, save bounding boxes and labels for the corresponding images in train.csv and val.csv")  # noqa
@click.argument("filenames", nargs=-1) # noqa
@click.option("--percentage", help="Percentage of images to split into training folder, rest of the images are saved to validation, default 0.8", required=False, type=float, default=0.8) # noqa
@click.option("--random_seed", help="Random seed to split data into training, validation images, default 43", required=False, type=int, default=43) # noqa
@click.option('--filter_dense_anns', help="Filter out images with only the dense class annotations, default they are not filtered", required=False, default=False)  # noqa
@click.option('--input_image_format', help="input image data format", required=True, type=str)  # noqa
@click.option('--output_dir', help="Absolute path to folder containing train, validation scaled uint8 jpg images and their annotations in csv file", required=True, type=str)  # noqa
def split_train_val(
        filenames,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir):

    print("Note: If giving multiple filenames, the output directory " +
          "will contain images with long names to prevent the " +
          "overwriting image of same name from a different sample")

    assert percentage < 1.0
    split_data_to_train_val(
        filenames,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir,
        OUTPUT_IMAGE_FORMAT)

    print("Formatted lumi csv folder at: {}".format(output_dir))


if __name__ == '__main__':
    split_train_val()
