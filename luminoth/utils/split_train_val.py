# -*- coding: utf-8 -*-

import glob
import itertools
import math
import natsort
import random
import os

import numpy as np
from skimage import io
import pandas as pd
import click

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

"""
LUMI_CSV_COLUMNS = ['image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
# TODO make the columns match
INPUT_CSV_COLUMNS = ['image_path', 'x1', 'x2', 'y1', 'y2', 'class_name']

OUTPUT_IMAGE_FORMAT = ".jpg"


def add_basename_gather_df(filenames, input_image_format):
    """
    Writes to output_dir two folders train, val images and two csv files,
    train.csv, val.csv

    Args:
        filenames: List of paths to comma separated txt/csv file with
            image_path,x1,y1,x2,y2,class_name
        input_image_format: remove this format tag from the basename of the
            image_path while adding base_path column

    Returns:
        bb_labels_df: Returns a dataframe with all the bounding boxes & labels
            from all the paths in filenames. Also adds a column base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be img
    """
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename))
    bb_labels_df = pd.concat(dfs, ignore_index=True)
    base_names = [
        os.path.basename(row["image_path"]).replace(
            input_image_format, "") for index, row in bb_labels_df.iterrows()]
    bb_labels_df["base_path"] = pd.Series(base_names)
    label_name = bb_labels_df['class_name'].iloc[0]
    # Cast required columns to integers
    if type(label_name) is str:
        cols = ['x1', 'x2', 'y1', 'y2']
        bb_labels_df[cols] = bb_labels_df[cols].applymap(np.int64)
    elif type(label_name) is np.float64:
        cols = ['x1', 'x2', 'y1', 'y2', 'class_name']
        bb_labels_df[cols] = bb_labels_df[cols].applymap(np.int64)
    bb_labels_df.reset_index(drop=True, inplace=True)
    return bb_labels_df


def get_image_paths_per_class(bb_labels_df):
    """
    Returns dict containing class name as the key and the
    list of image paths containing the class annotation as values

    Args:
        bb_labels: Returns a dataframe with all the bounding boxes and labels
            from all the paths in filenames. Also adds a column base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be img

    Returns:
        image_paths_per_class: dict containing class name as the key and the
        list of image paths containing the class annotation as values
    """
    # Print meta for each unique class
    class_labels = np.unique(bb_labels_df.class_name)
    image_paths_per_class = {}
    for class_name in class_labels:
        # This dict collects all the unique images of a class
        filtered_df = bb_labels_df[bb_labels_df['class_name'] == class_name]
        images = np.unique(filtered_df['image_path']).tolist()
        image_paths_per_class[class_name] = images
        print(
            'There are {} images with {} {} labeled classes in dataset'.format(
                len(image_paths_per_class[class_name]),
                len(filtered_df),
                class_name))
    return image_paths_per_class


def get_lumi_csv_df(bb_labels, images, output_image_format):
    """
    Filters out the list of images given from bb_labels and
    formats it to a csv dataformat required by luminoth

    Args:
        bb_labels: Dataframe with image_path,x1,y1,x2,y2,class_name
        images: List of images to filter by
        output_image_format: Defaults to jpg

    Returns:
        df: pandas.DataFrame Filters out the list of images given from bb_label
            and formats it to a csv dataformat required by luminoth
    """
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    # Find boxes in each image and put them in a dataframe
    for img_name in images:
        # Filter out the df for all the bounding boxes in one image
        basename = os.path.basename(img_name).replace(output_image_format, "")
        tmp_df = bb_labels[bb_labels.base_path == basename]
        # Add all the bounding boxes for the images to the dataframe
        for index, row in tmp_df.iterrows():
            label_name = row['class_name']
            df = df.append({'image_id': img_name,
                            'xmin': np.int64(row['x1']),
                            'xmax': np.int64(row['x2']),
                            'ymin': np.int64(row['y1']),
                            'ymax': np.int64(row['y2']),
                            'label': label_name},
                           ignore_index=True)

    # Cast required columns to integers
    label_name_type = type(df['label'].iloc[0])
    if label_name_type is str:
        cols = ['xmin', 'xmax', 'ymin', 'ymax']
        df[cols] = df[cols].applymap(np.int64)
    elif label_name_type is float:
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
    Copies images to path with a output_image_format
    and writes a csv containing LUMI_CSV_COLUMNS
    listing out the images and annotations for the same images

    Args:
        images: list of images to copy to path
        path: Directory to output the scaled uint8 jpg files or files
            of output_image_format to
        input_image_format: raw input image data format
        output_image_format: output image format,
            lumi currently accepts only jpg for inputs, so the train and
            val folders created would be saving input images as uint8 jpgs to
            the output_dir. In that case the output_image_format would be .jpg
        bb_labels: Returns a dataframe with all the bounding boxes and labels
            from all the paths in filenames. Also adds a column base_path which
            is just the basename of a fullpath, e.g /data/bla/img.tif's
            base_path would be img

    Returns:
        Writes images to path and writes a csv file containing LUMI_CSV_COLUMNS
        to output_csv_path
    """
    # Save each classes' images to given path as output_image_format
    os.makedirs(path, exist_ok=True)
    for index, original_path in enumerate(images):
        new_path = os.path.join(path, os.path.basename(original_path))
        new_path = new_path.replace(input_image_format, output_image_format)
        image = io.imread(original_path)
        image = (image / image.max() * 255).astype(np.uint8)
        io.imsave(new_path, io.imread(original_path))
    images = natsort.natsorted(
        glob.glob(os.path.join(path, "*" + output_image_format)))
    print(
        'number of images in path {} : {}'.format(
            path, len(images)))
    df = get_lumi_csv_df(bb_labels, images, output_image_format)
    df.to_csv(output_csv_path, index=False)


def filter_dense_annotation(image_paths_per_class):
    """
    Returns dict containing class name as the key and the
    list of image paths containing the class annotation as values
    after deleting the key:value pair, for the label that contains
    the highest number of annotations

    Args:
        image_paths_per_class: dict containing class name as the key and the
        list of image paths containing the class annotation as values

    Returns:
        image_paths_per_class: dict containing class name as the key and the
        list of image paths containing the class annotation as values
        after suppressing the dense annotation of a class
    """
    max_class_count = max(
        [len(value) for key, value in image_paths_per_class.items()])
    max_class_count_name = [
        key for key, value in image_paths_per_class.items() if len(
            value) == max_class_count][0]
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
        filenames: List of paths to comma separated txt file with
            image_path,x1,y1,x2,y2,class_name
        output_dir: Full path to save the train,
            val folders and their csv files
        random_seed: Randomize the images so no
            two continuous slices go to the train or validation directory
        percentage: Percentage of data for training
            and 1 - percentage images are copied to validation directory
        filter_dense_anns: It is assumed that an image
            consists of one densely annotated class and sparsely
            annotated other class labels.
            So, images with just dense classes are filtered out if this flag
            is set to True
        input_image_format: raw input image data format
        output_dir: Directory to output the scaled uint8 jpg files or files
            of output_image_format to
        output_image_format: lumi output image format,
            lumi currently accepts only jpg for inputs, so the train and
            val folders created would be saving input images as uint8 jpgs to
            the output_dir
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

    all_imgs = np.unique(
        list(
            itertools.chain.from_iterable(
                [
                    value for key, value in image_paths_per_class.items()
                ]))).tolist()
    all_imgs_length = len(all_imgs)
    random.shuffle(all_imgs)

    training_image_index = math.floor(percentage * all_imgs_length)
    write_lumi_images_csv(
        all_imgs[:training_image_index],
        os.path.join(output_dir, "train"),
        input_image_format,
        output_image_format,
        bb_labels,
        os.path.join(output_dir, 'train.csv'))
    write_lumi_images_csv(
        all_imgs[training_image_index:],
        os.path.join(output_dir, "val"),
        input_image_format,
        output_image_format,
        bb_labels,
        os.path.join(output_dir, 'val.csv'))


@click.command(help="Split and arrange images into 2 folders for grayscale jpgs for train, and validation, save bounding boxes and labels for the corresponding images in train.csv and val.csv")  # noqa
@click.argument("filenames", nargs=-1) # noqa
@click.option("--percentage", help="Percentage of images to split into training folder, rest of the images are equally divided to validation", required=False, type=float, default=0.9) # noqa
@click.option("--random_seed", help="Random seed to split data into training, validation images", required=False, type=int, default=43) # noqa
@click.option('--filter_dense_anns', help="Filter out images with only the dense class annotations", required=False)  # noqa
@click.option('--input_image_format', help="output image data format", required=False, type=str, default=".jpg")  # noqa
@click.option('--output_dir', help="Absolute path to folder containing train, validation scaled uint8 jpg images and their annotations in csv file", required=True, type=str)  # noqa
def split_train_val(
        filenames,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir):

    assert percentage < 1.0
    split_data_to_train_val(
        filenames,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir,
        OUTPUT_IMAGE_FORMAT)

    print("Formatted lumi csv folder at {}".format(output_dir))


if __name__ == '__main__':
    split_train_val()
