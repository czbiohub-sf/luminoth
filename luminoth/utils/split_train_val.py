import itertools
import glob
import math
import natsort
import random
import os

import numpy as np
from skimage import io
import pandas as pd
import click

"""
Split data in bb_labels .txt ir .csv files to train
and val categories data and create csvs containing
their metadata as follows::
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
OUTPUT_IMAGE_FORMAT = ".jpg"


def get_lumi_csv_df(bb_labels, images, output_image_format):
    """
    Filters out the list of images given from bb_labels and
    formats it to a csv dataformat required by luminoth

    Args:
        bb_labels: Dataframe with image_path,x1,y1,x2,y2,class_name
        images: List of images to filter by

    Returns:
        pandas.DataFrame Filters out the list of images given from bb_labels
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
                            'xmin': int(row['x1']),
                            'xmax': int(row['x2']),
                            'ymin': int(row['y1']),
                            'ymax': int(row['y2']),
                            'label': label_name},
                           ignore_index=True)
    if type(label_name) is str:
        cols = ['xmin', 'xmax', 'ymin', 'ymax']
    else:
        cols = ['xmin', 'xmax', 'ymin', 'ymax', 'label']
    df[cols] = df[cols].applymap(np.int64)
    return df


def split_data_to_train_val(
        filenames,
        class_labels,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir,
        output_image_format):
    """
    Writes to output_dir two folders train, val and two csv files,
    train.csv, val.csv

    Args:
        filenames: Full path to comma separated txt file with
            image_path,x1,y1,x2,y2,class_name
        class_labels: list of classes
        output_dir: Full path to save the train,
            val folders and their csv files
        random_seed: Randomize the images so no
            two continuos slices go to the train or validation directory
        percentage: Percentage of data for training
            and 1- percentage images are copied to validation directory
        filter_dense_anns: It is assumed that an image
            consists of one densely annotated class and sparsely
            annotated other class labels.
            So, images with just dense classes are filtered out
        input_image_format: raw image data format
        output_dir: Directory to output the scaled uint8 jpg files, data
        output_image_format: lumi output image format,
        lumi currently accepts only jpg
    """
    random.seed(random_seed)
    # Add base_path column to filter into training and validation images later
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filenames))
    bb_labels = pd.concate(dfs, ignore_index=True)
    print(filter_dense_anns)
    base_names = [
        os.path.basename(row.image_path).replace(
            input_image_format, "") for index, row in bb_labels.iterrows()]
    bb_labels["base_path"] = pd.Series(base_names)

    # Print meta for each class in class_labels
    image_ids_classes = {}
    for class_name in class_labels:

        # This dict collects all the unique images of a class
        filtered_df = bb_labels[bb_labels['class_name'] == class_name]
        print('There are {} {} classes in the dataset'.format(
            len(filtered_df), class_name))
        image_ids_classes[class_name] = np.unique(filtered_df['image_path'])
        print('There are {} images with {} classes in the dataset'.format(
            len(image_ids_classes[class_name]), class_name))
    # Balancing class count by
    # taking into account only images with classes that are sparsely present
    # these images are
    # expected to contain the other class that has lots of annotations
    if filter_dense_anns:
        max_class_count = max(
            [len(value) for key, value in image_ids_classes.items()])
        max_class_count_name = [
            key for key, value in image_ids_classes.items() if len(
                value) == max_class_count][0]
        image_ids_classes.pop(max_class_count_name)

    # Get unique images that contains desired classes
    all_imgs = np.unique(list(itertools.chain.from_iterable(
        [value for key, value in image_ids_classes.items()]))).tolist()
    all_imgs_length = len(all_imgs)
    all_imgs = natsort.natsorted(all_imgs)
    print("all images {}". format(all_imgs_length))
    random.shuffle(all_imgs)

    # Prepare dataset format for faster rcnn code
    # (fname_path, xmin, xmax, ymin, ymax, class_name)
    # train: 0.9
    # validation: 0.1

    # Save images to train and val directory
    train_path = os.path.join(output_dir, 'train')
    os.makedirs(train_path, exist_ok=True)
    val_path = os.path.join(output_dir, 'val')
    os.makedirs(val_path, exist_ok=True)
    training_image_index = math.floor(percentage * all_imgs_length)
    print("training image_index: ", training_image_index)
    train_imgs = np.unique(all_imgs[0:training_image_index]).tolist()
    print("len(train_imgs)", len(train_imgs))
    val_imgs = np.unique(all_imgs[training_image_index:]).tolist()
    print("len(val_imgs)", len(val_imgs))
    print("all_imgs_length", all_imgs_length)
    # Save each classes' images to train directory as jpgs
    print(train_path)
    for index, original_path in enumerate(train_imgs):
        print(index)
        new_path = os.path.join(train_path, os.path.basename(original_path))
        new_path = new_path.replace(input_image_format, output_image_format)
        image = io.imread(original_path)
        image = (image / image.max() * 255).astype(np.uint8)
        io.imsave(new_path, io.imread(original_path))
    print(val_path)
    # Save each classes' images to val directory as jpgs
    for index, original_path in enumerate(val_imgs):
        print(index)
        new_path = os.path.join(val_path, os.path.basename(original_path))
        new_path = new_path.replace(input_image_format, output_image_format)
        image = io.imread(original_path)
        image = (image / image.max() * 255).astype(np.uint8)
        io.imsave(new_path, io.imread(original_path))
    print(output_image_format)
    train_images = natsort.natsorted(
        glob.glob(os.path.join(train_path, "*" + output_image_format)))
    val_images = natsort.natsorted(
        glob.glob(os.path.join(val_path, "*" + output_image_format)))
    print('number of training images: ', len(train_images))
    print('number of validation images: ', len(val_images))
    # create metadata dataframes for training and validation images
    train_df = get_lumi_csv_df(bb_labels, train_images, output_image_format)
    val_df = get_lumi_csv_df(bb_labels, val_images, output_image_format)

    # Save the csvs
    train_df.to_csv(os.path.join(output_dir, 'train.csv'))
    val_df.to_csv(os.path.join(output_dir, 'val.csv'))




@click.command(help="Split and arrange images into 2 folders for grayscale jpgs for train, and validation, save bounding boxes and labels for the corresponding images in train.csv and val.csv")  # noqa
@click.argument("filenames", nargs=-1) # noqa
@click.option("--class_labels", help="List of labels for classes", multiple=True, required=True) # noqa
@click.option("--percentage", help="Percentage of images to split into training folder, rest of the images are equally divided to validation", required=False, type=float, default=0.9) # noqa
@click.option("--random_seed", help="Random seed to split data into training, validation images", required=False, type=int, default=43) # noqa
@click.option('--filter_dense_anns', help="Filter out images with only the dense class annotations", required=False)  # noqa
@click.option('--input_image_format', help="output image data format", required=False, type=str, default=".jpg")  # noqa
@click.option('--output_dir', help="Absolute path to folder containing train, validation scaled uint8 jpg images and their annotations in csv file", required=True, type=str)  # noqa
def split_train_val(
        filenames,
        class_labels,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir):

    assert percentage < 1.0
    split_data_to_train_val(
        filenames,
        class_labels,
        percentage,
        random_seed,
        filter_dense_anns,
        input_image_format,
        output_dir,
        OUTPUT_IMAGE_FORMAT)

    print("Formatted lumi csv folder at {}".format(output_dir))


if __name__ == '__main__':
    split_train_val()
