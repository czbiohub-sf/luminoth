import cv2
import os
import numpy as np
import itertools
import pandas as pd
import random

random.seed(42)
"""
Given a image directory, load given position and overlay
with bounding boxes
"""
im_dir = \
    "/Users/pranathivemuri/Downloads/CrossTrain-2019-12-02-15-17-48/{}/RBC Instances/405 nm 40x/sl1"
fmt = "tif"
image_shape = [2048, 2048]
tile_size_x, tile_size_y = 60, 60
x_tiles, y_tiles = image_shape[0] // tile_size_x, image_shape[1] // tile_size_y
indices = list(itertools.product(range(x_tiles), range(y_tiles)))
num_tiles_per_image = x_tiles * y_tiles

df = pd.read_excel(
    "/Users/pranathivemuri/Downloads/CrossTrain-2019-12-02-15-17-48/HumanAnnotated_405 nm 40x.xls")

# Remove 477 healthy rows
count = 0
for index, row in df.iterrows():
    if row["HumanLabels"] == "healthy" and count < 477:
        df = df.drop([index])
        count += 1
# Randomize rows
df = df.sample(frac=1).reset_index(drop=True)

# make a csv with only required columns
LUMI_CSV_COLUMNS = ['image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
output_df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
output_random_df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
count = 0
image_count = 0

kernel = np.ones((5, 5), np.uint8)
for index, row in df.iterrows():
    image = cv2.imread(
        os.path.join(
            im_dir.format(row["datasetID"]), row["Filename"]),
        cv2.IMREAD_GRAYSCALE)
    # save the image
    saved_image_path = os.path.join(
        "/Users/pranathivemuri/Downloads/CrossTrain-2019-12-02-15-17-48/",
        "mosaic",
        "{}.tif".format(image_count))
    saved_random_image_path = os.path.join(
        "/Users/pranathivemuri/Downloads/CrossTrain-2019-12-02-15-17-48/",
        "random_mosaic",
        "{}.tif".format(image_count))

    if count == 0:
        mosaiced_im = np.ones(image_shape, dtype=np.uint8) * 214
        randomized_rgb = np.ones((image_shape), dtype=np.uint8) * 214
        masked_random = np.ones((image_shape), dtype=np.uint8)
        x, y = indices[count]
        image[image == 255] = 214
        mosaiced_im[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1)] = image
        threshold = np.zeros_like(mosaiced_im)
        threshold[mosaiced_im == 214] = 0
        threshold[mosaiced_im != 214] = 255
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        ctrs, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(ctrs) == 1
        x, y, w, h = cv2.boundingRect(ctrs[0])
        # save rois as txt or csv
        output_df = output_df.append(
            {'image_id': saved_image_path,
             'xmin': x,
             'xmax': x + w,
             'ymin': y,
             'ymax': y + h,
             'label': row["HumanLabels"]},
            ignore_index=True)
        random_x = random.randint(0, image_shape[0] - w)
        random_y = random.randint(0, image_shape[1] - h)
        subset = masked_random[random_y: random_y + h, random_x: random_x + w]

        if subset.sum() == subset.size:
            randomized_rgb[
                random_y: random_y + h, random_x: random_x + w] = mosaiced_im[
                y: y + h, x: x + w]
            masked_random[random_y: random_y + h, random_x: random_x + w] = 0
            output_random_df = output_random_df.append(
                {'image_id': saved_random_image_path,
                    'xmin': random_x,
                    'xmax': random_x + w,
                    'ymin': random_y,
                    'ymax': random_y + h,
                    'label': row["HumanLabels"]},
                ignore_index=True)
        else:
            for i in range(10):
                if subset.sum() == subset.size:
                    break
                random_x = random.randint(0, image_shape[0] - w)
                random_y = random.randint(0, image_shape[1] - h)
                subset = masked_random[
                    random_y: random_y + h, random_x: random_x + w]
            if subset.sum() == subset.size:
                randomized_rgb[
                    random_y: random_y + h, random_x: random_x + w
                ] = mosaiced_im[
                    y: y + h, x: x + w]
                masked_random[
                    random_y: random_y + h, random_x: random_x + w] = 0
                output_random_df = output_random_df.append(
                    {'image_id': saved_random_image_path,
                        'xmin': random_x,
                        'xmax': random_x + w,
                        'ymin': random_y,
                        'ymax': random_y + h,
                        'label': row["HumanLabels"]},
                    ignore_index=True)
        count += 1
    elif count < 1156:
        x, y = indices[count]
        image[image == 255] = 214
        mosaiced_im[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1)] = image
        mosaiced_im_current = np.ones(image_shape, dtype=np.uint8) * 214
        mosaiced_im_current[
            x * tile_size_x: tile_size_x * (x + 1),
            y * tile_size_y: tile_size_y * (y + 1)] = image
        threshold = np.zeros_like(mosaiced_im_current)
        threshold[mosaiced_im_current == 214] = 0
        threshold[mosaiced_im_current != 214] = 255
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        ctrs, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(ctrs) == 1
        x, y, w, h = cv2.boundingRect(ctrs[0])
        # save rois as txt or csv
        output_df = output_df.append(
            {'image_id': saved_image_path,
             'xmin': x,
             'xmax': x + w,
             'ymin': y,
             'ymax': y + h,
             'label': row["HumanLabels"]},
            ignore_index=True)
        random_x = random.randint(0, image_shape[0] - w)
        random_y = random.randint(0, image_shape[1] - h)
        subset = masked_random[random_y: random_y + h, random_x: random_x + w]

        if subset.sum() == subset.size:
            randomized_rgb[
                random_y: random_y + h, random_x: random_x + w] = mosaiced_im[
                y: y + h, x: x + w]
            masked_random[random_y: random_y + h, random_x: random_x + w] = 0
            output_random_df = output_random_df.append(
                {'image_id': saved_random_image_path,
                    'xmin': random_x,
                    'xmax': random_x + w,
                    'ymin': random_y,
                    'ymax': random_y + h,
                    'label': row["HumanLabels"]},
                ignore_index=True)
        else:
            for i in range(10):
                if subset.sum() == subset.size:
                    break
                random_x = random.randint(0, image_shape[0] - w)
                random_y = random.randint(0, image_shape[1] - h)
                subset = masked_random[
                    random_y: random_y + h, random_x: random_x + w]
            if subset.sum() == subset.size:
                randomized_rgb[
                    random_y: random_y + h, random_x: random_x + w
                ] = mosaiced_im[
                    y: y + h, x: x + w]
                masked_random[
                    random_y: random_y + h, random_x: random_x + w] = 0
                output_random_df = output_random_df.append(
                    {'image_id': saved_random_image_path,
                        'xmin': random_x,
                        'xmax': random_x + w,
                        'ymin': random_y,
                        'ymax': random_y + h,
                        'label': row["HumanLabels"]},
                    ignore_index=True)

        count += 1
    else:
        count = 0
        cv2.imwrite(saved_image_path, mosaiced_im)
        cv2.imwrite(saved_random_image_path, randomized_rgb)
        image_count += 1

cols = ['xmin', 'xmax', 'ymin', 'ymax']
output_df[cols] = output_df[cols].applymap(np.int64)
output_df.to_csv("output_df_montage.csv")
output_random_df.to_csv("output_random_df_montage.csv")

tile_size = (512, 512)
x_tiles = image_shape[0] // tile_size[0]
y_tiles = image_shape[1] // tile_size[1]

output_random_disassembled_df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
split_images = []
# Set each of the tiles with an image in images_in_path
split_indices = list(itertools.product(range(x_tiles), range(y_tiles)))

output_dir = \
    "/Users/pranathivemuri/Downloads/CrossTrain-2019-12-02-15-17-48/split/"
for index, row in output_random_df.iterrows():
    for i, split_index in enumerate(split_indices):
        x_min_within_bounds = split_index[1] * tile_size[0] < row['xmin'] < (split_index[1] + 1) * tile_size[0]
        x_max_within_bounds = split_index[1] * tile_size[0] < row['xmax'] < (split_index[1] + 1) * tile_size[0]
        y_min_within_bounds = split_index[0] * tile_size[1] < row['ymin'] < (split_index[0] + 1) * tile_size[1]
        y_max_within_bounds = split_index[0] * tile_size[1] < row['ymax'] < (split_index[0] + 1) * tile_size[1]

        if (
            x_min_within_bounds and x_max_within_bounds and
            y_min_within_bounds and y_max_within_bounds):
            xmin = row['xmin'] - (split_index[1] * tile_size[0])
            xmax = row['xmax'] - (split_index[1] * tile_size[0])
            ymin = row['ymin'] - (split_index[0] * tile_size[1])
            ymax = row['ymax'] - (split_index[0] * tile_size[1])
            save_index = i
            path = os.path.join(
                output_dir,
                "{}_{}.{}".format(
                    os.path.basename(row["image_id"]).split(".")[0],
                    save_index, "tif"))
            output_random_disassembled_df = output_random_disassembled_df.append(
                {'image_id': path,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'label': row["label"]},
                ignore_index=True)
            break
        elif (
            x_min_within_bounds and not x_max_within_bounds and
            y_min_within_bounds and not y_max_within_bounds):
            xmin = row['xmin'] - (split_index[1] * tile_size[0])
            xmax = (split_index[1] * tile_size[0]) - row['xmin']
            ymin = row['ymin'] - (split_index[0] * tile_size[1])
            ymax = 512
            save_index = i
            path = os.path.join(
                output_dir,
                "{}_{}.{}".format(
                    os.path.basename(row["image_id"]).split(".")[0],
                    save_index, "tif"))
            output_random_disassembled_df = output_random_disassembled_df.append(
                {'image_id': path,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'label': row["label"]},
                ignore_index=True)
            break
        elif (
            not x_min_within_bounds and x_max_within_bounds and
            not y_min_within_bounds and y_max_within_bounds):
            xmin = (split_index[1] * tile_size[0]) - row['xmin']
            xmax = row['xmax'] - (split_index[1] * tile_size[0])
            ymin = 0
            ymax = row['ymax'] - (split_index[0] * tile_size[1])
            save_index = i
            path = os.path.join(
                output_dir,
                "{}_{}.{}".format(
                    os.path.basename(row["image_id"]).split(".")[0],
                    save_index, "tif"))
            output_random_disassembled_df = output_random_disassembled_df.append(
                {'image_id': path,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'label': row["label"]},
                ignore_index=True)
            break

output_random_disassembled_df.to_csv(
    "output_random_disassembled_df_montage.csv")
