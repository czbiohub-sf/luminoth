import cv2
import os
import numpy as np
import itertools
import pandas as pd
import random

"""
This program is used to preprocess the 2048 x 2048 leica commerical microscope
images, each image consisting of about 9000 cells. The human annotated data is
for only random subset of the cells on each image. The dataset folder
CrossTrain-2019-12-02-15-17-48 consists of two folders SCP-2019-10-24 Malaria,
SCP-2019-11-12 Malaria and the human annotated labels, bounding box locations
for the random rbc cell instances and their names under FileName and they
are stored in RBC Instances/405 nm 40x/sl1
"""
# Constants
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "Downloads", "CrossTrain-2019-12-02-15-17-48")

# Known input data parameters for the leica commercial microscope data
# 512 x 512 randomly placed cells mosaic image is created instead of
# 2048 x 2048, for prediction 2048 x 2048 are split into 512 x 512 images
# using lumi disassemble
IMAGE_SHAPE = [512, 512]
TILE_SIZE_X = TILE_SIZE_Y = 60  # each rbc cell is in a 60 x 60 tile
INPUT_IMAGE_FORMAT = "tif"
BACKGROUND_COLOR = 214  # for 405 nm wavelength data, avg background intensity
NUM_TRIALS = 10  # Number of trials to find unoccupied random location subset
RANDOM_SEED = 42

# Change this to where you have the downloaded the dataset
im_dir = DATA_DIR + "/{}/RBC Instances/405 nm 40x/sl1"
df = pd.read_excel(DATA_DIR + "/HumanAnnotated_405 nm 40x.xls")

# Set random seed, so the randomized locations in the artificially generated
# mosaic stay the same on running the program again
random.seed(RANDOM_SEED)
# Maximum number of tiles is 512 / 60, get the cartesean product for each of
# the 60 x 60 locations inside the 512 x 512 array
x_tiles, y_tiles = IMAGE_SHAPE[0] // TILE_SIZE_X, IMAGE_SHAPE[1] // TILE_SIZE_Y
indices = list(itertools.product(range(x_tiles), range(y_tiles)))


# Remove 477 healthy rows - Specific observation for the groundtruth
# data in excel sheet above to correct for class imbalancing
count = 0
for index, row in df.iterrows():
    if row["HumanLabels"] == "healthy" and count < 477:
        df = df.drop([index])
        count += 1
# Randomize rows to not pick two similar labeled cells or same dataset cells
df = df.sample(frac=1).reset_index(drop=True)

# make a csv as required by lumi with only required columns
LUMI_CSV_COLUMNS = ['image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
output_random_df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
image_count = 0
# List of indices, already copied to the csv file and rbc cells in the row
# are in an image
indices_seen = []


count = 0
# Iterate throw each row containing annotations for rbc cell tile
kernel = np.ones((5, 5), np.uint8)
for index, row in df.iterrows():
    image = cv2.imread(
        os.path.join(
            im_dir.format(row["datasetID"]), row["Filename"]),
        cv2.IMREAD_GRAYSCALE)
    # save the image one with tiles randomly placed,
    # doesn't create the directory if it doesn't exist
    saved_random_image_path = os.path.join(
        DATA_DIR, "random_mosaic_1", "{}.tif".format(image_count))

    # First image, create arrays to place the randomized cells and a mask to
    # set the already occupied cell locations to 255
    if count == 0:
        mosaiced_im = np.ones(
            IMAGE_SHAPE, dtype=np.uint8) * BACKGROUND_COLOR
        random_mosaiced_im = np.ones(
            (IMAGE_SHAPE), dtype=np.uint8) * BACKGROUND_COLOR
        masked_random = np.ones(
            (IMAGE_SHAPE), dtype=np.uint8)

        # Set the cell at the x, y tile location
        x, y = indices[count]
        image[image == 255] = BACKGROUND_COLOR
        mosaiced_im[
            x * TILE_SIZE_X: TILE_SIZE_X * (x + 1),
            y * TILE_SIZE_Y: TILE_SIZE_Y * (y + 1)] = image
        # Binarize the array
        threshold = np.zeros_like(mosaiced_im)
        threshold[mosaiced_im == BACKGROUND_COLOR] = 0
        threshold[mosaiced_im != BACKGROUND_COLOR] = 255
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # Find the contours for the mosaic image to find the cell bounding box
        ctrs, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(ctrs) == 1
        x, y, w, h = cv2.boundingRect(ctrs[0])

        # Get a random location not touching the boundary
        random_x = random.randint(0, IMAGE_SHAPE[0] - w)
        random_y = random.randint(0, IMAGE_SHAPE[1] - h)

        # Get the subset of the randomized extent & test if its already filled
        subset = masked_random[random_y: random_y + h, random_x: random_x + w]

        # If the subset is not filled
        if subset.sum() == subset.size:
            # Ser the randomized location to the tile
            random_mosaiced_im[
                random_y: random_y + h, random_x: random_x + w] = mosaiced_im[
                y: y + h, x: x + w]
            # Set the filled subset of the array to zero
            masked_random[random_y: random_y + h, random_x: random_x + w] = 0
            # Save the location and label in csv file
            output_random_df = output_random_df.append(
                {'image_id': saved_random_image_path,
                    'xmin': random_x,
                    'xmax': random_x + w,
                    'ymin': random_y,
                    'ymax': random_y + h,
                    'label': row["HumanLabels"]},
                ignore_index=True)
            indices_seen.append(index)
        else:
            # Try to get a unoccupied subset of random location
            for i in range(NUM_TRIALS):
                if subset.sum() == subset.size:
                    break
                random_x = random.randint(0, IMAGE_SHAPE[0] - w)
                random_y = random.randint(0, IMAGE_SHAPE[1] - h)
                subset = masked_random[
                    random_y: random_y + h, random_x: random_x + w]
                # If the subset is not filled
                if subset.sum() == subset.size:
                    # Ser the randomized location to the tile
                    random_mosaiced_im[
                        random_y: random_y + h, random_x: random_x + w
                    ] = mosaiced_im[y: y + h, x: x + w]
                    # Set the filled subset of the array to zero
                    masked_random[
                        random_y: random_y + h, random_x: random_x + w] = 0
                    # Save the location and label in csv file
                    output_random_df = output_random_df.append(
                        {'image_id': saved_random_image_path,
                            'xmin': random_x,
                            'xmax': random_x + w,
                            'ymin': random_y,
                            'ymax': random_y + h,
                            'label': row["HumanLabels"]},
                        ignore_index=True)
                    indices_seen.append(index)
        count += 1
    elif count < len(indices):
        # Repeat above for other indices at count greater than zero
        x, y = indices[count]
        image[image == 255] = BACKGROUND_COLOR
        mosaiced_im[
            x * TILE_SIZE_X: TILE_SIZE_X * (x + 1),
            y * TILE_SIZE_Y: TILE_SIZE_Y * (y + 1)] = image

        # Get a mask for the only current cell tile
        mosaiced_im_current = np.ones(
            IMAGE_SHAPE, dtype=np.uint8) * BACKGROUND_COLOR
        mosaiced_im_current[
            x * TILE_SIZE_X: TILE_SIZE_X * (x + 1),
            y * TILE_SIZE_Y: TILE_SIZE_Y * (y + 1)] = image
        threshold = np.zeros_like(mosaiced_im_current)
        threshold[mosaiced_im_current == BACKGROUND_COLOR] = 0
        threshold[mosaiced_im_current != BACKGROUND_COLOR] = 255
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        ctrs, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(ctrs) == 1
        x, y, w, h = cv2.boundingRect(ctrs[0])

        random_x = random.randint(0, IMAGE_SHAPE[0] - w)
        random_y = random.randint(0, IMAGE_SHAPE[1] - h)
        subset = masked_random[random_y: random_y + h, random_x: random_x + w]

        if subset.sum() == subset.size:
            random_mosaiced_im[
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
            indices_seen.append(index)
        else:
            for i in range(NUM_TRIALS):
                if subset.sum() == subset.size:
                    break
                random_x = random.randint(0, IMAGE_SHAPE[0] - w)
                random_y = random.randint(0, IMAGE_SHAPE[1] - h)
                subset = masked_random[
                    random_y: random_y + h, random_x: random_x + w]
            if subset.sum() == subset.size:
                random_mosaiced_im[
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
                indices_seen.append(index)

        count += 1

    else:
        # Reset count, increment image count, save image
        count = 0
        cv2.imwrite(saved_random_image_path, random_mosaiced_im)
        image_count += 1

# Note: Drop the rows already seen and re-run the above code again
for index, row in df.iterrows():
    if index in indices_seen:
        df = df.drop([index])
# Re-run the above code starting at count = 0, make sure you don't reset the
# image_count to 0, Repeat until indices_seen covers all indices

# Note: Manually remove cells in csv file where an image has less than 45
# contours per 512 x 512 image, this might be the last few rows from each time
# running the above code, check for that
output_random_df.to_csv("output_random_df_montage_1.csv")
