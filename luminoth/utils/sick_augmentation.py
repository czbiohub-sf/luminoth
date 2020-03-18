import cv2
import os
import numpy as np
import pandas as pd
import random
import re
import imgaug as ia
from imgaug import augmenters as iaa

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
DATA_DIR = os.path.join(
    HOME, "Downloads", "AllUVScopePreProcData/")

# Known input data parameters for the UV  microscope data
# using lumi disassemble
IMAGE_SHAPE = [520, 696]
TILE_SIZE_X = TILE_SIZE_Y = 120  # each rbc cell is in a 120 x 120 tile
INPUT_IMAGE_FORMAT = "tif"
BACKGROUND_COLOR = 214  # for 405 nm wavelength data, avg background intensity
NUM_TRIALS = 10  # Number of trials to find unoccupied random location subset
RANDOM_SEED = 42

# Change this to where you have the downloaded the dataset
im_dir = DATA_DIR + "{}/Matlab-RBC Instances/285 nm/sl{}/"
xls_file_name = DATA_DIR + "mergedMetadata_285nm_20200217.xls"
df = pd.read_excel(xls_file_name, sheet_name=None, ignore_index=True)
df = pd.concat(df.values(), ignore_index=True)

# Set random seed, so the randomized locations in the artificially generated
# mosaic stay the same on running the program again
random.seed(RANDOM_SEED)

# Filter for only one focus slice to avoid redundant images
count = 0
for index, row in df.iterrows():
    if "healthy" in row["HumanLabels"]:
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
saved_random_image_paths = []

while len(df) > 19:
    count = 0
    dicts = []
    # Iterate throw each row containing annotations for rbc cell tile
    kernel = np.ones((5, 5), np.uint8)
    for index, row in df.iterrows():
        sl = [
            i for i in row[
                "ParentFilename"].split("_") if "sl" in i][0]
        focus_slice = int(re.search(r'\d+', sl).group())
        image = cv2.imread(
            os.path.join(
                im_dir.format(row["datasetID"], focus_slice),
                row["InstanceFilename"]),
            cv2.IMREAD_GRAYSCALE)
        # save the image one with tiles randomly placed,
        # doesn't create the directory if it doesn't exist
        saved_random_image_path = os.path.join(
            DATA_DIR, "random_mosaic_sick", "sick_{}.tif".format(image_count))
        # First image, create arrays to place the randomized cells and a mask
        # to set the already occupied cell locations to 255
        if count == 0:
            random_mosaiced_im = np.ones(
                (IMAGE_SHAPE), dtype=np.uint8) * BACKGROUND_COLOR
            masked_random = np.ones(
                (IMAGE_SHAPE), dtype=np.uint8)

            # Find the contours for the image to find the bounding box
            threshold = np.zeros((TILE_SIZE_X, TILE_SIZE_Y), dtype=np.uint8)
            threshold[image == 255] = 0
            threshold[image != 255] = 255
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            ctrs, _ = cv2.findContours(
                threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            assert len(ctrs) == 1
            x, y, w, h = cv2.boundingRect(ctrs[0])

            # Get a random location not touching the boundary
            random_x = random.randint(0, IMAGE_SHAPE[1] - w)
            random_y = random.randint(0, IMAGE_SHAPE[0] - h)

            # Get the subset of the randomized extent & test if its already
            # filled
            subset = masked_random[
                random_y: random_y + h, random_x: random_x + w]

            # If the subset is not filled
            if subset.sum() == subset.size:
                # Ser the randomized location to the tile
                image[image == 255] = BACKGROUND_COLOR
                random_mosaiced_im[
                    random_y: random_y + h, random_x: random_x + w
                ] = image[y: y + h, x: x + w]
                # Set the filled subset of the array to zero
                masked_random[
                    random_y: random_y + h, random_x: random_x + w][
                    image[y: y + h, x: x + w] != 255] = 0
                # Save the location and label in csv file
                dicts.append(
                    {'image_id': saved_random_image_path,
                        'xmin': random_x,
                        'xmax': random_x + w,
                        'ymin': random_y,
                        'ymax': random_y + h,
                        'label': row["HumanLabels"]})
                indices_seen.append(index)
                count += 1
            else:
                # Try to get a unoccupied subset of random location
                trial_count = 0
                for i in range(NUM_TRIALS):
                    random_x = random.randint(0, IMAGE_SHAPE[1] - w)
                    random_y = random.randint(0, IMAGE_SHAPE[0] - h)
                    subset = masked_random[
                        random_y: random_y + h, random_x: random_x + w]
                    if subset.sum() == subset.size:
                        trial_count += 1
                        break
                assert trial_count <= 1, "trial_count {}, {}".format(
                    trial_count, saved_random_image_path)
                # If the subset is not filled
                if subset.sum() == subset.size:
                    # Ser the randomized location to the tile
                    image[image == 255] = BACKGROUND_COLOR
                    random_mosaiced_im[
                        random_y: random_y + h, random_x: random_x + w
                    ] = image[y: y + h, x: x + w]
                    # Set the filled subset of the array to zero
                    masked_random[
                        random_y: random_y + h, random_x: random_x + w][
                        image[y: y + h, x: x + w] != 255] = 0
                    # Save the location and label in csv file
                    dicts.append(
                        {'image_id': saved_random_image_path,
                            'xmin': random_x,
                            'xmax': random_x + w,
                            'ymin': random_y,
                            'ymax': random_y + h,
                            'label': row["HumanLabels"]})
                    indices_seen.append(index)
                    count += 1
        elif count >= 20 or index == len(df):
            # Reset count, increment image count, save image
            count = 0
            saved_random_image_paths.append(saved_random_image_path)
            cv2.imwrite(saved_random_image_path, random_mosaiced_im)
            for d in dicts:
                output_random_df = output_random_df.append(
                    d, ignore_index=True)
            image_count += 1
            dicts = []
        elif count < 20:
            # Repeat above for other indices at count greater than zero
            # Find the contours for the image to find the bounding box
            threshold = np.zeros((TILE_SIZE_X, TILE_SIZE_Y), dtype=np.uint8)
            threshold[image == 255] = 0
            threshold[image != 255] = 255
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            ctrs, _ = cv2.findContours(
                threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            assert len(ctrs) == 1
            x, y, w, h = cv2.boundingRect(ctrs[0])

            random_x = random.randint(0, IMAGE_SHAPE[1] - w)
            random_y = random.randint(0, IMAGE_SHAPE[0] - h)
            subset = masked_random[
                random_y: random_y + h, random_x: random_x + w]
            if subset.sum() == subset.size:
                image[image == 255] = BACKGROUND_COLOR
                random_mosaiced_im[
                    random_y: random_y + h, random_x: random_x + w
                ] = image[
                    y: y + h, x: x + w]
                masked_random[
                    random_y: random_y + h, random_x: random_x + w][
                    image[y: y + h, x: x + w] != 255] = 0
                dicts.append(
                    {'image_id': saved_random_image_path,
                        'xmin': random_x,
                        'xmax': random_x + w,
                        'ymin': random_y,
                        'ymax': random_y + h,
                        'label': row["HumanLabels"]})
                indices_seen.append(index)
                count += 1
            else:
                trial_count = 0
                for i in range(NUM_TRIALS):
                    random_x = random.randint(0, IMAGE_SHAPE[1] - w)
                    random_y = random.randint(0, IMAGE_SHAPE[0] - h)
                    subset = masked_random[
                        random_y: random_y + h, random_x: random_x + w]
                    if subset.sum() == subset.size:
                        trial_count += 1
                        break
                assert trial_count <= 1, "trial_count {}, {}".format(
                    trial_count, saved_random_image_path)
                if subset.sum() == subset.size:
                    image[image == 255] = BACKGROUND_COLOR
                    random_mosaiced_im[
                        random_y: random_y + h, random_x: random_x + w
                    ] = image[
                        y: y + h, x: x + w]
                    masked_random[
                        random_y: random_y + h, random_x: random_x + w][
                        image[y: y + h, x: x + w] != 255] = 0
                    dicts.append(
                        {'image_id': saved_random_image_path,
                            'xmin': random_x,
                            'xmax': random_x + w,
                            'ymin': random_y,
                            'ymax': random_y + h,
                            'label': row["HumanLabels"]})
                    indices_seen.append(index)
                    count += 1
    # Remove the bounding boxes that couldn't form an image because of going
    # through the dataframe
    if count != 0:
        indices_seen = indices_seen[0:len(indices_seen) - count]
        dicts = dicts[0:len(dicts) - count]

    # Note: Drop the rows already seen and re-run the above code again
    for index, row in df.iterrows():
        if index in indices_seen:
            df = df.drop([index])

# Note: Manually remove cells in csv file where an image has less than 20
# contours per 596 x 620 image, this might be the last few rows from each time
# running the above code, check for that
output_random_df.to_csv(os.path.join(
    DATA_DIR, "random_mosaic_sick/output_random_df_montage_sick.csv"))

bbs = []
images = []
dicts = []

for image_path in saved_random_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    images.append(image)
    tmp_df = output_random_df[output_random_df.image_id == image_path]

    bbs_per_image = []
    labels_per_image = []
    for index, row in tmp_df.iterrows():
        bbs_per_image.append(
            ia.BoundingBox(
                x1=row.xmin, y1=row.ymin,
                x2=row.xmax, y2=row.ymax,
                label=row.label))
    bbs.append(bbs_per_image)

# The array has shape (320, 520, 696) and dtype uint8.
images = np.array(images, dtype=np.uint8)

seq = iaa.Sequential([
    iaa.Flipud(1.0),  # vertically flips
    iaa.Fliplr(1.0),  # horizontal flips
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.AdditiveGaussianNoise(
        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)], random_order=True)
# Make some images brighter and some darker.
# In 20% of all cases, we sample the multiplier once per channel,
# which can end up changing the color of the images.
# iaa.Multiply((0.8, 1.2), per_channel=0.2)], random_order=True)
images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)

for index, bbs_per_image in enumerate(bbs_aug):
    aug_path = saved_random_image_paths[index].replace(
        "sick_", "aug_sick_")
    cv2.imwrite(aug_path, images_aug[index, :, :])
    for bb in bbs_per_image:
        dicts.append(
            {
                'image_id': aug_path,
                'xmin': bb.x1,
                'xmax': bb.x2,
                'ymin': bb.y1,
                'ymax': bb.y2,
                'label': bb.label})

for d in dicts:
    output_random_df = output_random_df.append(
        d, ignore_index=True)

output_random_df.to_csv(os.path.join(
    DATA_DIR, "random_mosaic_sick/output_random_df_montage_sick_aug.csv"))
