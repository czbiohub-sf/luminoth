.. _cli/data_aug_demo:

Data augmentation demo augments one image and stitches together all the results images to one large image to view in one go
===========================================================================================================================

Assuming you have image and bounding boxes, and you want to visualize how data augmentation would look like::


  $ lumi data_aug_demo --input_image image.jpg --csv_path val.csv --output_png data_aug_mosaic.png --input_image_format .jpg --image_path_column image_id --fill_value 128

The ``lumi data_aug_demo`` CLI tool provides the following options related to assembling a mosaic from all the images 
available data augmentation strategies. The output array is of shape
TILE_SIZE[0] * sqrt(len(DATA_AUGMENTATION_STRATEGIES)), TILE_SIZE[1] * sqrt(len(DATA_AUGMENTATION_STRATEGIES)).
TILE_SIZE, DATA_AUGMENTATION_STRATEGIES are constants declared. TILE_SIZE is 256, 256.

* ``--input_image``: Image to augment and mosaic

* ``--csv_path``: Path to the data frame that contains bounding boxes, labels to overlay with

* ``--output_png``: Path to write the mosaiced image to

* ``--input_image_format``: Path to write the mosaiced image to

* ``--image_path_column``: Column in the dataframe that has the image path on disk

* ``--fill_value``:All the pixels that are not filled in the symmetrical mosaic image
    by the resized tiles are filled with fill_value. For example specify an integer like --fill_value "30" or if you want to fill it with first pixel in the first image --fill_value first
