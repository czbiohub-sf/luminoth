.. _cli/data_aug_demo:

Data augmentation demo augments one image and stitches together all the augmentation result images to one large image to view in one go
=======================================================================================================================================

Assuming you have image and bounding boxes, and you want to visualize how data augmentation would look like::


  $ lumi data_aug_demo --input_image image.jpg --csv_path val.csv --output_png data_aug_mosaic.png --input_image_format .jpg --image_path_column image_id --fill_value 19

The ``lumi data_aug_demo`` CLI tool provides the following options related to assembling a mosaic from all the images 
resulting from available data augmentation strategies. The output array is of shape
TILE_SIZE[0] * sqrt(len(DATA_AUGMENTATION_CONFIGS)), TILE_SIZE[1] * sqrt(len(DATA_AUGMENTATION_CONFIGS)).
TILE_SIZE, DATA_AUGMENTATION_CONFIGS are constants declared in data_aug_demo.py. Change the configs in the module ``data_aug_demo.py`` to see change in augmentatation results. TILE_SIZE is tile of the resized augmentation image in the mosaic displayed. It is fixed at (256,256).

* ``--input_image``: Image to augment and mosaic the resulted augmented images to

* ``--csv_path``: Path to the data frame that contains bounding boxes, labels present in the input_image

* ``--output_png``: Path to write the mosaiced image to

* ``--input_image_format``: Format of the input image

* ``--image_path_column``: Column in the dataframe that has the image path on disk that indicates the bounding box, label's presence in that image

* ``--fill_value``:All the pixels that are not filled in the symmetrical mosaic image by the resized tiles are filled with fill_value, defaults to 128.  For example specify an integer like ``--fill_value "30"`` or if you want to fill it with first pixel in the first image and unsure what it is ``--fill_value first``
