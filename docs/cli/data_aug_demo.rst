.. _cli/data_aug_demo:

Mosaic stitches together small images to one large image to view in one go
===========================================================================

Assuming you already have a set of images to look at in one frame::


  $ lumi data_aug_demo --input_image /data/uv_microscopy_data/all_data_lumi_csv_unfiltered/val/MalariaRefocused_sl3_ch1_p97_t1.jpg --csv_path /data/uv_microscopy_data/all_data_lumi_csv_unfiltered/val.csv --output_png data_aug_mosaic.png --input_image_format .jpg --image_path_column image_id --fill_value 128

The ``lumi mosaic`` CLI tool provides the following options related to assembling a mosaic of shape
tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(len(images_in_path)) from tiles in im_dir.

* ``--im_dir``: Directory containing all the images to assemble to form the mosaic

* ``--tile_size``: Size of each tile in the mosaic

* ``--fill_value``:  All the pixels that are not filled in the symmetrical mosaic image
    by the resized tiles are filled with fill_value. For example specify an integer like --fill_value "30" or if you want to fill it with first pixel in the first image --fill_value first

* ``--output_png``: Path to write the mosaiced image to

* ``--fmt``: Format of images in the input directory