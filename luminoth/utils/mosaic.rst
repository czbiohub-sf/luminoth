.. _cli/mosaic:

Mosaic stitches together small images to one large image to view in one go
===========================================================================

Assuming you already have a set of images to look at in one frame::


  $ lumi mosaic --im_dir all_data_no_mosaic_lumi_csv/ --tile_size 40, 50 --fill_value "30" --output_png mosaiced.png --fmt .tif

The ``lumi mosaic`` CLI tool provides the following options related to assembling a mosaic of shape
tile_size[0] * sqrt(len(images_in_path)) * tile_size[1] * sqrt(len(images_in_path)) from tiles in im_dir.

* ``--im_dir``: Directory containing all the images to assemble to form the mosaic

* ``--tile_size``: Size of each tile in the mosaic

* ``--fill_value``:  All the pixels that are not filled in the symmetrical mosaic image
    by the resized tiles are filled with fill_value. For example specify an integer like --fill_value "30" or if you want to fill it with first pixel in the first image --fill_value first

* ``--output_png``: Path to write the mosaiced image to

* ``--fmt``: Format of images in the input directory