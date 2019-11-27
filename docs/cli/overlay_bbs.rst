.. _cli/overlay_bbs:

Overlay bounding boxes and their annotations on images
======================================================

Assuming you already have both your dataset and their bounding box, labeled annotations ready::

  $ lumi overlay_bbs im_dir predicted_images/ --csv_path groundtruth_val.csv --image_path_column image_path --output_dir overlaid_images --input_image_format .jpg

The ``lumi overlay_bbs`` CLI tool provides the following options related to overlaying the bbs, labels on images.

* ``--im_dir``: Directory containing images to overlay on

* ``--csv_path``: Path to the data frame that contains bounding boxes, labels to overlay with

* ``--image_path_column``: Column in the dataframe that has the image path on disk

* ``--output_dir``: Save the overlaid images ot this directory

* ``--input_image_format``: Format of images in input directory

