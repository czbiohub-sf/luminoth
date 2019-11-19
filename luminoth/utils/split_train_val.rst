.. _cli/split_train_val:

Confusion matrix comparing ground truth and predicted bounding boxes detected by a model
========================================================================================

Assuming you already have both your dataset and predicted output ready::

  $ lumi split_train_val bb_labels_no_mosaic.txt --output_dir all_data_no_mosaic_lumi_csv

The ``lumi split_train_val`` CLI tool provides the following options related to splitting and organizing the data.

* ``filenames``: List of all the bounding box annotation files, can be 1 to n

* ``--percentage``: Percentage of total images to add to the training directory, 1 - percentage is added to the val directory

* ``--random_seed``: Random seed for shuffling the images

* ``--filter_dense_anns``: If this flag is set to True, images with only one class that has more annotations
  are completely ignored

* ``--input_image_format``: Input image format

* ``--output_image_format``: Output image format for the images getting saved in train and val directory
