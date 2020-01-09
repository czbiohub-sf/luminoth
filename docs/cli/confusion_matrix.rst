.. _cli/confusion_matrix:

Confusion matrix comparing ground truth and predicted bounding boxes detected by a model
========================================================================================

Assuming you already have both your dataset and predicted output ready to evaluate using confusion matrix, other validation metrics::

  $ lumi confusion_matrix --groundtruth_csv lumi_csv/val.csv --predicted_csv preds_val/objects.csv --output_txt cm.txt --output_fif cm.png --classes_json all_data/classes.json

The ``lumi confusion_matrix`` CLI tool provides the following options related to evaluation.

* ``--groundtruth_csv``: Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label and several rows corresponding to the groundtruth bounding box objects

* ``--predicted_csv``: Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label,prob and several rows corresponding to the predicted bounding box objects

* ``--output_txt``: Path to output txt file containing confusion matrix, normalized confusion matrix, precision, recall per class, defaults to None which means it prints the results to the terminal/stdout

* ``--output_fig``: Path to output fig file containing confusion matrix of the format returned by the matlab's plotConfusion function. The format of this figure can be either .png, .pdf, .eps, .svg

* ``--classes_json``: Path to a json file containing list of class label for the objects

* ``--iou_threshold``: IOU threshold below which the match of the predicted bounding box with the ground truth box is invalid, defaults to 0.5

* ``--confidence_threshold``: Confidence score threshold below which bounding box detection is of low confidence and is ignored while considering true positives, defaults to 0.9
