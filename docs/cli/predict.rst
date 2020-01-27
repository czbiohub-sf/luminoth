.. _cli/predict:

Predict with a model
====================

Assuming you already have both your dataset and the config file ready, you can
start your evaluation session by running the command as follows::

  $ lumi predict val/ -d preds_val/ --checkpoint trial -f objects.csv

The ``lumi predict`` CLI tool provides the following options related to training.

* ``path to an image or a directory``

* ``--checkpoint``: Checkpoint to use, is either a pretrained remote checkpoint downloaded or checkpoint created from your own dataset.

* ``-save-media-to`` or ``-d``: Bounding boxes overlaid images are saved in the directory specified here

* ``-f``: Bounding boxes found in each image are written into the csv, If -f is not present, the bounding boxes are printed out to the terminal STDOUT, otherwise output is redirected to the file.

* ``--min-prob``: When drawing, only draw bounding boxes with probability larger than min_prob

* ``--max-prob``: When drawing, only draw bounding boxes with probability lesser than max_prob

* ``--max-detections``: Maximum number of detections per image.

* ``--only-class``: Class to include when predicting.

* ``--ignore-class``: Class to ignore when predicting.

* ``--debug``: Boolean flag, if set, print debug level logging.


For more info on prediction go to ``.. _tutorial/01-first-steps``