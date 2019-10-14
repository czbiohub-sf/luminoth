.. _cli/predict:

Predict with a model
====================

Assuming you already have both your dataset and the config file ready, you can
start your evaluation session by running the command as follows::

  $ lumi predict val/ -d preds_val/ --checkpoint trial -f objects.json

The ``lumi predict`` CLI tool provides the following options related to training.

* ``--checkpoint``: Checkpoint to use, is either a pretrained remote checkpoint downloaded or checkpoint created from your own dataset.

* ``-d``: Bounding boxes overlaid images are saved in the directory specified here

* ``-f``: Bounding boxes for each image are written into the json, If -f is not present, the bounding boxes are printed out to the terminal STDOUT, otherwise output is redirected to the file. This json needs formatting as each bounding boxes are not separated by comma.

For more info on prediction go to ``.. _tutorial/01-first-steps``