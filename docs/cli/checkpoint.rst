.. _cli/checkpoint:

Checkpoint management
=====================

Assuming you already have both your dataset and the config file ready, you can
create checkpoint for the model weights and model by running the command as follows::

  $ lumi checkpoint create config.yml -e name='Faster RCNN' -e alias=cnn_trial

The ``lumi checkpoint`` CLI tool provides the following options related to training.

* ``--config``/``-c``: Config file to use. If the flag is repeated, all config
  files will be merged in left-to-right order so that every file overwrites the
  configuration of keys defined previously.

For more info on checkpoints go to ``.. _tutorial/06-creating-own-checkpoints:``