.. _cli/eval:

Evaluating a model
==================

Assuming you already have both your dataset and the config file ready, you can
start your evaluation session by running the command as follows::

  $ lumi eval -c config.yml --no-watch

The ``lumi eval`` CLI tool provides the following options related to training.

* ``--config``/``-c``: Config file to use. If the flag is repeated, all config
  files will be merged in left-to-right order so that every file overwrites the
  configuration of keys defined previously.

* ``--no-watch``: no-watch flag implies it would stop looking for checkpoints 
  and stops the processes, otherwise the process would not return to bash and
  keeps a look out for new check points getting created from the config file


For more info on evalutation go to ``.. _tutorial/05-evaluating-models``
