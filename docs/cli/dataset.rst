.. _cli/dataset:

Dataset management
==================
Assuming you already have both your dataset in csv::

  $ lumi dataset transform \
          --type csv \
          --data-dir /lumi_csv/ \
          --output-dir /tfdata/ \
          --split train --split val 
          --only-classes=table

The ``lumi dataset`` CLI tool provides support for other dataset ``--type`` like pascal, imagenet, openimages, coco

For more info on dataset management go to ``.. _usage/dataset:``
