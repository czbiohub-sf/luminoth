[![Luminoth](https://user-images.githubusercontent.com/270983/31414425-c12314d2-ae15-11e7-8cc9-42d330b03310.png)](https://luminoth.ai)

---

[![Build Status](https://travis-ci.com/czbiohub/luminoth-uv-imaging.svg)
[![codecov](https://codecov.io/gh/czbiohub/luminoth-uv-imaging/branch/master/graph/badge.svg)](https://codecov.io/gh/czbiohub/luminoth-uv-imaging)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Luminoth is an open source toolkit for **computer vision**. Currently, we support object detection, but we are aiming for much more. It is built in Python, using [TensorFlow](https://www.tensorflow.org/) and [Sonnet](https://github.com/deepmind/sonnet).

Read the full documentation [here](http://luminoth.readthedocs.io/).

![Example of Object Detection with Faster R-CNN](https://user-images.githubusercontent.com/1590959/36434494-e509be42-163d-11e8-99c1-d1aa728929ec.jpg)

> **DISCLAIMER**: Luminoth is still alpha-quality release, which means the internal and external interfaces (such as command line) are very likely to change as the codebase matures.

# Installation

Luminoth currently supports Python 2.7 and 3.4–3.6.

## Pre-requisites

To use Luminoth, [TensorFlow](https://www.tensorflow.org/install/) must be installed beforehand. If you want **GPU support**, you should install the GPU version of TensorFlow with `pip install tensorflow-gpu`, or else you can use the CPU version using `pip install tensorflow`.


## Installing from source

First, clone the repo on your machine and then install with `pip`:

```bash
git clone https://github.com/czbiohub/luminoth-uv-imaging.git
cd luminoth
pip install -e .
```

## Check that the installation worked

Simply run `lumi --help`.

# Supported models

Currently, we support the following models:

* **Object Detection**
  * [Faster R-CNN](https://arxiv.org/abs/1506.01497)
  * [SSD](https://arxiv.org/abs/1512.02325)

We also provide **pre-trained checkpoints** for the above models trained on popular datasets such as [COCO](http://cocodataset.org/) and [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/).

We also provide **pre-trained checkpoints** for the Faster R-CNN trained on RBC cell detection/parasite stage classification are available in release https://github.com/czbiohub/luminoth-uv-imaging/releases/tag/v0.3.0

68b8787e0bae - Faster R-CNN with commercial microscope dataset to detect healthy/ring/schizont/normal RBC
e6fa22ca6045 - Faster R-CNN with commercial microscope dataset to healthy/unhealthy RBC

Use `lumi predict` and the checkpoints above to test on your own dataset with similar 4 or 2 classes as mentioned above 

# Usage

There is one main command line interface which you can use with the `lumi` command. Whenever you are confused on how you are supposed to do something just type:

`lumi --help` or `lumi <subcommand> --help`

and a list of available options with descriptions will show up.

## Working with datasets

See [Adapting a dataset](http://luminoth.readthedocs.io/en/latest/usage/dataset.html).

## Training

See [Training your own model](http://luminoth.readthedocs.io/en/latest/usage/training.html) to learn how to train locally or in Google Cloud.

## Visualizing results

We strive to get useful and understandable summary and graph visualizations. We consider them to be essential not only for monitoring (duh!), but for getting a broader understanding of what's going under the hood. The same way it is important for code to be understandable and easy to follow, the computation graph should be as well.

By default summary and graph logs are saved to `jobs/` under the current directory. You can use TensorBoard by running:

```bash
tensorboard --logdir path/to/jobs
```

## Why the name?

> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)
>

## Dataset to model to prediction in a few steps

``` bash
lumi split_train_val annotated_bounding_boxes.txt annotated_bounding_boxes_1.txt annotated_bounding_boxes_2.txt --output_dir all_data_lumi_csv --percentage 0.9 --random_seed 42 --input_image_format .tif
lumi dataset transform --type csv --data-dir /lumi_csv/ --output-dir /tfdata/ --split train --split val --only-classes=table
lumi train -c config.yml
lumi eval --split train -c config.yml --no-watch
lumi eval --split val -c config.yml --no-watch
lumi checkpoint create config.yml -e name='Faster RCNN' -e alias=cnn_trial
lumi predict val/ -d preds_val/ --checkpoint trial -f objects.csv
lumi confusion_matrix --groundtruth_csv val.csv --predicted_csv pred_val.csv --output_txt output.txt --classes_json classes.json
```

# License

Copyright © 2018, [Tryolabs](https://tryolabs.com).
Released under the [BSD 3-Clause](LICENSE).
