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
python setup.py install
```

## Check that the installation worked

Simply run `lumi --help`.

# Supported models

Currently, we support the following models:

* **Object Detection**
  * [Faster R-CNN](https://arxiv.org/abs/1506.01497)
  * [SSD](https://arxiv.org/abs/1512.02325)

We also provide **pre-trained checkpoints** for the above models trained on popular datasets such as [COCO](http://cocodataset.org/) and [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/).

We also provide **pre-trained checkpoints** for the Faster R-CNN trained on RBC cell detection/parasite stage classification are available in release https://github.com/czbiohub/luminoth-uv-imaging/releases/tag/v0.4.0
1a0f3002f674 - Faster R-CNN with customized UV microscope dataset to detect healthy/ring/schizont/normal RBC
1fbb5e928fd5 - Faster R-CNN with Leica commercial microscope dataset to detect healthy/ring/schizont/normal RBC

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

### Docker

It is recommended that you run luminoth-uv0imaging inside a Docker container, especially if you're using shared resources like a GPU server. 
you can do so:
```buildoutcfg
docker build -t imaging_docker:gpu_py36_cu90 -f Dockerfile .
```
Now you want to start a Docker container from your image, which is the virtual environment you will run your code in.
```buildoutcfg
nvidia-docker run -it -p <your port>:<exposed port> -v <your dir>:/<dirname inside docker> imaging_docker:gpu_py36_cu90 bash
```
If you look in the Dockerfile, you can see that there are two ports exposed, one is typically used for Jupyter (8888)
and one for Tensorboard (6006). To be able to view these in your browser, you need map the port with the -p argument.
The -v arguments similarly maps directories. You can use multiple -p and -v arguments if you want to map multiple things.
The final 'bash' is to signify that you want to run bash (your usual Unix shell). 

If you want to launch a Jupyter notebook inside your container, you can do so with the following command:
```buildoutcfg
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then you can access your notebooks in your browser at:
```buildoutcfg
http://<your server name (e.g. fry)>:<whatever port you mapped to when starting up docker>
```
You will need to copy/paste the token generated in your Docker container.

## Why the name?

> The Dark Visor is a Visor upgrade in Metroid Prime 2: Echoes. Designed by the **Luminoth** during the war, it was used by the Champion of Aether, A-Kul, to penetrate Dark Aether's haze in battle against the Ing.
>
> -- [Dark Visor - Wikitroid](http://metroid.wikia.com/wiki/Dark_Visor)
>

## Dataset to model to prediction in a few steps
Also if you have image formats that are not compatible with lumi, use imagemagick to convert image from one format to another 
`brew install imagemagick`  and it installs mogrify in addition to a command called convert `mogrify -format png *.tif` or for videos use ffmpeg  
`brew install ffmpeg 
ffmpeg -pattern_type glob -i '*.jpg' -vf "setpts=5*PTS" test_r5.mp4
ffmpeg -i in.MOV  -pix_fmt rgb24 output_tflite.gif
ffmpeg -i in.m4v out.mp4
ffmpeg -i out.mp4 -vf format=gray gray.mp4` 
If you have any trouble please print `lumi command --help` example `lumi predict --help` or refer to documentation files in ./docs/tutorial/
1. Use the first command below, To test if your bounding boxes and objects in csv file and images look right
2. Split data given a csv file containing path to image and bounding boxes with extreme corners of rectangles of objects with labels. The csv file contains
xmin, xmax, ymin, ymax, label, image_id. Given that csv file creates 2 folders with train, val and 2 csv files with train.csv and val.csv
3. Third command below converts different images and csv files to tensorflow record files, and depending on percentage paramter creates - train.tfrecords containing that fraction of data 1-fraction of data into val.tfrecords
4. Fourth command, Train using the config, example config is examples folder, change and copy the dictionary from base_config.yml if any other parameters should be changed for your training, dataset and it your config.yml
5. Fifth command below, Evaluate training data -  prints precision, losses for each class on training data
6. Sixth command below, Evaluate validation -  data prints precision, losses for each class on validation data
7. Checkpoint creation, Create checkpoint to save the model so far created
8. Export the checkpoint to a .tar folder of same name as checkpoint so if you lose it or the docker exits you can import it using `lumi checkpoint import cb0e5d92a854.tar`
9. Predict on a dataset given a checkpoint creates a pred_val.csv file and boundary boxes overlaid images in a folder as well
10. Print confusion matrix and confusion matrix png and other metrics comparing ground truth csv file and prediction csv file
``` bash
lumi overlay_bbs --im_dir "/set_2" --csv_path "set_2/bb_labels.csv" --output_dir overlaid_mosaic_cells/ --input_image_format tif
lumi split_train_val annotated_bounding_boxes.csv annotated_bounding_boxes_1.csv annotated_bounding_boxes_2.csv --output_dir lumi_csv --percentage 0.9 --random_seed 42 --input_image_format .tif
lumi dataset transform --type csv --data-dir /lumi_csv/ --output-dir /tfdata/ --split train --split val --only-classes=table
lumi train -c config.yml
lumi eval --split train -c config.yml --no-watch
lumi eval --split val -c config.yml --no-watch
lumi checkpoint create config.yml -e name='Faster RCNN' -e alias=cnn_trial
lumi checkpoint export cb0e5d92a854
lumi predict val/ -d preds_val/ --checkpoint trial -f pred_val.csv
lumi confusion_matrix --groundtruth_csv val.csv --predicted_csv pred_val.csv --output_txt output.txt --classes_json classes.json
```

# License

Copyright © 2018, [Tryolabs](https://tryolabs.com).
Released under the [BSD 3-Clause](LICENSE).
