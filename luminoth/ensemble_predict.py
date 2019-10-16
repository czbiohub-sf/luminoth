import click
import os
import itertools
import sys
import json
from luminoth.utils.bbox_overlap import bbox_overlap
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


from .predict import predict


@click.command(help='Merge trained (or training) models and evaluate')
@click.argument('path-or-dir', nargs=-1)
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('output_path',
 '--output', '-f', default='-',
 help='Output file with the predictions (for example, csv bounding boxes) containing image_id,xmin,ymin,xmax,ymax,label')  # noqa
@click.option('--save-media-to', '-d', help='Directory to store media to.')
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--max-detections', default=100, type=int, help='Maximum number of detections per image.')  # noqa
@click.option('--only-class', '-k', default=None, multiple=True, help='Class to ignore when predicting.')  # noqa
@click.option('--ignore-class', '-K', default=None, multiple=True, help='Class to ignore when predicting.')  # noqa
@click.option('--iou-threshold', type=float, default=0.5, help='IOU threshold below which the bounding box is invalid')  # noqa
@click.option('--confidence-threshold', type=float, default=0.9, help='Confidence score threshold below which bounding box detection is invalid')  # noqa
@click.option('--weights', nargs='+', type=float, help='weight of detection from each model', default=None) # noqa
def ensemble_predict(
        path_or_dir, config_files, checkpoint, override_params,
        output_path, save_media_to, min_prob, max_detections, only_class,
        ignore_class, debug, confidence_threshold, iou_threshold, weights):
    submissions = {}
    for config_file in config_files:
        if debug:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            tf.logging.set_verbosity(tf.logging.ERROR)

        if only_class and ignore_class:
            click.echo(
                "Only one of `only-class` or `ignore-class` may be specified."
            )
            return

        # Process the input and get the actual files to predict.
        files = predict.resolve_files(path_or_dir)
        if not files:
            error = ('No files to predict. Accepted formats are: {}.'.format(
                     ', '.join(predict.IMAGE_FORMATS)))
            click.echo(error)
            return
        else:
            click.echo('Found {} files to predict.'.format(len(files)))

        # Create `save_media_to` if specified and it doesn't exist.
        if save_media_to:
            tf.gfile.MakeDirs(save_media_to)

        # Resolve the config to use and initialize the model.
        if checkpoint:
            config = predict.get_checkpoint_config(checkpoint)
        elif config_files:
            config = predict.get_config(config_files)
        else:
            click.echo(
                'Neither checkpoint not config specified, assuming `accurate`.'
            )
            config = predict.get_checkpoint_config('accurate')

        if override_params:
            config = predict.override_config_params(config, override_params)
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        if tf.gfile.Exists(classes_file):
            class_labels = json.load(tf.gfile.GFile(classes_file))
        else:
            class_labels = None

        # Filter bounding boxes according to `min_prob` and `max_detections`.
        if config.model.type == 'fasterrcnn':
            if config.model.network.with_rcnn:
                config.model.rcnn.proposals.total_max_detections = \
                    max_detections
            else:
                config.model.rpn.proposals.post_nms_top_n = max_detections
            config.model.rcnn.proposals.min_prob_threshold = min_prob
        elif config.model.type == 'ssd':
            config.model.proposals.total_max_detections = max_detections
            config.model.proposals.min_prob_threshold = min_prob
        else:
            raise ValueError(
                "Model type '{}' not supported".format(config.model.type)
            )

        # Instantiate the model indicated by the config.
        network = predict.PredictorNetwork(config)
        dets = []
        image_names = []
        # Iterate over files and run the model on each.
        for file in files:
            objects = predict.predict_image(
                network, file,
                only_classes=only_class,
                ignore_classes=ignore_class,
                save_path=None,
            )

            if objects is not None:
                for obj in objects:
                    label_name = obj['label']
                    image_names.append(file)
                    dets.append(
                        obj + [class_labels.index(label_name), obj["prob"]])

        submissions[config['train']['job_dir']] = dets
    predict_and_save_ensemble_result(
        submissions,
        image_names,
        confidence_threshold,
        iou_threshold,
        output_path,
        save_media_to,
        class_labels,
        weights)


def predict_and_save_ensemble_result(
        submissions,
        image_names,
        conf_thresh,
        iou_thresh,
        output_csv_path,
        output_image_path,
        class_labels,
        weights):
    dets = list(itertools.chain.from_iterable(list(submissions.values())))
    image_names = list(itertools.chain.from_iterable(image_names))
    ensembled = ensemble(
        dets, conf_thresh=conf_thresh, iou_thresh=iou_thresh, weights=weights)

    df = pd.DataFrame(columns=predict.LUMI_CSV_COLUMNS)
    for index, e in enumerate(ensembled):
        file = image_names[index]
        # Open and read the image to predict.
        with tf.gfile.Open(file, 'rb') as f:
            try:
                image = Image.open(f).convert('RGB')
            except (tf.errors.OutOfRangeError, OSError) as e:
                click.echo()
                click.echo(
                    'Error while processing {}: {}'.format(
                        file, e))
                return
        x, y, w, h, cls_name, confidence = e
        obj = {'image_id': file,
               'xmin': x,
               'xmax': x + w,
               'ymin': y,
               'ymax': y + h,
               'label': class_labels[cls_name],
               'prob': confidence}
        # Get the media output path, if media storage is requested.
        predict.vis_objects(np.array(image), obj).save(output_image_path)
        # TODO: Not writing csv for video files for now.
        df = df.append(obj, ignore_index=True)

    # Build the `Formatter` based on the outputs, which automatically writes
    # the formatted output to all the requested output files.
    if output_csv_path == '-':
        output = sys.stdout
        output.write(df.to_string())
        output.close()
    else:
        df.to_csv(output_csv_path)


def ensemble(dets, conf_thresh=0.5, iou_thresh=0.1, weights=None):
    # Detections from different models
    ndets = len(dets)

    # weights is none, all models are equally weighted
    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(len(weights)):
            weights[i] /= s

    out = list()
    used = set()

    for idet in range(ndets):
        det = dets[idet]
        for box in det:
            if tuple(box) in used:
                continue

            used.add(tuple(box))
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not tuple(obox) in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = bbox_overlap(box[:4], obox[:4])
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if bestbox is not None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.add(tuple(bestbox))

            # Now we've gone through all other detectors
            if not found:
                new_box = list(box)
                new_box[5] /= ndets
                if new_box[5] >= conf_thresh:
                    out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                bx = 0.0
                by = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    # weight
                    w = bb[1]
                    wsum += w

                    # (box_x, box_y, box_w, box_h)
                    b = bb[0]

                    bx += w * b[0]
                    by += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                bx /= wsum
                by /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [
                    bx, by, bw, bh, box[4], conf]
                if new_box[5] >= conf_thresh:
                    out.append(new_box)
    return out
