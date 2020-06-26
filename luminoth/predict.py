import click
import pandas as pd
import numpy as np
import os
import cv2
import skvideo.io
import sys
import time
import tempfile
import tensorflow as tf
import xlsxwriter
import json

from PIL import Image
from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork
from luminoth.utils.split_train_val import get_image_paths_per_class
from luminoth.utils.vis import vis_objects

IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'tif']
VIDEO_FORMATS = ['mov', 'mp4', 'avi']  # TODO: check if more formats work
LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'prob']


def get_file_type(filename):
    extension = filename.split('.')[-1].lower()
    if extension in IMAGE_FORMATS:
        return 'image'
    elif extension in VIDEO_FORMATS:
        return 'video'


def resolve_files(path_or_dir):
    """Returns the file paths for `path_or_dir`.

    Args:
        path_or_dir: String or list of strings for the paths or directories to
            run predictions in. For directories, will return all the files
            within.

    Returns:
        List of strings with the full path for each file.
    """
    if not isinstance(path_or_dir, tuple):
        path_or_dir = (path_or_dir,)

    paths = []
    for entry in path_or_dir:
        if tf.gfile.IsDirectory(entry):
            paths.extend([
                os.path.join(entry, f)
                for f in tf.gfile.ListDirectory(entry)
                if get_file_type(f) in ('image', 'video')
            ])
        elif get_file_type(entry) in ('image', 'video'):
            if not tf.gfile.Exists(entry):
                click.echo('Input {} not found, skipping.'.format(entry))
                continue
            paths.append(entry)

    return paths


def filter_classes(objects, only_classes=None, ignore_classes=None):
    if ignore_classes:
        objects = [o for o in objects if o['label'] not in ignore_classes]

    if only_classes:
        objects = [o for o in objects if o['label'] in only_classes]

    return objects


def filter_probabilities(objects, min_prob=None, max_prob=None):
    if min_prob:
        objects = [o for o in objects if o['prob'] > min_prob]

    if max_prob:
        objects = [o for o in objects if o['prob'] <= max_prob]

    return objects


def bbs_pixel_apart(obj, objects, pixel_distance):
    repeated_indices = []
    for index, each_obj in enumerate(objects):
        set_index_flags = 0
        unique_differences = \
            np.unique(np.fabs(np.subtract(each_obj, obj))).tolist()
        for i in unique_differences:
            if i <= pixel_distance:
                set_index_flags += 1
        if set_index_flags == len(unique_differences):
            repeated_indices.append(index)
    return repeated_indices


def filter_close_bbs(predictions, pixel_distance):
    # Save a prediction by suppressing the class with
    # lowest probability for the same bounding box
    objects = [prediction["bbox"] for prediction in predictions]
    labels = [prediction["label"] for prediction in predictions]
    probs = [prediction["prob"] for prediction in predictions]

    predictions = [None] * len(objects)
    assert len(objects) == len(labels) == len(probs)
    count = 0
    for obj, label, prob in zip(objects, labels, probs):
        repeated_indices = bbs_pixel_apart(obj, objects, pixel_distance)
        if len(repeated_indices) > 0:
            repeated_probs = [probs[i] for i in repeated_indices]
            repeated_probs.append(prob)
            repeated_indices.append(count)
            max_prob = max(repeated_probs)
            assert len(repeated_probs) == len(repeated_indices)
            prob_index = [
                index for index, prob in zip(
                    repeated_indices, repeated_probs)
                if prob == max_prob][0]
            d = {
                'bbox': objects[prob_index],
                'label': labels[prob_index],
                'prob': round(max_prob, 4)}
            predictions[prob_index] = d
        else:
            if objects.count(obj) == 1:
                d = {
                    'bbox': obj,
                    'label': label,
                    'prob': round(prob, 4)}
                predictions[count] = d
            elif objects.count(obj) > 1:
                prob_repeated_objs = [
                    [i, probs[i]] for i, value in enumerate(objects)
                    if value == obj]
                repeated_indices = [i for (i, _) in prob_repeated_objs]
                repeated_probs = [j for (_, j) in prob_repeated_objs]
                max_prob = max(repeated_probs)
                prob_index = [
                    index for index, prob in zip(
                        repeated_indices, repeated_probs)
                    if prob == max_prob][0]
                d = {
                    'bbox': obj,
                    'label': labels[prob_index],
                    'prob': round(max_prob, 4)}
                predictions[prob_index] = d
        count += 1
    predictions = list(filter(None, predictions))
    predictions = sorted(
        predictions, key=lambda x: x['prob'], reverse=True)

    return predictions


def rename_labels(predictions, new_labels):
    with open(new_labels, "r") as f:
        new_labels = json.load(f)
    objects = [prediction["bbox"] for prediction in predictions]
    labels = [prediction["label"] for prediction in predictions]
    new_labels = [new_labels[label] for label in labels]
    probs = [prediction["prob"] for prediction in predictions]
    predictions = [None] * len(objects)
    predictions = sorted([
        {
            'bbox': obj,
            'label': label,
            'prob': round(prob, 4),
        } for obj, label, prob in zip(objects, new_labels, probs)
    ], key=lambda x: x['prob'], reverse=True)
    return predictions


def predict_image(network, path, only_classes=None, ignore_classes=None,
                  save_path=None, min_prob=None, max_prob=None,
                  pixel_distance=0, new_labels=None):
    click.echo('Predicting {}...'.format(path), nl=False)
    extension = path.split(".")[-1]
    basename = os.path.basename(path)

    # Write to only a png - as jpg creates all zeros uint16 data and conversion
    # to rgb, pil and tiffle can't handle uin16 data conversiont to RGB as wekk
    if extension == "tif":
        image = Image.open(path)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        tempname = basename.replace("." + extension, "." + "png")
        path = os.path.join(tempfile.mkdtemp(prefix='lumi'), tempname)
        cv2.imwrite(path, image)
    # Open and read the image to predict.
    with tf.gfile.Open(path, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
        except (tf.errors.OutOfRangeError, OSError) as e:
            click.echo()
            click.echo('Error while processing {}: {}'.format(path, e))
            return

    # Run image through the network.
    objects = network.predict_image(image)

    # Filter the results according to the user input.
    objects = filter_classes(
        objects,
        only_classes=only_classes,
        ignore_classes=ignore_classes
    )

    # Filter the results according to the user input.
    objects = filter_probabilities(
        objects,
        min_prob=min_prob,
        max_prob=max_prob)

    objects = filter_close_bbs(objects, pixel_distance)

    objects = rename_labels(objects, new_labels)

    # Save predicted image.
    if save_path:
        image = cv2.cvtColor(
            vis_objects(np.array(image), objects),
            cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, image)

    click.echo(' done.')
    return objects


def predict_video(network, path, only_classes=None, ignore_classes=None,
                  save_path=None, min_prob=None, max_prob=None,
                  pixel_distance=0, new_labels=None):
    if save_path:
        # We hardcode the video output to mp4 for the time being.
        save_path = os.path.splitext(save_path)[0] + '.mp4'
        try:
            writer = skvideo.io.FFmpegWriter(save_path)
        except AssertionError as e:
            tf.logging.error(e)
            tf.logging.error(
                'Please install ffmpeg before making video predictions.'
            )
            exit()
    else:
        click.echo(
            'Video not being saved. Note that for the time being, no JSON '
            'output is being generated. Did you mean to specify `--save-path`?'
        )

    num_of_frames = int(skvideo.io.ffprobe(path)['video']['@nb_frames'])

    video_progress_bar = click.progressbar(
        skvideo.io.vreader(path),
        length=num_of_frames,
        label='Predicting {}'.format(path)
    )

    objects_per_frame = []
    with video_progress_bar as bar:
        try:
            start_time = time.time()
            for idx, frame in enumerate(bar):
                # Run image through network.
                objects = network.predict_image(frame)

                # Filter the results according to the user input.
                objects = filter_classes(
                    objects,
                    only_classes=only_classes,
                    ignore_classes=ignore_classes
                )

                objects = filter_probabilities(
                    objects,
                    min_prob=min_prob,
                    max_prob=max_prob)

                objects = filter_close_bbs(objects, pixel_distance)

                objects = rename_labels(objects, new_labels)

                objects_per_frame.append({
                    'frame': idx,
                    'objects': objects
                })

                # Draw the image and write it to the video file.
                if save_path:
                    image = vis_objects(frame, objects)
                    writer.writeFrame(image)

            stop_time = time.time()
            click.echo(
                'fps: {0:.1f}'.format(num_of_frames / (stop_time - start_time))
            )
        except RuntimeError as e:
            click.echo()  # Error prints next to progress bar otherwise.
            click.echo('Error while processing {}: {}'.format(path, e))
            if save_path:
                click.echo(
                    'Partially processed video file saved in {}'.format(
                        save_path
                    )
                )

    if save_path:
        writer.close()

    return objects_per_frame


def write_xlsx(csv_path, spacing, class_labels_percentage):
    folder_path = os.path.dirname(csv_path)
    workbook = xlsxwriter.Workbook(csv_path.replace(".csv", ".xlsx"))
    worksheet = workbook.add_worksheet('sheet1')

    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:B', 10)
    temp_folder = os.path.join(folder_path, "predict_temp_bbs")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    else:
        print(
            "Path {} already exists, might be overwriting data".format(
                temp_folder))
    df = pd.read_csv(csv_path)
    rowy = 0
    for label, frac in class_labels_percentage.items():
        subset_df = df[df['label'] == label].sample(frac=frac)
        for index, row in subset_df.iterrows():
            for i in range(len(row)):
                worksheet.write(rowy * spacing, i, row[i])
            image = cv2.imread(
                row['image_id'],
                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if len(image.shape) == 3:
                image = image[
                    row.ymin: row.ymax, row.xmin: row.xmax, :]
            else:
                image = image[
                    row.ymin: row.ymax, row.xmin: row.xmax]
            temp_image = os.path.join(temp_folder, "temp_{}.png".format(rowy))
            cv2.imwrite(temp_image, image)
            worksheet.insert_image(
                rowy * spacing, i + 1,
                temp_image,
                {'x_scale': 0.3, 'y_scale': 0.3})
            rowy += 1

    workbook.close()


def get_key(new_labels, val):
    for key, value in new_labels.items():
        if val == value:
            return key


@click.command(help="Obtain a model's predictions.")
@click.argument('path-or-dir', nargs=-1)
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('output_path', '--output', '-f', default='-', help='Output file with the predictions (for example, csv bounding boxes) containing image_id,xmin,ymin,xmax,ymax,label')  # noqa
@click.option('--save-media-to', '-d', help='Directory to store media to.')
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--max-prob', default=1.0, type=float, help='When drawing, only draw bounding boxes with probability lesser than.')  # noqa
@click.option('--max-detections', default=100, type=int, help='Maximum number of detections per image.')  # noqa
@click.option('--only-class', '-k', default=None, multiple=True, help='Class to include when predicting.')  # noqa
@click.option('--ignore-class', '-K', default=None, multiple=True, help='Class to ignore when predicting.')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
@click.option('--xlsx-spacing', default=2, type=int, help='When inserting images in xlsx, space between rows')  # noqa
@click.option('--classes-json', required=False, help='path to a json file containing dictionary of class labels as keys and the float between 0 to 1 representing fraction of the rows/objects for the class to be saved in the xlsx as values')  # noqa
@click.option('--pixel-distance', type=int, required=False, help='If 2 bounding boxes are pixel-distance apart, then keep the one with highest confidence/probability and remove the other')  # noqa
@click.option('--new-labels', type=str, required=False, help='path to a json file containing dictionary of class labels as keys and the new label to replace with as value')  # noqa
def predict(path_or_dir, config_files, checkpoint, override_params,
            output_path, save_media_to, min_prob, max_prob,
            max_detections, only_class,
            ignore_class, debug, xlsx_spacing,
            classes_json, pixel_distance, new_labels):
    """Obtain a model's predictions.

    Receives either `config_files` or `checkpoint` in order to load the correct
    model. Afterwards, runs the model through the inputs specified by
    `path-or-dir`, returning predictions according to the format specified by
    `output`.

    Additional model behavior may be modified with `min-prob`, `only-class` and
    `ignore-class`.
    """
    # Read class labels as a list
    class_labels_percentage = {}
    if classes_json is not None:
        with open(classes_json, "r") as f:
            class_labels_percentage = json.load(f)
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
    files = resolve_files(path_or_dir)
    if not files:
        error = 'No files to predict found. Accepted formats are: {}.'.format(
            ', '.join(IMAGE_FORMATS + VIDEO_FORMATS)
        )
        click.echo(error)
        return
    else:
        click.echo('Found {} files to predict.'.format(len(files)))

    # Create `save_media_to` if specified and it doesn't exist.
    if save_media_to:
        tf.gfile.MakeDirs(save_media_to)

    # Resolve the config to use and initialize the model.
    if checkpoint:
        config = get_checkpoint_config(checkpoint)
    elif config_files:
        config = get_config(config_files)
    else:
        click.echo(
            'Neither checkpoint not config specified, assuming `accurate`.'
        )
        config = get_checkpoint_config('accurate')

    if override_params:
        config = override_config_params(config, override_params)

    # Filter bounding boxes according to `min_prob` and `max_detections`.
    if config.model.type == 'fasterrcnn':
        if config.model.network.with_rcnn:
            config.model.rcnn.proposals.total_max_detections = max_detections
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
    network = PredictorNetwork(config)
    # Iterate over files and run the model on each.
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    for file in files:
        # Get the media output path, if media storage is requested.
        save_path = os.path.join(
            save_media_to, 'pred_{}'.format(os.path.basename(file))
        ) if save_media_to else None

        file_type = get_file_type(file)
        predictor = predict_image if file_type == 'image' else predict_video

        objects = predictor(
            network, file,
            only_classes=only_class,
            ignore_classes=ignore_class,
            save_path=save_path,
            min_prob=min_prob,
            max_prob=max_prob,
            pixel_distance=pixel_distance,
            new_labels=new_labels
        )

    if new_labels is not None:
        with open(new_labels, "r") as f:
            new_labels = json.load(f)
    # TODO: Not writing csv for video files for now.
    unique_classes = []
    if objects is not None and file_type == 'image':
        for obj in objects:
            if new_labels is not None:
                label_name = get_key(new_labels, obj['label'])
            else:
                label_name = obj['label']
            unique_classes.append(label_name)
            df = df.append({'image_id': file,
                            'xmin': obj['bbox'][0],
                            'xmax': obj['bbox'][2],
                            'ymin': obj['bbox'][1],
                            'ymax': obj['bbox'][3],
                            'label': label_name,
                            'prob': obj["prob"]},
                           ignore_index=True)

    get_image_paths_per_class(df)
    # Build the `Formatter` based on the outputs, which automatically writes
    # the formatted output to all the requested output files.
    if output_path == '-':
        output = sys.stdout
        pd.set_option('display.max_colwidth', -1)
        output.write(df.to_string())
        output.close()
    else:
        sys.stdout.write(output_path.replace(".csv", ".txt"))
        df.to_csv(output_path)
        if class_labels_percentage == {}:
            unique_classes = list(set(unique_classes))
            class_labels_percentage = \
                {unique_class: 1.0 for unique_class in unique_classes}
        write_xlsx(output_path, xlsx_spacing, class_labels_percentage)
