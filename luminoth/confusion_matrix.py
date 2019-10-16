import numpy as np
import pandas as pd
import click
import sys
import json


def compute_iou(groundtruth_box, detection_box):
    """
    Return a float intersection pixel area of two
    bounding boxes divided by the union of the two

    :param list groundtruth_box: xmin, g_xmax, ymin, ymax
    x_leftmost_corner, x_rightmost_corner,y_leftmost_corner,y_rightmost_corner
    :param list detection_box: xmin, g_xmax, ymin, ymax
    x_leftmost_corner, x_rightmost_corner,y_leftmost_corner,y_rightmost_corner
    :return float iou intersection pixel area of two
    bounding boxes divided by the union of the two
    """
    if groundtruth_box == detection_box:
        return 1.0
    g_xmin, g_xmax, g_ymin, g_ymax = tuple(groundtruth_box)
    d_xmin, d_xmax, d_ymin, d_ymax = tuple(detection_box)

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)
    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    if boxAArea + boxBArea == intersection:
        return 1.0

    iou = float(intersection) / float(boxAArea + boxBArea - intersection)
    return iou


def get_confusion_matrix(
        groundtruth_csv, predicted_csv,
        categories, iou_threshold, confidence_threshold):
    """
    Returns confusion matrix of shape (categories, categories)

    :param str groundtruth_csv: Absolute path to csv
    containing image_id,xmin,ymin,xmax,ymax,label"
    and several rows corresponding to the groundtruth bounding box objects
    :param str predicted_csv: Absolute path to csv
    containing image_id,xmin,ymin,xmax,ymax,label,prob"
    and several rows corresponding to the predicted bounding box objects
    :param list categories: list of names of the objects detected
    :param iou_threshold IOU threshold below which the bounding box is invalid
    :param confidence_threshold
    :return confusion_matrix numpy array of (categories, categories) shape
    """
    confusion_matrix = np.zeros(shape=(len(categories), len(categories)))

    df = pd.read_csv(groundtruth_csv)

    groundtruth_boxes = []
    groundtruth_classes = []
    for index, row in df.iterrows():
        groundtruth_boxes.append([row.xmin, row.xmax, row.ymin, row.ymax])
        groundtruth_classes.append(categories.index(row.label))

    df = pd.read_csv(predicted_csv)
    detection_boxes = []
    detection_classes = []
    detection_scores = []
    for index, row in df.iterrows():
        if row.prob > confidence_threshold:
            detection_boxes.append([row.xmin, row.xmax, row.ymin, row.ymax])
            detection_classes.append(categories.index(row.label))
            detection_scores.append(row.prob)

    matches = []

    for i in range(len(groundtruth_boxes)):
        for j in range(len(detection_boxes)):
            iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

            if iou > iou_threshold:
                matches.append([i, j, iou])

    matches = np.array(matches)
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU
        # so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU.
        # Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    for i in range(len(groundtruth_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
            confusion_matrix[
                groundtruth_classes[i]][
                detection_classes[
                    int(matches[matches[:, 0] == i, 1][0])]] += 1
        else:
            confusion_matrix[
                groundtruth_classes[i]][
                    confusion_matrix.shape[1] - 1] += 1

    for i in range(len(detection_boxes)):
        if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
            confusion_matrix[
                confusion_matrix.shape[0] - 1][
                    detection_classes[i]] += 1

    return confusion_matrix


def display(
        confusion_matrix, categories,
        iou_threshold, confidence_threshold,
        output_path):
    """
    Save and display confusion matrix, precision, recall scores of each of
    the categories of objects detected.

    :param np.array confusion_matrix:
    confusion matrix of length (categories, categories)
    :param list categories: list of names of the objects detected
    :param str output_path: Redirect stdout to file in paht
    """
    # Redirect printing output to
    stdout_origin = sys.stdout
    sys.stdout = open(output_path, "w")
    number_classes = len(categories)

    # printing the headers with class names and parameters used
    length_name = max([len(str(s)) for s in categories] + [5])
    spacing = "- " * max(
        (int(7 + ((length_name + 3) * (number_classes + 3)) / 2)),
        length_name + 33)
    print(spacing + "\nConfusion Matrix\n" + spacing)
    print(("thresh_confidence: %f" % confidence_threshold).rstrip("0"))
    confusion_matrix = np.uint32(confusion_matrix)
    confusion_matrix /= confusion_matrix.astype(np.float).sum(
        axis=1, keepdims=True)
    content = " " * (length_name + 3 + 12)
    for j in range(number_classes):
        content += "[%*s] " % (length_name, categories[j])
    print("%*sPrediction" % (12 + (len(content) - 10) // 2, ""))
    print(content)

    # printing the normalized confusion matrix elements
    for i in range(number_classes):
        content = "Groundtruth " if i == int(
            (number_classes) / 2) else " " * 12
        content += "[%*s] " % (length_name, categories[i])
        for j in range(number_classes):
            content += "%*f " % (length_name + 2, confusion_matrix[i, j])
        print(content)

    print(spacing)
    results = []

    # Print precision and recall
    for i in range(len(categories)):
        name = categories[i]

        total_target = np.sum(confusion_matrix[i, :])
        total_predicted = np.sum(confusion_matrix[:, i])

        precision = float(confusion_matrix[i, i] / total_predicted)
        recall = float(confusion_matrix[i, i] / total_target)

        print(
            'precision_{}@{}IOU: {:.2f}'.format(
                name, iou_threshold, precision))
        print(
            'recall_{}@{}IOU: {:.2f}'.format(name, iou_threshold, recall))

        results.append(
            {'category': name,
             'precision_@{}IOU'.format(iou_threshold): precision,
             'recall_@{}IOU'.format(iou_threshold): recall})

    # Close STDOUT and reset
    sys.stdout.close()
    sys.stdout = stdout_origin


@click.command(help="Save or print confusion matrix per class after comparing ground truth and prediced bounding boxes")  # noqa
@click.option("--groundtruth_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label and several rows corresponding to the groundtruth bounding box objects", required=True, type=str) # noqa
@click.option("--predicted_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label,prob and several rows corresponding to the predicted bounding box objects", required=True, type=str) # noqa
@click.option("--output_txt", help="output txt file containing confusion matrix, precision, recall per class", required=True, type=str) # noqa
@click.option('--iou_threshold', type=float, default=0.5, help='IOU threshold below which the bounding box is invalid')  # noqa
@click.option('--confidence_threshold', type=float, default=0.9, help='Confidence score threshold below which bounding box detection is of low confidence and is ignored while considering true positives')  # noqa
@click.option('--classes_json', required=True, help='path to a json file containing list of class label for the objects')  # noqa
def confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        output_txt,
        iou_threshold,
        confidence_threshold,
        classes_json):
    # Attempt to get class names, if available.
    with open(classes_json) as f:
        class_labels = json.loads(f)
    confusion_matrix = get_confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        class_labels,
        iou_threshold,
        confidence_threshold)

    display(
        confusion_matrix,
        class_labels,
        iou_threshold,
        confidence_threshold,
        output_txt)


if __name__ == '__main__':
    confusion_matrix()
