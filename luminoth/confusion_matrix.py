import numpy as np
import pandas as pd
import click
import sys
import json
import sklearn.metrics

from luminoth.utils.bbox_overlap import bbox_overlap


def get_confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold):
    """
    Returns confusion matrix of shape (labels, labels)

    Args:
        groundtruth_csv: Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label"
            and several rows corresponding to the groundtruth
            bounding box objects
        predicted_csv: Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding box objects
        labels: list of names of the objects detected
        iou_threshold: IOU threshold below which the bounding box is invalid
        confidence_threshold: Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives for a class

    Returns:
        confusion_matrix: numpy array of (labels, labels) shape
    """
    number_classes = len(labels)
    df = pd.read_csv(groundtruth_csv)
    groundtruth_boxes = []
    groundtruth_classes = []
    for index, row in df.iterrows():
        groundtruth_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
        groundtruth_classes.append(row.label)

    df = pd.read_csv(predicted_csv)
    prediction_boxes = []
    prediction_classes = []
    prediction_scores = []
    for index, row in df.iterrows():
        if row.prob > confidence_threshold:
            prediction_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
            prediction_classes.append(row.label)
            prediction_scores.append(row.prob)

    matches = []

    for i in range(len(groundtruth_boxes)):
        for j in range(len(prediction_boxes)):
            iou = bbox_overlap(
                np.array(groundtruth_boxes[i]).reshape(1, 4),
                np.array(prediction_boxes[j]).reshape(1, 4))

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

    if matches.size == 0:
        return np.zeros((len(labels), len(labels)))

    matches_list = matches.tolist()
    gt_matched_classes = [
        groundtruth_classes[int(match[0])] for match in matches_list]
    predicted_matched_classes = [
        prediction_classes[int(match[1])] for match in matches_list]

    confusion_matrix = sklearn.metrics.confusion_matrix(
        gt_matched_classes, predicted_matched_classes, labels=labels)

    # Completing confusion matrix with unmatched ground truths and predictions
    # False negatives and False positives respectively
    complete_confusion_matrix = np.zeros(
        (number_classes + 1, number_classes + 1), dtype=np.uint8)
    complete_confusion_matrix[
        :number_classes, :number_classes] = confusion_matrix

    for i, label in enumerate(sorted(labels)):
        predictions_per_label = prediction_classes.count(label)
        matched_predictions_per_label = predicted_matched_classes.count(label)
        complete_confusion_matrix[i, number_classes] = \
            predictions_per_label - matched_predictions_per_label

        gts_per_label = groundtruth_classes.count(label)
        matched_gts_per_label = gt_matched_classes.count(label)

        complete_confusion_matrix[number_classes, i] = \
            gts_per_label - matched_gts_per_label

    return complete_confusion_matrix


def display(
        groundtruth_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold,
        output_path):
    """
    Save and display confusion matrix, precision, recall scores of each of
    the labels of objects detected.

    Args:
        groundtruth_csv: Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label"
            and several rows corresponding to the
            groundtruth bounding box objects
        predicted_csv: Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding
            to the predicted bounding box objects
        labels: list of names of the objects detected
            iou_threshold: float IOU threshold below which
            the bounding box is invalid
        confidence_threshold: float Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives for a class
        labels: list of names of the objects detected
        output_path: Redirect stdout to file in path
    """
    # Redirect printing output to
    stdout_origin = sys.stdout
    sys.stdout = open(output_path, "w")

    df = pd.read_csv(groundtruth_csv)
    for label in labels:
        print("There are {} {} classes in the ground truth dataset".format(
            len(df[df.label == label]), label))

    df = pd.read_csv(predicted_csv)
    print(
        "The prediction classes printed below & " +
        "considered are higher than confidence_threshold {}".format(
            confidence_threshold))
    for label in labels:
        print("There are {} {} classes in the prediction dataset".format(
            len(
                df[(df['label'] == label) & (
                    df['prob'] > confidence_threshold)]), label))

    confusion_matrix = get_confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold)

    print("Confusion matrix before normalization")
    print(confusion_matrix)
    inclusive_labels = labels + ["Unmatched"]
    inclusive_num_classes = len(inclusive_labels)

    # printing the headers with class names and parameters used
    length_name = max([len(str(s)) for s in labels] + [5])
    spacing = "- " * max(
        (int(7 + ((length_name + 3) * (inclusive_num_classes + 3)) / 2)),
        length_name + 33)
    print(spacing + "\nConfusion Matrix\n" + spacing)
    print(("confidence_threshold: %f" % confidence_threshold).rstrip("0"))
    print(
        "Where Unmatched in Groundtruth means"
        "False Positive and Unmatched in Prediction means False Negative.")
    confusion_matrix = np.uint32(confusion_matrix)
    confusion_matrix = confusion_matrix / confusion_matrix.astype(
        np.float).sum(axis=1, keepdims=True)
    content = " " * (length_name + 3 + 12)
    for j in range(inclusive_num_classes):
        content += "[%*s] " % (length_name, labels[j])
    print("%*sPrediction" % (12 + (len(content) - 10) // 2, ""))
    print(content)

    # printing the normalized confusion matrix elements
    for i in range(inclusive_num_classes):
        content = "Groundtruth " if i == int(
            (inclusive_num_classes) / 2) else " " * 12
        content += "[%*s] " % (length_name, labels[i])
        for j in range(inclusive_num_classes):
            content += "%*f " % (length_name + 2, confusion_matrix[i, j])
        print(content)

    print(spacing)
    results = []

    # Print precision and recall
    for i in range(len(labels)):
        name = labels[i]

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
            {'label': name,
             'precision_@{}IOU'.format(iou_threshold): precision,
             'recall_@{}IOU'.format(iou_threshold): recall})

    # Close STDOUT and reset
    sys.stdout.close()
    sys.stdout = stdout_origin


@click.command(help="Save or print confusion matrix per class after comparing ground truth and prediced bounding boxes")  # noqa
@click.option("--groundtruth_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label and several rows corresponding to the groundtruth bounding box objects", required=True, type=str) # noqa
@click.option("--predicted_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label,prob and several rows corresponding to the predicted bounding box objects", required=True, type=str) # noqa
@click.option("--output_txt", help="output txt file containing confusion matrix, precision, recall per class", type=str) # noqa
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
    with open(classes_json, "r") as f:
        class_labels = json.load(f)
    display(
        groundtruth_csv,
        predicted_csv,
        class_labels,
        iou_threshold,
        confidence_threshold,
        output_txt)


if __name__ == '__main__':
    confusion_matrix()
