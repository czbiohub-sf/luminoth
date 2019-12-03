import numpy as np
import pandas as pd
import click
import sys
import json
import sklearn.metrics

from luminoth.utils.bbox_overlap import bbox_overlap


"""
Output from this cli tool would as below:
lumi confusion_matrix --groundtruth_csv val.csv --predicted_csv preds_val.csv --classes_json classes.json

Confusion matrix before normalization

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
confidence_threshold: 0.9
Where Unmatched in Groundtruth means False Positive and Unmatched in Prediction means False Negative.
                                                 Prediction
                        [        0] [        1] [        2] [Unmatched] [    Total] 
            [        0]           1           0           0           0           1 
            [        1]           0           0           1           3           3 
Groundtruth [        2]           0           0           2           0           3 
            [Unmatched]           1           1           0           0           0 
            [    Total]           2           2           2           0           0 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Confusion matrix after normalization

Note: Normalized by number of elements in each class in all groundtruth, includes both matched and unmatched
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
confidence_threshold: 0.9
Where Unmatched in Groundtruth means False Positive and Unmatched in Prediction means False Negative.
                             Prediction
                    [    0] [    1] [    2] 
            [    0]   0.500   0.000   0.000 
Groundtruth [    1]   0.000   0.000   0.500 
            [    2]   0.000   0.000   1.000 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Note: Precision and recall here doesnt include the bounding boxes that were present in ground truth but not detected in predicted and viceversa
             precision    recall  f1-score   support

        0.0       1.00      1.00      1.00         1
        1.0       0.00      0.00      0.00         1
        2.0       0.67      1.00      0.80         2

avg / total       0.58      0.75      0.65         4

""" # noqa


def get_matched_gt_predict(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold):
    """
    Returns all groundtruth classes, predicted classes, matched groundtruth
    classes, matched predicted classes with iou greater than iou_threshold,
    after filtering the predicted classes with confidence greater than
    confidence_threshold

    Args:
        gt_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present
        predicted_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding boxes, labels, image in which they are present
        labels: list of unqiue names of the objects present
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
        confidence_threshold: flot Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data

    Returns:
        gt_classes: list of all the annotations/labels in groundtruth
        predicted_classes: list of all the annotation/labels in predicted
        gt_matched_classes: list of gt bounding boxs annotations that
            match the predicted of greater than confidence_threshold with
            greater than iou_threshold
        predicted_matched_classes: list of predicted bounding box annotations
            greater than confidence_threshold annotations
            that match the groundtruth with greater than iou_threshold
    """
    # Collect the bboxes, labels in 2 lists for ground truth
    df = pd.read_csv(gt_csv)
    gt_boxes = []
    gt_classes = []
    for index, row in df.iterrows():
        gt_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
        gt_classes.append(row.label)

    # Collect the bboxes, labels in 2 lists for predicted only if their
    # prob is greater than confidence_threshold
    df = pd.read_csv(predicted_csv)
    predicted_boxes = []
    predicted_classes = []
    predicted_scores = []
    for index, row in df.iterrows():
        if row.prob > confidence_threshold:
            predicted_boxes.append([row.xmin, row.ymin, row.xmax, row.ymax])
            predicted_classes.append(row.label)
            predicted_scores.append(row.prob)

    # Find IOU Matches
    matches = []

    for i in range(len(gt_boxes)):
        for j in range(len(predicted_boxes)):
            iou = bbox_overlap(
                np.array(gt_boxes[i]).reshape(1, 4),
                np.array(predicted_boxes[j]).reshape(1, 4))

            if iou > iou_threshold:
                matches.append([i, j, iou])

    # Remove redundant IOU matches with different labels
    matches = np.array(matches)
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU
        # so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
        # Remove duplicate predictions for the same bbox from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU.
        # Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate groundtruths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    matches_list = matches.tolist()

    # Get the list of gt classses that matched with prediction
    gt_matched_classes = [
        gt_classes[int(match[0])] for match in matches_list]
    # Get the list of predicted classses that matched with ground truth
    predicted_matched_classes = [
        predicted_classes[int(match[1])] for match in matches_list]
    return (
        gt_classes,
        predicted_classes,
        gt_matched_classes,
        predicted_matched_classes)


def append_unmatched_gt_predict(
        confusion_matrix,
        labels,
        gt_matched_classes,
        predicted_matched_classes,
        gt_classes,
        predicted_classes):
    """
    Returns confusion matrix of shape (len(labels) + 2, len(labels) + 2)

    Args:
        confusion_matrix: numpy array of (labels, labels) shape
        labels: list of unique names of the objects detected
        gt_matched_classes: list of gt bounding boxs annotations that
            match the predicted of greater than confidence_threshold with
            greater than iou_threshold
        predicted_matched_classes: list of predicted bounding box annotations
            greater than confidence_threshold annotations
            that match the groundtruth with greater than iou_threshold
        gt_classes: list of all the annotations/labels in groundtruth
        predicted_classes: list of all the annotation/labels in predicted

    Returns:
        complete_confusion_matrix: numpy array of
        (len(labels) + 2, len(labels) + 2) shape,
        includes total groundtruth, predicted per class in the last row, column
        and umatched groundtruth, unmatched predicted per class
        in the last row - 1, column -1 respectively. Look at the text in the
        beginning of the program to understand by example
    """

    # Allocate & set first labels-1,labels-1 row, colums as confusion matrix
    number_classes = len(labels)
    complete_confusion_matrix = np.zeros(
        (number_classes + 2, number_classes + 2), dtype=np.uint32)
    complete_confusion_matrix[
        :number_classes, :number_classes] = confusion_matrix

    # Set labels,labels + 1 rows,columns in the confusion matrix for each class
    for i, label in enumerate(sorted(labels)):
        predicteds_per_label = predicted_classes.count(label)
        matched_predicteds_per_label = predicted_matched_classes.count(label)

        gts_per_label = gt_classes.count(label)
        matched_gts_per_label = gt_matched_classes.count(label)

        # Number of unmatched predicted objects are set at labels column
        complete_confusion_matrix[i, number_classes] = \
            predicteds_per_label - matched_predicteds_per_label
        # Number of unmatched groundtruth objects are set at labels row
        complete_confusion_matrix[number_classes, i] = \
            gts_per_label - matched_gts_per_label

        # Number of total predicted objects are set at labels + 1 column
        complete_confusion_matrix[i, number_classes + 1] = predicteds_per_label
        # Number of total groundtruth objects are set at labels + 1 row
        complete_confusion_matrix[number_classes + 1, i] = gts_per_label

    return complete_confusion_matrix


def get_confusion_matrix(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold):
    """
    Returns confusion matrix of shape (len(labels) + 2, len(labels) + 2)

    Args:
        gt_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present
        predicted_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding boxes, labels, image in which they are present
        labels: list of unqiue names of the objects present
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
        confidence_threshold: flot Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data

    Returns:
        complete_confusion_matrix: numpy array of
        (len(labels) + 2, len(labels) + 2) shape,
        includes total groundtruth, predicted per class in the last row, column
        and umatched groundtruth, unmatched predicted per class
        in the last row - 1, column -1 respectively. Look at the text in the
        beginning of the program to understand by example
    """
    (gt_classes,
     predicted_classes,
     gt_matched_classes,
     predicted_matched_classes) = get_matched_gt_predict(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold)

    confusion_matrix = np.zeros(
        (len(labels), len(labels)), dtype=np.uint32)
    if gt_matched_classes != [] and predicted_matched_classes != []:
        confusion_matrix = sklearn.metrics.confusion_matrix(
            gt_matched_classes, predicted_matched_classes, labels=labels)

    # Completing confusion matrix with unmatched ground truths and predicteds
    # False negatives and False positives respectively
    complete_confusion_matrix = append_unmatched_gt_predict(
        confusion_matrix,
        labels,
        gt_matched_classes,
        predicted_matched_classes,
        gt_classes,
        predicted_classes)

    return complete_confusion_matrix


def print_cm(confusion_matrix, labels, confidence_threshold):
    """
    Prints confusion matrix

    Args:
        confusion_matrix: numpy array of symmetrical (x, x) shape
        labels: list of names of the symmetrical headers for rows, columns
        confidence_threshold: Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives for a class

    Returns:
        Prints confusion matrix with class headers, groundtruth, predicted
        headers. Look at the text in the
        beginning of the program to understand by example
    """
    num_classes = len(labels)
    data_type = confusion_matrix.dtype
    length_name = max([len(str(s)) for s in labels] + [5])
    spacing = "- " * max(
        (int(7 + ((length_name + 3) * (num_classes + 3)) / 2)),
        length_name + 33)
    print(spacing)
    print(("confidence_threshold: %f" % confidence_threshold).rstrip("0"))
    print(
        "Where Unmatched in Groundtruth means "
        "False Positive and Unmatched in Prediction means False Negative.")
    content = " " * (length_name + 3 + 12)
    for j in range(num_classes):
        content += "[%*s] " % (length_name, labels[j])
    print("%*sPrediction" % (12 + (len(content) - 10) // 2, ""))
    print(content)

    for i in range(num_classes):
        content = "Groundtruth " if i == int(
            (num_classes) / 2) else " " * 12
        content += "[%*s] " % (length_name, labels[i])
        for j in range(num_classes):
            if data_type == np.dtype(float).type:
                content += "%*.3f " % (length_name + 2, confusion_matrix[i, j])
            else:
                content += "%*.d " % (length_name + 2, confusion_matrix[i, j])
        print(content)

    print(spacing)


def print_precision_recall(
        groundtruth_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold):
    """
    Print precision, recall scores of each of the labels of objects detected.
    Note: This doesn't include the bounding boxes that were present in
    ground truth but not detected in predicted and viceversa

    Args:
        gt_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present
        predicted_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding boxes, labels, image in which they are present
        labels: list of unqiue names of the objects present
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
        confidence_threshold: flot Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data

    Returns:
        Prints precision, recall, f1_score, support per class (number of
        elements in each class in groundtruth that were matched with
        prediction)
    """
    # TODO PV Calculate actual precision, recall, & remove the below note
    print(
        "Note: Precision and recall here doesnt include the bounding boxes "
        "that were present in ground truth but not detected in predicted "
        "and viceversa")

    # Getting list of matched GT, Predicted classes
    (gt_classes,
     predicted_classes,
     gt_matched_classes,
     predicted_matched_classes) = get_matched_gt_predict(
        groundtruth_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold)
    # Using sklearn to get precision, recall etc
    print(sklearn.metrics.classification_report(
        gt_matched_classes,
        predicted_matched_classes))


def normalize_confusion_matrix(confusion_matrix):
    """
    Returns normalized confusion matrix of shape (len(labels), len(labels))
    after normalizing with the total number of elements in each class in
    groundtruth, includes both matched and unmatched

    Args:
        confusion_matrix: numpy array of (len(labels) + 2, len(labels) + 2)
        shape,
        includes total groundtruth, predicted per class in the last row, column
        respectively

    Returns:
        normalized_confusion_matrix: numpy array of shape
        (len(labels), len(labels)) after normalizing the detections with
        total groundtruth per class
    """

    total_gt = confusion_matrix[-1, :-2]
    confusion_matrix = confusion_matrix[:-2, :-2]
    confusion_matrix = confusion_matrix.astype(np.float64)
    total_gt = total_gt.astype(np.float64)
    normalized_confusion_matrix = confusion_matrix / total_gt
    assert all(
        i <= 1.0 for i in normalized_confusion_matrix.flatten().tolist()
    ) is True
    return normalized_confusion_matrix


def display(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold,
        output_path):
    """
    Save and display confusion matrix, precision, recall scores of each of
    the unique labels

    Args:
        gt_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present
        predicted_csv: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding boxes, labels, image in which they are present
        labels: list of unqiue names of the objects present
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
        confidence_threshold: flot Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data

    Returns:
        Prints confusion matrix, normalized confusion matrix
        precision, recall, f1_score, support per class,
        either to stdout or to a text file as
        specified in output_path
    """
    # Redirect printing output to
    stdout_origin = sys.stdout
    if output_path:
        sys.stdout = open(output_path, "w")

    confusion_matrix = get_confusion_matrix(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold)

    print("Confusion matrix before normalization\n")
    inclusive_labels = labels + ["Unmatched", "Total"]
    print_cm(confusion_matrix, inclusive_labels, confidence_threshold)

    print("Confusion matrix after normalization\n")
    print(
        "Note: Normalized by number of elements in each class in all " +
        "groundtruth, includes both matched and unmatched")
    normalized_confusion_matrix = normalize_confusion_matrix(
        confusion_matrix)
    print_cm(normalized_confusion_matrix, labels, confidence_threshold)

    print_precision_recall(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold)

    # Close STDOUT and reset
    sys.stdout.close()
    sys.stdout = stdout_origin


@click.command(help="Save or print confusion matrix per class after comparing ground truth and prediced bounding boxes")  # noqa
@click.option("--groundtruth_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label and several rows corresponding to the groundtruth bounding box objects", required=True, type=str) # noqa
@click.option("--predicted_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label,prob and several rows corresponding to the predicted bounding box objects", required=True, type=str) # noqa
@click.option("--output_txt", help="output txt file containing confusion matrix, precision, recall per class", type=str) # noqa
@click.option('--iou_threshold', type=float, required=False, default=0.5, help='IOU threshold below which the bounding box is invalid')  # noqa
@click.option('--confidence_threshold', type=float, required=False, default=0.9, help='Confidence score threshold below which bounding box detection is of low confidence and is ignored while considering true positives')  # noqa
@click.option('--classes_json', required=True, help='path to a json file containing list of class label for the objects, labels are alphabetically sorted')  # noqa
def confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        output_txt,
        iou_threshold,
        confidence_threshold,
        classes_json):
    # Read class labels as a list
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
