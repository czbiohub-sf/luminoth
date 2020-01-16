import numpy as np
import click
import itertools
import os
import sys
import json
import sklearn.metrics
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from luminoth.utils.bbox_overlap import bbox_overlap
from luminoth.utils.overlay_bbs import add_base_path


DEFAULT_CMAP = "Oranges"
DEFAULT_FONT_SIZE = 11
DEFAULT_LINE_WIDTH = 0.5
DEFAULT_FIG_SIZE = (9, 9)
NUM_CPUS = 4

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

""" # noqa


def get_valid_match_iou(i, j, gt_boxes, predicted_boxes, iou_threshold):
    """
    Returns all groundtruth classes, predicted classes, matched groundtruth
    classes, matched predicted classes with iou greater than iou_threshold,
    after filtering the predicted classes with confidence greater than
    confidence_threshold for one image given in image_path with format
    input_image_format

    Args:
        i: image_id to look for in the dataframes
        j: str Format of the image_id file in the csv files
            for groundtruth and prediction
        gt_boxes: dataframe with image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present. image_id
            should be same for all rows
        predicted_boxes:
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
    Returns:
        list: [index_gt, index_predicted, iou] or None
    """
    iou = bbox_overlap(
        np.array(gt_boxes[i]).reshape(1, 4),
        np.array(predicted_boxes[j]).reshape(1, 4))[0][0]
    if iou >= iou_threshold:
        return [i, j, iou]


def get_matched_gt_predict_per_image(
        im_path,
        input_image_format,
        df_gt,
        df_predicted,
        labels,
        iou_threshold,
        confidence_threshold,
        num_cpus):
    """
    Returns all groundtruth classes, predicted classes, matched groundtruth
    classes, matched predicted classes with iou greater than iou_threshold,
    after filtering the predicted classes with confidence greater than
    confidence_threshold for one image given in image_path with format
    input_image_format

    Args:
        im_path: image_id to look for in the dataframes
        input_image_format: str Format of the image_id file in the csv files
            for groundtruth and prediction
        df_gt: dataframe with image_id,xmin,ymin,xmax,ymax,label as header
            and several rows corresponding to the groundtruth
            bounding boxes, labels, image in which they are present. image_id
            should be same for all rows
        df_predicted: str Absolute path to csv
            containing image_id,xmin,ymin,xmax,ymax,label,prob"
            and several rows corresponding to the predicted
            bounding boxes, labels, image in which they are present. image_id
            should be same for all rows
        labels: list of unqiue names of the objects present
        iou_threshold: float, IOU threshold below which the
            match of the predicted bounding box with the
            ground truth box is invalid
        confidence_threshold: flot Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data
        num_cpus: int number of cpus to run comparison between groundtruth
            and predicted to obtain matched classses

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
    basename = os.path.basename(im_path).replace(input_image_format, "")
    df_gt = df_gt[df_predicted.base_path == basename]
    df_predicted = df_predicted[df_predicted.base_path == basename]
    gt_boxes = []
    gt_classes = []
    for index, row in df_gt.iterrows():
        box = [row.xmin, row.ymin, row.xmax, row.ymax]
        if box in gt_boxes:
            row_index = gt_boxes.index(box)
            gt_classes.insert(row_index, row.label)
        else:
            gt_boxes.append(box)
            gt_classes.append(row.label)

    # Collect the bboxes, labels in 2 lists for predicted only if their
    # prob is greater than confidence_threshold
    predicted_boxes = []
    predicted_classes = []
    predicted_scores = []
    for index, row in df_predicted.iterrows():
        prob = row.prob
        if prob >= confidence_threshold:
            box = [row.xmin, row.ymin, row.xmax, row.ymax]
            if box in predicted_boxes:
                row_index = predicted_boxes.index(box)
                previous_prob = df_predicted.iloc[row_index]['prob']
                if prob > previous_prob:
                    predicted_classes.insert(row_index, row.label)
                    predicted_scores.insert(row_index, row.prob)
            else:
                predicted_boxes.append(box)
                predicted_classes.append(row.label)
                predicted_scores.append(row.prob)

    # Find IOU Matches

    iterator = itertools.product(
        range(len(gt_boxes)), range(len(predicted_boxes)))
    matches = Parallel(
        n_jobs=num_cpus)(
        delayed(
            get_valid_match_iou)(
            i,
            j,
            gt_boxes,
            predicted_boxes,
            iou_threshold) for i,
        j in iterator)
    matches_before = list(filter(None, matches))
    # Remove redundant IOU matches with different labels
    matches = np.array(matches_before)
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
    return [
        gt_classes,
        predicted_classes,
        gt_matched_classes,
        predicted_matched_classes]


def get_matched_gt_predict(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold,
        input_image_format,
        num_cpus):
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
        input_image_format: str Format of the image_id file in the csv files
            for groundtruth and prediction
        num_cpus: int number of cpus to run comparison between groundtruth
            and predicted to obtain matched classses

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
    gt_df = add_base_path(gt_csv, input_image_format)
    predicted_df = add_base_path(predicted_csv, input_image_format)
    unique_image_paths = gt_df['base_path'].unique().tolist()

    result = Parallel(
        n_jobs=num_cpus)(
        delayed(
            get_matched_gt_predict_per_image)(
            im_path,
            input_image_format,
            gt_df,
            predicted_df,
            labels,
            iou_threshold,
            confidence_threshold,
            num_cpus) for im_path in unique_image_paths)
    gt_classes = []
    gt_matched_classes = []
    predicted_classes = []
    predicted_matched_classes = []
    for per_image_result in result:
        gt_classes_image = per_image_result[0]
        predicted_classes_image = per_image_result[1]
        gt_matched_classes_image = per_image_result[2]
        predicted_matched_classes_image = per_image_result[3]
        for item0 in gt_classes_image:
            gt_classes.append(item0)
        for item1 in predicted_classes_image:
            predicted_classes.append(item1)
        for item2 in gt_matched_classes_image:
            gt_matched_classes.append(item2)
        for item3 in predicted_matched_classes_image:
            predicted_matched_classes.append(item3)
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
        (number_classes + 2, number_classes + 2), dtype=np.uint64)
    complete_confusion_matrix[
        :number_classes, :number_classes] = confusion_matrix

    # Set labels,labels + 1 rows,columns in the confusion matrix for each class
    for i, label in enumerate(sorted(labels)):
        predicteds_per_label = predicted_classes.count(label)
        matched_predicteds_per_label = predicted_matched_classes.count(label)

        gts_per_label = gt_classes.count(label)
        matched_gts_per_label = gt_matched_classes.count(label)

        # Number of unmatched groundtruth objects are set at labels row
        complete_confusion_matrix[i, number_classes] = \
            gts_per_label - matched_gts_per_label
        # Number of unmatched predicted objects are set at labels column
        complete_confusion_matrix[number_classes, i] = \
            predicteds_per_label - matched_predicteds_per_label

        # Number of total groundtruth objects are set at labels + 1 row
        complete_confusion_matrix[i, number_classes + 1] = gts_per_label
        # Number of total predicted objects are set at labels + 1 column
        complete_confusion_matrix[number_classes + 1, i] = predicteds_per_label
    return complete_confusion_matrix


def get_confusion_matrix(
        gt_csv,
        predicted_csv,
        labels,
        iou_threshold,
        confidence_threshold,
        input_image_format,
        num_cpus):
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
        confidence_threshold: float Confidence score threshold below which
            bounding box detection is of low confidence and
            is ignored while considering true positives in predicted data
        input_image_format: str Format of the image_id file in the csv files
            for groundtruth and prediction
        num_cpus: int number of cpus to run comparison between groundtruth
            and predicted to obtain matched classses

    Returns:
        complete_confusion_matrix: numpy array of
        (len(labels) + 2, len(labels) + 2) shape,
        includes total groundtruth, predicted per class in the last row, column
        and umatched groundtruth, unmatched predicted per class
        in the last row - 1, column - 1 respectively. Look at the text in the
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
        confidence_threshold,
        input_image_format,
        num_cpus)

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
        output_path,
        output_fig,
        input_image_format,
        num_cpus):
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
        output_path: str output txt file containing confusion matrix,
            precision, recall per class
        output_fig: str output figure file containing confusion matrix,
            precision, recall per class. Format of the figure file could be
            png, svg, eps, or pdf.
        input_image_format: str Format of the image_id file in the csv files
            for groundtruth and prediction
        num_cpus: int number of cpus to run comparison between groundtruth
            and predicted to obtain matched classses

    Returns:
        Prints confusion matrix, normalized confusion matrix
        precision, recall, f1_score, support per class,
        either to stdout or to a text file as
        specified in output_path, also saves a plotted confusion matrix image
        in output_fig. the format of the figure is similar to figure returned
        by matlab's plotconfusion function
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
        confidence_threshold,
        input_image_format,
        num_cpus)

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

    # set total at the last element to the sum of all predicted elements(tp+fp)
    confusion_matrix[-1, -1] = np.sum(confusion_matrix[-1, :])
    # remove unmatched row, column
    confusion_matrix = np.delete(confusion_matrix, -2, 0)
    confusion_matrix = np.delete(confusion_matrix, -2, 1)
    # Plot confusion matrix
    plot_cm(confusion_matrix, labels, output_fig)

    # Close STDOUT and reset
    sys.stdout.close()
    sys.stdout = stdout_origin


def format_element_to_matlab_confusion_matrix(row, col, confusion_matrix):
    """
    Return a string for the element on row, col location for
    confusion_matrix to either
    number of observation\npercentage of observations or
    percentage_correct_classifications\npercentage_incorrect_classifications
    per class

    Args:
        row: int row location inside array
        col: int col locat
        confusion_matrix: numpy array of symmetrical (x, x) shape with target
            class/groundtruth in columns and output/predicted class in rows and
            total for each class including unmatched groundtruth
            (not detected by the model) & unmatched predicted class(due to low
            confidence)

    Returns:
        text: str string depending on row, col location for
            element in the confusion_matrix to either
            number of observation\npercentage of observations or
            percentage_correct_classifications\npercentage_incorrect
            classifications per class.
            If row, col are equal to the last row, col of the array then the
            element represents totals per class,
            percentage_correct_classifications\npercentage_incorrect
            classifications per class is returned else
            it is the number of objects per label in either ground truth or
            predicted and number of observation\npercentage of observations is
            returned

    """
    current_element = confusion_matrix[row][col]
    total_predicted = confusion_matrix[-1][-1]
    percentage_total = (float(current_element) / total_predicted) * 100
    cm_length = confusion_matrix.shape[0]

    # for totals calculate percentage accuracy and percentage error
    if(col == (cm_length - 1)) or (row == (cm_length - 1)):
        # totals and percents
        if(current_element != 0):
            if(col == cm_length - 1) and (row == cm_length - 1):
                total_correct = 0
                for i in range(confusion_matrix.shape[0] - 1):
                    total_correct += confusion_matrix[i][i]
                percentage_correct_classifications = (
                    float(total_correct) / current_element) * 100
            elif(col == cm_length - 1):
                true_positives_for_label = confusion_matrix[row][row]
                true_predicted_per_class = current_element
                percentage_correct_classifications = (
                    float(true_positives_for_label) / true_predicted_per_class
                ) * 100
            elif(row == cm_length - 1):
                true_positives_for_label = confusion_matrix[col][col]
                true_groundtruth_per_class = current_element
                percentage_correct_classifications = (
                    float(
                        true_positives_for_label) / true_groundtruth_per_class
                ) * 100
            percentage_incorrect_classifications = \
                100 - percentage_correct_classifications
        else:
            percentage_correct_classifications = \
                percentage_incorrect_classifications = 0

        percentage_correct_classifications_s = [
            '%.2f%%' % (percentage_correct_classifications), '100%'][
            percentage_correct_classifications == 100]
        txt = '%s\n%.2f%%' % (
            percentage_correct_classifications_s,
            percentage_incorrect_classifications)
    else:
        if(percentage_total > 0):
            txt = '%s\n%.2f%%' % (current_element, percentage_total)
        else:
            txt = '0\n0.0%'
    return txt


def plot_cm(confusion_matrix, labels, output_fig):
    """
    Save confusion matrix, precision, recall scores of each of
    the unique labels to a figure

    Args:
        confusion_matrix: numpy array of symmetrical (x, x) shape,
        along the columns is predicted data
        labels: list of unqiue names of the objects present
        output_fig: str output figure file containing confusion matrix,
            precision, recall per class (format can be png, pdf, eps, svg)

    Returns:
        Plots and saves matlab like confusion matrix with
        precision, recall, and total percentages to output_fig
    The rows correspond to the predicted class (Output Class) and
    the columns correspond to the true class (Target Class).
    The diagonal cells correspond to observations that are correctly
    classified. The off-diagonal cells correspond to incorrectly
    classified observations. Both the number of observations and the
    percentage of the total number of observations are shown in each cell.

    The column on the far right of the plot shows the percentages of all
    the examples predicted to belong to each class that are correctly and
    incorrectly classified. These metrics are often called the precision
    (or positive predictive value) and false discovery rate, respectively.
    The row at the bottom of the plot shows the percentages of all the examples
    belonging to each class that are correctly and incorrectly classified.
    These metrics are often called the recall
    (or true positive rate or sensitivity)
    and false negative rate, respectively.
    The cell in the bottom right of the plot shows the overall accuracy.
    """
    # Transpose to set the ground truth to be along columns
    confusion_matrix = confusion_matrix.T
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=DEFAULT_CMAP)

    # set ticklabels rotation
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Loop over data dimensions and create text annotations.
    cm_length = confusion_matrix.shape[0]
    for row in range(cm_length):
        for col in range(cm_length):
            text = ax.text(
                row, col,
                format_element_to_matlab_confusion_matrix(
                    row, col, confusion_matrix),
                ha="center", va="center", color="black", fontsize=8)

    # Turn spines off and create black grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(confusion_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(confusion_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Titles and labels
    ax.set_title('Confusion matrix', fontweight='bold')
    ax.set_xlabel('Output Class', fontweight='bold')
    ax.set_ylabel('Target Class', fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    del im, text


@click.command(help="Save or print confusion matrix per class after comparing ground truth and prediced bounding boxes")  # noqa
@click.option("--groundtruth_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label and several rows corresponding to the groundtruth bounding box objects", required=True, type=str) # noqa
@click.option("--predicted_csv", help="Absolute path to csv containing image_id,xmin,ymin,xmax,ymax,label,prob and several rows corresponding to the predicted bounding box objects", required=True, type=str) # noqa
@click.option("--input_image_format", help="Format of images in image_id column in the csvs", required=False, type=str, default=".jpg") # noqa
@click.option("--output_txt", help="output txt file containing confusion matrix, precision, recall per class", type=str) # noqa
@click.option("--output_fig", help="output fig file (format can be png, eps, pdf, svg) containing confusion matrix, precision, recall per class", type=str) # noqa
@click.option('--iou_threshold', type=float, required=False, default=0.5, help='IOU threshold below which the bounding box is invalid')  # noqa
@click.option('--confidence_threshold', type=float, required=False, default=0.5, help='Confidence score threshold below which bounding box detection is of low confidence and is ignored while considering true positives')  # noqa
@click.option('--classes_json', required=True, help='path to a json file containing list of class label for the objects, labels are alphabetically sorted')  # noqa
@click.option('--num_cpus', required=False, default=NUM_CPUS, type=int, help='number of cpus to run comparison between groundtruth and predicted to obtain matched classses')  # noqa
def confusion_matrix(
        groundtruth_csv,
        predicted_csv,
        output_txt,
        output_fig,
        iou_threshold,
        confidence_threshold,
        classes_json,
        num_cpus,
        input_image_format):
    # Read class labels as a list
    with open(classes_json, "r") as f:
        class_labels = json.load(f)
    display(
        groundtruth_csv,
        predicted_csv,
        class_labels,
        iou_threshold,
        confidence_threshold,
        output_txt,
        output_fig,
        input_image_format,
        num_cpus
    )


if __name__ == '__main__':
    confusion_matrix()
