import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

import luminoth.confusion_matrix as confusion_matrix
from luminoth.predict import LUMI_CSV_COLUMNS


class ConfusionMatrixTest(tf.test.TestCase):
    """Tests for confusion_matrix
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.9
        self.labels = ["normal"]
        gt_bboxes = np.array(
            [[38, 1, 51, 18, 1],
             [28, 70, 83, 99, 0],
             [77, 29, 94, 99, 2],
             [41, 81, 68, 99, 0],
             [43, 24, 99, 94, 1],
             [30, 67, 99, 99, 2],
             ])

        predicted_bboxes = np.array(
            [[38, 1, 51, 9, 1],
             [42, 70, 83, 99, 0],
             [77, 29, 94, 99, 2],
             [30, 67, 99, 99, 2],
             [41, 81, 0, 99, 1],
             [43, 24, 99, 104, 1],
             [43, 24, 99, 94, 2]
             ])

        self.expected_gt_classes = gt_bboxes[:, 4]
        self.expected_predicted_classes = predicted_bboxes[:, 4]
        self.expected_gt_matched_classes = [0, 2, 1, 2]
        self.expected_predicted_matched_classes = [0, 2, 2, 2]
        self.gt_csv = self.get_test_data(
            gt_bboxes, self.expected_gt_classes)
        self.predicted_csv = self.get_test_data(
            predicted_bboxes, self.expected_predicted_classes)
        self.labels = [0, 1, 2]
        self.expected_cm = np.array(([
            [1, 0, 0, 0, 1],
            [0, 0, 1, 3, 3],
            [0, 0, 2, 0, 3],
            [1, 1, 0, 0, 0],
            [2, 2, 2, 0, 0]]))
        self.expected_ncm = np.array((
            [[0.5, 0., 0.],
             [0., 0., 0.5],
             [0., 0., 1.]]))

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def get_test_data(self, bboxes, labels):
        # Write test data csv file
        tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv = tf.name
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        for i, bbox in enumerate(bboxes):
            label_name = labels[i]
            df = df.append({'image_id': "file",
                            'xmin': bbox[0],
                            'xmax': bbox[2],
                            'ymin': bbox[1],
                            'ymax': bbox[3],
                            'label': label_name,
                            'prob': 0.95},
                           ignore_index=True)
        if type(label_name) is str:
            cols = ['xmin', 'xmax', 'ymin', 'ymax']
            df[cols] = df[cols].applymap(np.int64)
        elif type(label_name) is float:
            cols = ['xmin', 'xmax', 'ymin', 'ymax', 'label']
            df[cols] = df[cols].applymap(np.int64)
        df.to_csv(csv)
        self.tempfiles_to_delete.append(csv)
        return csv

    def testGetMatchedGTPredict(self):
        # Get matched gt, predict, total classes test
        (obtained_gt_classes,
         obtained_predicted_classes,
         obtained_gt_matched_classes,
         obtained_predicted_matched_classes) = \
            confusion_matrix.get_matched_gt_predict(
            self.gt_csv,
            self.predicted_csv,
            self.labels,
            self.iou_threshold,
            self.confidence_threshold)

        np.testing.assert_array_equal(
            obtained_gt_classes, self.expected_gt_classes)
        np.testing.assert_array_equal(
            obtained_predicted_classes, self.expected_predicted_classes)
        np.testing.assert_array_equal(
            obtained_gt_matched_classes, self.expected_gt_matched_classes)
        np.testing.assert_array_equal(
            obtained_predicted_matched_classes,
            self.expected_predicted_matched_classes)

    def testAppendUnmatchedGTPredict(self):
        # Add unmatched, total to confusion matrix test
        (gt_classes,
         predicted_classes,
         gt_matched_classes,
         predicted_matched_classes) = \
            confusion_matrix.get_matched_gt_predict(
            self.gt_csv,
            self.predicted_csv,
            self.labels,
            self.iou_threshold,
            self.confidence_threshold)

        cm = confusion_matrix.sklearn.metrics.confusion_matrix(
            gt_matched_classes, predicted_matched_classes, labels=self.labels)

        complete_cm = \
            confusion_matrix.append_unmatched_gt_predict(
                cm,
                self.labels,
                gt_matched_classes,
                predicted_matched_classes,
                gt_classes,
                predicted_classes)

        np.testing.assert_array_equal(complete_cm, self.expected_cm)

    def testUnequalBoxes(self):
        # Single unequal box confusion matrix test
        labels = ["normal"]
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, labels)

        bboxes = [[11, 11, 20, 20]]
        predicted_csv = self.get_test_data(bboxes, labels)

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            labels, self.iou_threshold, self.confidence_threshold)
        expected = np.zeros((3, 3))
        expected[0, 1] = expected[1, 0] = 1
        expected[0, 2] = expected[2, 0] = 1
        np.testing.assert_array_equal(cm, expected)

    def testEqualBoxes(self):
        # Equal boxes confusion matrix test
        labels = ["normal"]
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, labels)

        bboxes = [[0, 0, 10, 10]]
        predicted_csv = self.get_test_data(bboxes, labels)

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            labels, self.iou_threshold, self.confidence_threshold)
        expected = np.zeros((3, 3))
        expected[0, 0] = expected[2, 0] = expected[0, 2] = 1
        np.testing.assert_array_equal(cm, expected)

    def testGetConfusionMatrix(self):
        # Getting confusion matrix on random bboxes test
        cm = confusion_matrix.get_confusion_matrix(
            self.gt_csv, self.predicted_csv,
            self.labels, self.iou_threshold, self.confidence_threshold)
        np.testing.assert_array_equal(cm, self.expected_cm)

    def testNormalizeConfusionMatrix(self):
        # Getting normalized confusion matrix on random bboxes test

        normalized_cm = confusion_matrix.normalize_confusion_matrix(
            self.expected_cm)
        np.testing.assert_array_equal(normalized_cm, self.expected_ncm)


if __name__ == '__main__':
    tf.test.main()
