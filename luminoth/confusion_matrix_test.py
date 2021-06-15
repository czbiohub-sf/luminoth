import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

import luminoth.confusion_matrix as confusion_matrix
from luminoth.predict import LUMI_CSV_COLUMNS


class ConfusionMatrixTest(tf.test.TestCase):
    """Tests for confusion_matrix"""

    def setUp(self):

        # Set up common input parameters, expected results
        self.input_image_format = ".png"
        self.input_image_path = "file.png"
        self.num_cpus = 1
        self.tempfiles_to_delete = []
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.9
        self.labels = ["normal"]
        self.gt_bboxes = np.array(
            [
                [38, 1, 51, 18, 1],
                [28, 70, 83, 99, 0],
                [77, 29, 94, 99, 2],
                [41, 81, 68, 99, 0],
                [43, 24, 99, 94, 1],
                [30, 67, 99, 99, 2],
            ]
        )

        self.predicted_bboxes = np.array(
            [
                [38, 1, 51, 9, 1],
                [42, 70, 83, 99, 0],
                [77, 29, 94, 99, 2],
                [30, 67, 99, 99, 2],
                [41, 81, 0, 99, 1],
                [43, 24, 99, 104, 1],
                [43, 24, 99, 94, 2],
            ]
        )

        self.expected_gt_classes = self.gt_bboxes[:, 4]
        self.expected_predicted_classes = self.predicted_bboxes[:, 4]
        self.expected_gt_matched_classes = [1, 0, 2, 1, 2]
        self.expected_predicted_matched_classes = [1, 0, 2, 2, 2]

        self.gt_csv = self.get_test_data(self.gt_bboxes, self.expected_gt_classes)
        self.predicted_csv = self.get_test_data(
            self.predicted_bboxes, self.expected_predicted_classes
        )

        self.labels = [0, 1, 2]

        self.expected_cm = np.array(
            (
                [
                    [1, 0, 0, 1, 2],
                    [0, 1, 1, 0, 2],
                    [0, 0, 2, 0, 2],
                    [0, 2, 0, 0, 0],
                    [1, 3, 3, 0, 0],
                ]
            )
        )

        self.expected_ncm = np.array(
            ([[1, 0.0, 0.0], [0.0, 0.333333, 0.333333], [0.0, 0.0, 0.666667]]),
            dtype=np.float32,
        )

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def get_test_data(self, bboxes, labels, input_image_path=None):
        # Write test data csv file
        if input_image_path is None:
            input_image_path = self.input_image_path
        tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv = tf.name
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        for i, bbox in enumerate(bboxes):
            label_name = labels[i]
            df = df.append(
                {
                    "image_id": input_image_path,
                    "base_path": input_image_path.replace(self.input_image_format, ""),
                    "xmin": bbox[0],
                    "xmax": bbox[2],
                    "ymin": bbox[1],
                    "ymax": bbox[3],
                    "label": label_name,
                    "prob": 0.95,
                },
                ignore_index=True,
            )
        if type(label_name) is str:
            cols = ["xmin", "xmax", "ymin", "ymax"]
            df[cols] = df[cols].applymap(np.int64)
        elif type(label_name) is float:
            cols = ["xmin", "xmax", "ymin", "ymax", "label"]
            df[cols] = df[cols].applymap(np.int64)
        df.to_csv(csv)

        self.tempfiles_to_delete.append(csv)

        return csv

    def testGetMatchedGTPredict(self):
        # Get matched gt, predict, total classes test
        (
            obtained_gt_classes,
            obtained_predicted_classes,
            obtained_gt_matched_classes,
            obtained_predicted_matched_classes,
        ) = confusion_matrix.get_matched_gt_predict(
            self.gt_csv,
            self.predicted_csv,
            self.labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.input_image_format,
            self.num_cpus,
        )

        np.testing.assert_array_equal(obtained_gt_classes, self.expected_gt_classes)
        np.testing.assert_array_equal(
            obtained_predicted_classes, self.expected_predicted_classes
        )
        np.testing.assert_array_equal(
            obtained_gt_matched_classes, self.expected_gt_matched_classes
        )
        np.testing.assert_array_equal(
            obtained_predicted_matched_classes, self.expected_predicted_matched_classes
        )

    def testAppendUnmatchedGTPredict(self):
        # Add unmatched, total to confusion matrix test
        labels = [0]
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, labels)

        bboxes = [[11, 11, 20, 20]]
        predicted_csv = self.get_test_data(bboxes, labels)

        df = pd.read_csv(self.gt_csv).append(pd.read_csv(groundtruth_csv))
        df.to_csv(self.gt_csv)
        df = pd.read_csv(self.predicted_csv).append(pd.read_csv(predicted_csv))
        df.to_csv(self.predicted_csv)
        (
            gt_classes,
            predicted_classes,
            gt_matched_classes,
            predicted_matched_classes,
        ) = confusion_matrix.get_matched_gt_predict(
            self.gt_csv,
            self.predicted_csv,
            self.labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.input_image_format,
            self.num_cpus,
        )

        cm = confusion_matrix.sklearn.metrics.confusion_matrix(
            gt_matched_classes, predicted_matched_classes, labels=self.labels
        )

        complete_cm = confusion_matrix.append_unmatched_gt_predict(
            cm,
            self.labels,
            gt_matched_classes,
            predicted_matched_classes,
            gt_classes,
            predicted_classes,
        )
        expected_cm = np.array(
            (
                [
                    [1, 0, 0, 2, 3],
                    [0, 1, 1, 0, 2],
                    [0, 0, 2, 0, 2],
                    [1, 2, 0, 0, 0],
                    [2, 3, 3, 0, 0],
                ]
            )
        )
        np.testing.assert_array_equal(complete_cm, expected_cm)

    def testUnequalBoxes(self):
        # Single unequal box confusion matrix test
        labels = ["normal"]
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, labels)

        bboxes = [[11, 11, 20, 20]]
        predicted_csv = self.get_test_data(bboxes, labels)

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv,
            predicted_csv,
            labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.input_image_format,
            self.num_cpus,
        )
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
            groundtruth_csv,
            predicted_csv,
            labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.input_image_format,
            self.num_cpus,
        )
        expected = np.zeros((3, 3))
        expected[0, 0] = expected[2, 0] = expected[0, 2] = 1
        np.testing.assert_array_equal(cm, expected)

    def testGetConfusionMatrix(self):
        # Getting confusion matrix on random bboxes test
        cm = confusion_matrix.get_confusion_matrix(
            self.gt_csv,
            self.predicted_csv,
            self.labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.input_image_format,
            self.num_cpus,
        )
        np.testing.assert_array_equal(cm, self.expected_cm)

    def testNormalizeConfusionMatrix(self):
        # Getting normalized confusion matrix on random bboxes test

        normalized_cm = confusion_matrix.normalize_confusion_matrix(self.expected_cm)
        np.testing.assert_array_almost_equal(normalized_cm, self.expected_ncm)

    def testformatElementToMatlabConfusionMatrix(self):
        self.expected_cm[-1, -1] = np.sum(self.expected_cm[-1, :])
        self.expected_cm = np.delete(self.expected_cm, -2, 0)
        self.expected_cm = np.delete(self.expected_cm, -2, 1)
        assert (
            confusion_matrix.format_element_to_matlab_confusion_matrix(
                0, 0, self.expected_cm
            )
            == "1\n14.29%"
        )
        assert (
            confusion_matrix.format_element_to_matlab_confusion_matrix(
                1, 1, self.expected_cm
            )
            == "1\n14.29%"
        )
        assert (
            confusion_matrix.format_element_to_matlab_confusion_matrix(
                3, 3, self.expected_cm
            )
            == "57.14%\n42.86%"
        )
        assert (
            confusion_matrix.format_element_to_matlab_confusion_matrix(
                3, 2, self.expected_cm
            )
            == "66.67%\n33.33%"
        )
        assert (
            confusion_matrix.format_element_to_matlab_confusion_matrix(
                2, 3, self.expected_cm
            )
            == "100%\n0.00%"
        )

    def testGetValidMatchIOU(self):
        i = j = 0
        obtained_iou_match = confusion_matrix.get_valid_match_iou(
            i,
            j,
            self.gt_bboxes[:, :4],
            self.predicted_bboxes[:, :4],
            self.iou_threshold,
        )
        assert obtained_iou_match == [i, j, 0.5]

        i = 0
        j = 1
        obtained_iou_match = confusion_matrix.get_valid_match_iou(
            i,
            j,
            self.gt_bboxes[:, :4],
            self.predicted_bboxes[:, :4],
            self.iou_threshold,
        )
        assert obtained_iou_match is None

        i = j = 1
        obtained_iou_match = confusion_matrix.get_valid_match_iou(
            i,
            j,
            self.gt_bboxes[:, :4],
            self.predicted_bboxes[:, :4],
            self.iou_threshold,
        )
        assert obtained_iou_match == [i, j, 0.75]

    def testGetMatchedGTPredictPerImage(self):
        # Get matched gt, predict, total classes test
        (
            obtained_gt_classes,
            obtained_predicted_classes,
            obtained_gt_matched_classes,
            obtained_predicted_matched_classes,
        ) = confusion_matrix.get_matched_gt_predict_per_image(
            self.input_image_path,
            self.input_image_format,
            pd.read_csv(self.gt_csv),
            pd.read_csv(self.predicted_csv),
            self.labels,
            self.iou_threshold,
            self.confidence_threshold,
            self.num_cpus,
        )

        np.testing.assert_array_equal(obtained_gt_classes, self.expected_gt_classes)
        np.testing.assert_array_equal(
            obtained_predicted_classes, self.expected_predicted_classes
        )
        np.testing.assert_array_equal(
            obtained_gt_matched_classes, self.expected_gt_matched_classes
        )
        np.testing.assert_array_equal(
            obtained_predicted_matched_classes, self.expected_predicted_matched_classes
        )


if __name__ == "__main__":
    tf.test.main()
