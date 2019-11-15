import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os

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

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def get_test_data(self, bboxes, labels):
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

    def testNoOverlap(self):
        # Single box test
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, self.labels)

        bboxes = [[11, 11, 20, 20]]
        predicted_csv = self.get_test_data(bboxes, self.labels)

        labels = ["normal"]

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            labels, self.iou_threshold, self.confidence_threshold)

        expected = np.zeros((1, 1))
        np.testing.assert_array_equal(cm, expected)

    def testAllOverlap(self):
        # Equal boxes
        bboxes = [[0, 0, 10, 10]]
        groundtruth_csv = self.get_test_data(bboxes, self.labels)

        bboxes = [[0, 0, 10, 10]]
        predicted_csv = self.get_test_data(bboxes, self.labels)

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            self.labels, self.iou_threshold, self.confidence_threshold)

        expected = np.zeros((2, 2))
        expected[0, 0] = 1
        np.testing.assert_array_equal(cm, expected)

    def testConfusionMatrix(self):
        gt_bboxes = np.array(
            [[38, 1, 51, 18, 1],
             [28, 70, 83, 99, 0],
             [77, 29, 94, 99, 2],
             [41, 81, 68, 99, 0],
             [43, 24, 99, 94, 1],
             [30, 67, 99, 99, 2]
             ])

        predicted_bboxes = np.array(
            [[38, 1, 51, 9, 1],
             [42, 70, 83, 99, 0],
             [77, 29, 94, 99, 2],
             [30, 67, 99, 99, 2],
             [41, 81, 0, 99, 0],
             [43, 24, 99, 104, 1]
             ])

        gt_csv = self.get_test_data(
            gt_bboxes, [1, 0, 2, 0, 1, 2])
        predicted_csv = self.get_test_data(
            predicted_bboxes, [1, 0, 2, 2, 0, 1])

        cm = confusion_matrix.get_confusion_matrix(
            gt_csv, predicted_csv,
            [0, 1, 2], self.iou_threshold, self.confidence_threshold)
        expected_cm = np.array(
            [[1, 0, 0, 1],
             [0, 1, 0, 1],
             [0, 0, 2, 0],
             [1, 1, 0, 0]])
        np.testing.assert_array_equal(cm, expected_cm)


if __name__ == '__main__':
    tf.test.main()
