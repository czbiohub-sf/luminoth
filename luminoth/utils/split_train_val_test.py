import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os

from luminoth.utils import split_train_val
from luminoth.predict import LUMI_CSV_COLUMNS
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class SplitTrainValTest(tf.test.TestCase):
    """Tests for split_train_val
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.9
        self.labels = ["normal"]

        # Create fake images, fake csv files
        # place them in different folders
        # atleast 3 -4 different images
        # with 2 different classe of bounding boxes in them

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def _gen_image(self, *shape):
        return np.random.rand(*shape)

    def _get_image_with_boxes(self, image_size, total_boxes):
        image = self._gen_image(*image_size)
        bboxes = generate_gt_boxes(
            total_boxes, image_size[:2],
        )
        return image, bboxes

    def get_test_data(self, bboxes, labels):
        tf = tempfile.NamedTemporaryDirectory(prefix="test_split_train_val", delete=False)
        im_dir = tf.name
        im_dir = tf.name
        im_dir = tf.name
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
