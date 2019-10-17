import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile

import luminoth.confusion_matrix as confusion_matrix
from luminoth.predict import LUMI_CSV_COLUMNS


class ConfusionMatrixTest(tf.test.TestCase):
    """Tests for confusion_matrix
    """
    def tearDown(self):
        tf.reset_default_graph()

    def get_test_data(self, bbox):
        tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv = tf.name
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        df = df.append({'image_id': "file",
                        'xmin': bbox[0],
                        'xmax': bbox[2],
                        'ymin': bbox[1],
                        'ymax': bbox[3],
                        'label': "normal",
                        'prob': 0.95},
                       ignore_index=True)
        df.to_csv(csv)
        return csv

    def test_NoOverlap(self):
        # Single box test
        bbox = [0, 0, 10, 10]
        groundtruth_csv = self.get_test_data(bbox)

        bbox = [11, 11, 20, 20]
        predicted_csv = self.get_test_data(bbox)

        categories = ["normal"]
        iou_threshold = 0.5
        confidence_threshold = 0.9

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            categories, iou_threshold, confidence_threshold)

        expected = np.zeros((1, 1))
        np.testing.assert_array_equal(cm, expected)

    def testAllOverlap(self):
        # Equal boxes
        bbox = [0, 0, 10, 10]
        groundtruth_csv = self.get_test_data(bbox)

        bbox = [0, 0, 10, 10]
        predicted_csv = self.get_test_data(bbox)

        categories = ["normal"]
        iou_threshold = 0.5
        confidence_threshold = 0.9

        cm = confusion_matrix.get_confusion_matrix(
            groundtruth_csv, predicted_csv,
            categories, iou_threshold, confidence_threshold)

        expected = np.ones((1, 1))
        np.testing.assert_array_equal(cm, expected)


if __name__ == '__main__':
    tf.test.main()
