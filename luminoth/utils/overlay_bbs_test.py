import glob
import os
import tempfile

import cv2
import natsort
import numpy as np
import pandas as pd
import tensorflow as tf


from luminoth.utils.overlay_bbs import (
    overlay_bbs_on_all_images, overlay_bb_labels, add_base_path)
from luminoth.utils.split_train_val import LUMI_CSV_COLUMNS
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class OverlayBbsTest(tf.test.TestCase):
    """
    Tests for overlay_bbs
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.labels = [0, 1, 2, 0, 2, 1, 0, 0, 2, 1, 0]
        self.input_image_format = ".png"
        self.image_shape = [50, 41]
        self.output_image_shape = (50, 41, 3)
        self.num_bboxes = 11
        self.im_path = "test_bb_labels_0.png"

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def _gen_image(self, *shape):
        np.random.seed(43)
        return np.random.rand(*shape)

    def _get_image_with_boxes(self, image_size, total_boxes):
        image = self._gen_image(*image_size)
        bboxes = generate_gt_boxes(
            total_boxes, image_size[:2],
        )
        image = image.astype(np.uint8)
        return image, bboxes

    def get_test_data(
            self, image, image_save_path, bboxes,
            labels, df):
        # Write test images, csv/txt files
        cv2.imwrite(image_save_path, image)
        for i, bbox in enumerate(bboxes):
            df = df.append({'image_id': image_save_path,
                            'xmin': np.int64(bbox[0]),
                            'xmax': np.int64(bbox[2]),
                            'ymin': np.int64(bbox[1]),
                            'ymax': np.int64(bbox[3]),
                            'label': np.int64(labels[i])},
                           ignore_index=True)
        self.tempfiles_to_delete.append(image_save_path)
        return df

    def get_ann_filename(
            self, num_images, columns, all_labels=None):
        # Get a test annotation csv/txt files
        if all_labels is None:
            all_labels = [self.labels] * num_images
        location = tempfile.mkdtemp()
        csv_filename = os.path.join(location, "test_bb_labels.csv")
        df = pd.DataFrame(columns=columns)

        for i in range(num_images):
            labels = all_labels[i]
            im_filename = "test_bb_labels_{}{}".format(
                i, self.input_image_format)
            image_save_path = os.path.join(location, im_filename)
            image, bboxes = self._get_image_with_boxes(
                self.image_shape, self.num_bboxes)
            df = self.get_test_data(image, image_save_path, bboxes, labels, df)

        df.to_csv(csv_filename)
        return csv_filename

    def testAddBasename(self):
        # Test add base_path col
        # Set inputs to add_base_path
        csv = self.get_ann_filename(7, LUMI_CSV_COLUMNS)
        df = add_base_path(
            csv, self.input_image_format)

        # Assert base path exists and is as expected
        for index, row in df.iterrows():
            assert row["base_path"] == (
                os.path.basename(row['image_id']).replace(
                    self.input_image_format, ""))

    def testOverlayBBsOnImage(self):
        # Test overlay bb on a single image with different dataframe where
        # Set inputs to overlay_bb_labels
        csv = self.get_ann_filename(1, LUMI_CSV_COLUMNS)
        df = pd.read_csv(csv)
        df['base_path'] = self.im_path.replace(self.input_image_format, "")
        image = overlay_bb_labels(
            df['image_id'].tolist()[0], self.input_image_format, df)

        # Assert overlaid image is as expected shape
        assert image.shape == self.output_image_shape
        assert image.sum() != 0

    def testOverlayBBsOnAllImage(self):
        # Test overlay bb on multiple images with different dataframe where
        # Set inputs to overlay_bbs_on_all_images
        output_dir = tempfile.mkdtemp() + os.sep
        num_images = 7
        csv = self.get_ann_filename(
            num_images, LUMI_CSV_COLUMNS)

        overlay_bbs_on_all_images(
            os.path.dirname(csv),
            csv,
            output_dir,
            self.input_image_format)

        # Assert overlaid images are as expected
        images = natsort.natsorted(
            glob.glob(
                os.path.join(
                    output_dir, "*" + self.input_image_format)))
        for path in images:
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            assert image.shape == self.output_image_shape
            assert image.sum() != 0
        assert len(images) == num_images


if __name__ == '__main__':
    tf.test.main()
