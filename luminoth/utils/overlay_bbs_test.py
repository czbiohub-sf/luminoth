import glob
import math
import os
import tempfile

import cv2
import natsort
import numpy as np
import pandas as pd
import tensorflow as tf


from luminoth.utils.overlay_bbs import (
    overlay_bbs_on_all_images, overlay_bb_labels)
from luminoth.utils.split_train_val import LUMI_CSV_COLUMNS
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class SplitTrainValTest(tf.test.TestCase):
    """Tests for overlay_bbs
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.labels = [0, 1, 2, 0, 2, 1, 0, 0, 2, 1, 0]
        self.input_image_format = ".png"
        self.num_images = 5
        self.output_image_format = ".jpg"
        self.image_shape = [50, 41]
        self.num_bboxes = 11
        self.bb_ann_filenames = self.get_ann_filenames(self.num_images)
        self.images = [
            path.replace("csv", "png") for path in self.bb_ann_filenames]

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

    def get_test_data(self, image, image_path, bboxes, labels, ann_path):
        # Write test images, csv/txt files
        location = tempfile.mkdtemp()
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        image_save_path = os.path.join(location, image_path)
        csv_save_path = os.path.join(location, ann_path)
        cv2.imwrite(image_save_path, image)
        for i, bbox in enumerate(bboxes):
            label_name = labels[i]
            df = df.append({'image_path': image_save_path,
                            'x1': bbox[0],
                            'x2': bbox[2],
                            'y1': bbox[1],
                            'y2': bbox[3],
                            'class_name': label_name},
                           ignore_index=True)
        if type(label_name) is str:
            cols = ['xmin', 'xmax', 'ymin', 'ymax']
            df[cols] = df[cols].applymap(np.int64)
        elif type(label_name) is float:
            cols = ['xmin', 'xmax', 'ymin', 'ymax', 'label']
            df[cols] = df[cols].applymap(np.int64)
        df.to_csv(csv_save_path)
        self.tempfiles_to_delete.append(image_save_path)
        self.tempfiles_to_delete.append(csv_save_path)
        return csv_save_path

    def get_ann_filenames(self, num_images, all_labels=None):
        # Get list of test annotation csv/txt files
        if all_labels is None:
            all_labels = [self.labels] * num_images
        filenames = []
        for i in range(num_images):
            labels = all_labels[i]
            im_filename = "test_bb_labels_{}{}".format(
                i, self.input_image_format)
            csv_filename = "test_bb_labels_{}.csv".format(i)
            image, bboxes = self._get_image_with_boxes(
                self.image_shape, self.num_bboxes)
            csv = self.get_test_data(
                image, im_filename, bboxes, labels, csv_filename)
            filenames.append(csv)
        return filenames

    def testOverlayBBsOnAllImages(self):
        # Combine list of annotation csv files and add base_path column test
        df = overlay_bb_labels(self.im_path, self.df)
        for index, row in df.iterrows():
            assert row["base_path"] == os.path.basename(
                row["image_path"].replace(self.input_image_format, ""))

    def testOverlayBBsOnImage(self):
        # Combine list of annotation txt files and add base_path column test
        df = overlay_bbs_on_all_images(
            self.im_dir,
            self.csv_path,
            self.output_dir,
            self.input_image_format)
        for index, row in df.iterrows():
            assert row["base_path"] == os.path.basename(
                row["image_path"].replace(self.input_image_format, ""))


if __name__ == '__main__':
    tf.test.main()
