import math
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from luminoth.utils.data_aug_demo import (
    update_augmentation, get_data_aug_images, mosaic_data_aug,
    DATA_AUGMENTATION_STRATEGIES, TILE_SIZE)
from luminoth.utils.split_train_val import LUMI_CSV_COLUMNS
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class DataAugDemoTest(tf.test.TestCase):
    """Tests for data_aug_demo
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.input_image_format = ".png"
        self.gray_image_shape = [50, 41]
        self.color_image_shape = [50, 41, 3]

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

    def write_test_data(
            self, image, image_save_path, bboxes, labels):
        # Write test images, csv/txt files
        cv2.imwrite(image_save_path, image)
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        for i, bbox in enumerate(bboxes):
            df = df.append({'image_id': image_save_path,
                            'xmin': np.int64(bbox[0]),
                            'xmax': np.int64(bbox[2]),
                            'ymin': np.int64(bbox[1]),
                            'ymax': np.int64(bbox[3]),
                            'label': labels[i]},
                           ignore_index=True)
        self.tempfiles_to_delete.append(image_save_path)
        csv_path = image_save_path + ".csv"
        df.to_csv(csv_path)
        return image_save_path, csv_path

    def testUpdateAugmentation(self):
        # Test update augmentation list
        augmented_images = []
        labels = ["foo", "bar", "bla"]
        augmented_dict = {}
        num_bboxes = 5
        image, bboxes = self._get_image_with_boxes(
            self.color_image_shape, num_bboxes)
        location = tempfile.mkdtemp()
        augmentation = "testAug"
        augmented_dict['image'] = image
        bboxes_with_labels = []
        for label in range(num_bboxes):
            bboxes_with_labels.append(bboxes[label].tolist() + [1])
        augmented_dict['bboxes'] = bboxes_with_labels
        update_augmentation(
            augmented_dict, labels, location, augmentation, augmented_images)
        assert len(augmented_images) == 1

    def testGetDataAugImages(self):
        # Test get all data augmented images
        num_bboxes = 3
        image, bboxes = self._get_image_with_boxes(
            self.color_image_shape, num_bboxes)
        labels = [0, 1, 2]
        bboxes_with_labels = []
        for label in range(num_bboxes):
            bboxes_with_labels.append(bboxes[label].tolist() + [1])
        bboxes_with_labels = np.array(bboxes_with_labels)
        augmented_images = get_data_aug_images(
            image, bboxes_with_labels, labels)
        assert len(augmented_images) == len(DATA_AUGMENTATION_STRATEGIES)

    def testMosaicDataAugGray(self):
        # gray mosaic data augmentation test
        location = tempfile.mkdtemp()
        num_images = len(DATA_AUGMENTATION_STRATEGIES)
        image, bboxes = self._get_image_with_boxes(self.gray_image_shape, 3)
        image_save_path = os.path.join(location, "input.png")
        labels = ["bla", "foo", "bar"]
        input_image, csv_path = self.write_test_data(
            image, image_save_path, bboxes, labels)
        output_png = os.path.join(location, "mosaic.png")
        tile_size = TILE_SIZE
        fill_value = 128
        mosaic_data_aug(
            input_image,
            ".png",
            csv_path,
            "image_id",
            tile_size,
            fill_value,
            output_png)
        assert os.path.exists(output_png)
        assert os.path.getsize(output_png) != 0
        mosaiced_image = cv2.imread(
            output_png, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1], 3)

    def testMosaicDataAugColor(self):
        # Color mosaic data augmentation test
        location = tempfile.mkdtemp()
        num_images = len(DATA_AUGMENTATION_STRATEGIES)
        image, bboxes = self._get_image_with_boxes(self.color_image_shape, 3)
        image_save_path = os.path.join(location, "input.png")
        labels = ["bla", "foo", "bar"]
        input_image, csv_path = self.write_test_data(
            image, image_save_path, bboxes, labels)
        output_png = os.path.join(location, "mosaic.png")
        tile_size = TILE_SIZE
        fill_value = 128
        mosaic_data_aug(
            input_image,
            ".png",
            csv_path,
            "image_id",
            tile_size,
            fill_value,
            output_png)
        assert os.path.exists(output_png)
        assert os.path.getsize(output_png) != 0
        mosaiced_image = cv2.imread(
            output_png, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1], 3)


if __name__ == '__main__':
    tf.test.main()
