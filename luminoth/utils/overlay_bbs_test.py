import glob
import math
import os
import tempfile

import natsort
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io

from luminoth.utils.overlay_bbs import (
    add_basename_gather_df, get_image_paths_per_class, INPUT_CSV_COLUMNS,
    get_lumi_csv_df, LUMI_CSV_COLUMNS, split_data_to_train_val,
    write_lumi_images_csv, filter_dense_annotation)
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
            path.replace("txt", "png") for path in self.bb_ann_filenames]

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
        df = pd.DataFrame(columns=INPUT_CSV_COLUMNS)
        image_save_path = os.path.join(location, image_path)
        csv_save_path = os.path.join(location, ann_path)
        io.imsave(image_save_path, image)
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
            csv_filename = "test_bb_labels_{}.txt".format(i)
            image, bboxes = self._get_image_with_boxes(
                self.image_shape, self.num_bboxes)
            csv = self.get_test_data(
                image, im_filename, bboxes, labels, csv_filename)
            filenames.append(csv)
        return filenames

    def testAddBasenameGatherDfCsv(self):
        # Combine list of annotation csv files and add base_path column test
        df = add_basename_gather_df(
            self.bb_ann_filenames, self.input_image_format)
        for index, row in df.iterrows():
            assert row["base_path"] == os.path.basename(
                row["image_path"].replace(self.input_image_format, ""))

    def testAddBasenameGatherDfTxt(self):
        # Combine list of annotation txt files and add base_path column test
        df = add_basename_gather_df(
            self.bb_ann_filenames, self.input_image_format)
        for index, row in df.iterrows():
            assert row["base_path"] == os.path.basename(
                row["image_path"].replace(self.input_image_format, ""))

    def testGetImagePathsPerClass(self):
        # Get unique image paths per class test
        bb_ann_filenames = self.get_ann_filenames(
            self.num_images,
            [[0] * self.num_bboxes,
             [1] * self.num_bboxes,
             self.labels,
             self.labels,
             [0] * self.num_bboxes])
        df = add_basename_gather_df(
            bb_ann_filenames, self.input_image_format)

        image_ids_per_class = get_image_paths_per_class(df)
        expected_key_values_length = {0: 4, 1: 3, 2: 2}
        for key, value in expected_key_values_length.items():
            assert value == len(image_ids_per_class[key])

    def testFilterDenseAnnotation(self):
        # Test filtered unique image paths per class test
        # after suppressing the dense annotation of a class
        bb_ann_filenames = self.get_ann_filenames(
            self.num_images,
            [[0] * self.num_bboxes,
             [1] * self.num_bboxes,
             self.labels,
             self.labels,
             [0] * self.num_bboxes])
        df = add_basename_gather_df(
            bb_ann_filenames, self.input_image_format)

        image_ids_per_class = get_image_paths_per_class(df)
        expected_key_values_length = {0: 4, 1: 3, 2: 2}
        for key, value in expected_key_values_length.items():
            assert value == len(image_ids_per_class[key])

        filtered_image_ids_per_class = filter_dense_annotation(
            image_ids_per_class)
        expected_key_values_length = {1: 3, 2: 2}
        for key, value in expected_key_values_length.items():
            assert value == len(filtered_image_ids_per_class[key])

    def testgetLumiCsvDf(self):
        # Get lumi csv dataframe test
        df = add_basename_gather_df(
            self.bb_ann_filenames, self.input_image_format)
        lumi_df = get_lumi_csv_df(
            df, self.images, self.input_image_format)
        assert len(lumi_df) == self.num_bboxes * self.num_images
        assert list(lumi_df.columns.values) == LUMI_CSV_COLUMNS

    def testWriteLumiImagesCsv(self):
        bb_labels = add_basename_gather_df(
            self.bb_ann_filenames, self.input_image_format)
        output_dir = tempfile.mkdtemp()
        output_ann_path = os.path.join(output_dir, "test_train.csv")
        write_lumi_images_csv(
            self.images,
            output_dir,
            self.input_image_format,
            self.output_image_format,
            bb_labels,
            output_ann_path)
        images = natsort.natsorted(
            glob.glob(
                os.path.join(output_dir, "*" + self.output_image_format)))
        assert len(images) == len(self.images)
        lumi_df = pd.read_csv(output_ann_path)
        assert len(lumi_df) == self.num_bboxes * self.num_images
        assert list(lumi_df.columns.values) == LUMI_CSV_COLUMNS

    def testSplitDataToTrainValFilter(self):
        percentage = 0.8
        random_seed = 42

        bb_ann_filenames = self.get_ann_filenames(
            self.num_images,
            [[0] * self.num_bboxes,
             [1] * self.num_bboxes,
             self.labels,
             self.labels,
             [0] * self.num_bboxes])

        output_dir = tempfile.mkdtemp()
        split_data_to_train_val(
            bb_ann_filenames,
            percentage,
            random_seed,
            True,
            self.input_image_format,
            output_dir,
            self.output_image_format)
        split_images = [2, 1]
        splits = ["train", "val"]
        for i, split in enumerate(splits):
            images = natsort.natsorted(
                glob.glob(
                    os.path.join(
                        output_dir, split, "*" + self.output_image_format)))
            assert len(images) == split_images[i]
            lumi_df = pd.read_csv(os.path.join(output_dir, split + ".csv"))
            assert len(lumi_df) == self.num_bboxes * split_images[i]
            assert list(lumi_df.columns.values) == LUMI_CSV_COLUMNS

    def testSplitDataToTrainValUnFilter(self):
        percentage = 0.6
        random_seed = 42
        split_images = [
            math.floor(percentage * self.num_images),
            self.num_images - math.floor(percentage * self.num_images)]
        output_dir = tempfile.mkdtemp()
        split_data_to_train_val(
            self.bb_ann_filenames,
            percentage,
            random_seed,
            False,
            self.input_image_format,
            output_dir,
            self.output_image_format)

        splits = ["train", "val"]
        for i, split in enumerate(splits):
            images = natsort.natsorted(
                glob.glob(
                    os.path.join(
                        output_dir, split, "*" + self.output_image_format)))
            assert len(images) == split_images[i]
            lumi_df = pd.read_csv(os.path.join(output_dir, split + ".csv"))
            assert len(lumi_df) == self.num_bboxes * split_images[i]
            assert list(lumi_df.columns.values) == LUMI_CSV_COLUMNS


if __name__ == '__main__':
    tf.test.main()
