import glob
import math
import os
import tempfile

import cv2
import natsort
import numpy as np
import tensorflow as tf


from luminoth.utils.mosaic import assemble_mosaic, mosaic_images


class OverlayBbsTest(tf.test.TestCase):
    """Tests for overlay_bbs
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

    def write_test_data(self, num_images, shape):
        location = tempfile.mkdtemp()
        for i in range(num_images):
            im_filename = "test_bb_labels_{}{}".format(
                i, self.input_image_format)
            image_save_path = os.path.join(location, im_filename)
            cv2.imwrite(image_save_path, self._gen_image(*shape))
            self.tempfiles_to_delete.append(image_save_path)
        return location

    def testAssembleGrayMosaic(self):
        # gray mosaic test
        num_images = 7
        im_dir = self.write_test_data(num_images, self.gray_image_shape)
        images_in_path = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))
        num_images = len(images_in_path)
        tile_size = [10, 12]
        fill_value = 1
        mosaiced_image = assemble_mosaic(
            images_in_path,
            tile_size,
            fill_value)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1],
            1)

    def testAssembleColorMosaic(self):
        # color mosaic test
        num_images = 13
        im_dir = self.write_test_data(num_images, self.color_image_shape)
        images_in_path = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))
        num_images = len(images_in_path)
        tile_size = [7, 9]
        fill_value = 1
        mosaiced_image = assemble_mosaic(
            images_in_path,
            tile_size,
            fill_value)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1],
            3)

    def testGrayMosaic(self):
        # gray mosaic test
        num_images = 7
        im_dir = self.write_test_data(num_images, self.gray_image_shape)
        output_png = os.path.join(im_dir, "mosaic.png")
        tile_size = self.gray_image_shape
        mosaic_images(
            im_dir,
            None,
            "first",
            output_png,
            self.input_image_format)
        assert os.path.exists(output_png)
        assert os.path.getsize(output_png) != 0
        mosaiced_image = cv2.imread(
            output_png, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1])

    def testColorMosaic(self):
        # Color mosaic test
        num_images = 12
        im_dir = self.write_test_data(num_images, self.color_image_shape)
        output_png = os.path.join(im_dir, "mosaic.png")
        tile_size = self.color_image_shape
        mosaic_images(
            im_dir,
            None,
            12,
            output_png,
            self.input_image_format)
        assert os.path.exists(output_png)
        assert os.path.getsize(output_png) != 0
        mosaiced_image = cv2.imread(output_png)
        assert mosaiced_image.shape == (
            math.ceil(np.sqrt(num_images)) * tile_size[0],
            math.ceil(np.sqrt(num_images)) * tile_size[1],
            3)


if __name__ == '__main__':
    tf.test.main()
