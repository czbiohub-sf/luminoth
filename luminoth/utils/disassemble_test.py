import glob
import os
import tempfile

import cv2
import natsort
import numpy as np
import tensorflow as tf


from luminoth.utils.disassemble import split_mosaic, disassemble_image


class DisassembleTest(tf.test.TestCase):
    """
    Tests for disassemble
    """
    def setUp(self):
        self.tempfiles_to_delete = []
        self.gray_image_shape = [50, 40]
        self.color_image_shape = [50, 40, 3]
        self.input_image_format = ".png"

    def tearDown(self):
        tf.reset_default_graph()
        for file in self.tempfiles_to_delete:
            os.remove(file)

    def _gen_image(self, *shape):
        # Generate an image
        np.random.seed(43)
        return np.random.rand(*shape)

    def write_test_data(self, num_images, shape):
        # Write input image to test
        location = tempfile.mkdtemp()
        for i in range(num_images):
            im_filename = "test_bb_labels_{}{}".format(
                i, self.input_image_format)
            image_save_path = os.path.join(location, im_filename)
            cv2.imwrite(image_save_path, self._gen_image(*shape))
            self.tempfiles_to_delete.append(image_save_path)
        return location

    def testSplitGrayMosaic(self):
        # Test split gray mosaic returns a valid image

        # Set inputs for split_mosaic
        tile_size = [10, 8]
        im_dir = self.write_test_data(1, self.gray_image_shape)
        images = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))

        split_images = split_mosaic(
            cv2.imread(images[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR),
            tile_size)

        # Assert mosaiced_image expected shape
        assert len(split_images) == 25
        for image in split_images:
            assert image.sum() != 0
            assert image.shape == tuple(tile_size)

    def testSplitColorMosaic(self):
        # Test asssemble color mosaic returns a valid image

        # Set inputs for assemble_mosaic
        tile_size = [10, 8]
        im_dir = self.write_test_data(1, self.color_image_shape)
        images = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))

        split_images = split_mosaic(
            cv2.imread(images[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR),
            tile_size)

        # Assert mosaiced_image expected shape
        assert len(split_images) == 25
        for image in split_images:
            assert image.sum() != 0
            assert image.shape == tuple((tile_size[0], tile_size[1], 3))

    def testGrayDisassemble(self):
        # Test gray mosaic exists in the expected path with expected shape

        # Set inputs for mosaic_images
        tile_size = [10, 8]
        im_dir = self.write_test_data(1, self.gray_image_shape)
        images = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))
        output_dir = tempfile.mkdtemp()

        disassemble_image(
            images[0],
            tile_size,
            output_dir)
        split_images = natsort.natsorted(
            glob.glob(os.path.join(output_dir, "*" + self.input_image_format)))
        # Assert mosaiced_image expected shape
        assert len(split_images) == 25
        for path in split_images:
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            assert image.sum() != 0
            assert image.shape == tuple(tile_size)

    def testColorDisassemble(self):
        # Test color mosaic exists in the expected path with expected shape

        # Set inputs for mosaic_images
        tile_size = [10, 8]
        im_dir = self.write_test_data(1, self.color_image_shape)
        images = natsort.natsorted(
            glob.glob(os.path.join(im_dir, "*" + self.input_image_format)))
        output_dir = tempfile.mkdtemp()

        disassemble_image(
            images[0],
            tile_size,
            output_dir)
        split_images = natsort.natsorted(
            glob.glob(os.path.join(output_dir, "*" + self.input_image_format)))
        # Assert mosaiced_image expected shape
        assert len(split_images) == 25
        for path in split_images:
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            assert image.sum() != 0
            assert image.shape == tuple((tile_size[0], tile_size[1], 3))


if __name__ == '__main__':
    tf.test.main()
