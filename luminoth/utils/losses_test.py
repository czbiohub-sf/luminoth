import math

import numpy as np
import tensorflow as tf

from luminoth.utils.losses import (
    smooth_l1_loss, focal_loss
)


def _logit(probability):
    return math.log(probability / (1. - probability))


class LossesTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def testSmoothL1LossImperfectScore(self):
        # Test smooth l1 loss for an imperfect case
        # Set inputs for smooth_l1_loss
        all_zeros = [0., 0., 0., 0.]
        all_ones = [1., 1., 1., 1.]

        bbox_prediction_tf = tf.placeholder(tf.float32)
        bbox_target_tf = tf.placeholder(tf.float32)
        loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)

        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    bbox_prediction_tf: [all_zeros],
                    bbox_target_tf: [all_ones]
                })
            self.assertAlmostEqual(loss, 4., delta=0.4)

    def testSmoothL1LossPerfectScore(self):
        # Test smooth l1 loss for a perfect case
        # Set inputs for smooth_l1_loss
        all_ones = [1., 1., 1., 1.]

        bbox_prediction_tf = tf.placeholder(tf.float32)
        bbox_target_tf = tf.placeholder(tf.float32)
        loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)

        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    bbox_prediction_tf: [all_ones],
                    bbox_target_tf: [all_ones]
                })
            self.assertAlmostEqual(loss, 0., delta=0.4)

    def testSmoothL1LossRandom(self):
        # Test smooth l1 loss for random case
        # Set inputs for smooth_l1_loss
        random_prediction = [
            0.47450006, -0.80413032, -0.26595005, 0.17124325]
        random_target = [
            0.10058594, 0.07910156, 0.10555581, -0.1224325]

        bbox_prediction_tf = tf.placeholder(tf.float32)
        bbox_target_tf = tf.placeholder(tf.float32)
        loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)

        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    bbox_prediction_tf: [random_prediction],
                    bbox_target_tf: [random_target]
                })
            self.assertAlmostEqual(loss, 2, delta=0.4)

    def testFocalLossPerfectScore(self):
        # Test focal loss for a perfect case where logit probabilities are
        # higher for the expected classes
        # Set inputs for focal_loss
        logits_array = np.transpose(np.array([
            [_logit(0.55), _logit(0.52), _logit(0.50), _logit(0.48),
             _logit(0.45)],
            [_logit(0.95), _logit(0.82), _logit(0.80), _logit(0.28),
             _logit(0.35)]], dtype=np.float32))
        labels_array = np.transpose(np.array([
            [0, 0, 0, 1, 1], [1, 1, 1, 0, 0]], dtype=np.float32))
        prediction_tensor = tf.placeholder(tf.float32)
        target_tensor = tf.placeholder(tf.float32)
        loss_tf = focal_loss(prediction_tensor, target_tensor)

        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    prediction_tensor: logits_array,
                    target_tensor: labels_array
                })
            expected_loss = [
                0.00022774, 0.00787424, 0.00892574, 0.03088996, 0.07966313]

            self.assertAllClose(loss, expected_loss)

    def testFocalLossImperfectScore(self):
        # Test focal loss for a perfect case where logit probabilities are
        # lower for the expected classes
        # Set inputs for focal_loss
        logits_array = np.transpose(np.array([
            [_logit(0.55), _logit(0.52), _logit(0.50), _logit(0.48),
             _logit(0.45)],
            [_logit(0.95), _logit(0.82), _logit(0.80), _logit(0.28),
             _logit(0.35)]], dtype=np.float32))
        labels_array = np.transpose(np.array([
            [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.float32))
        prediction_tensor = tf.placeholder(tf.float32)
        target_tensor = tf.placeholder(tf.float32)
        loss_tf = focal_loss(prediction_tensor, target_tensor)

        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    prediction_tensor: logits_array,
                    target_tensor: labels_array
                })
            expected_loss = [
                2.4771614, 1.0766783, 1.0300404, 0.60194975, 0.33609733]

            self.assertAllClose(loss, expected_loss)
