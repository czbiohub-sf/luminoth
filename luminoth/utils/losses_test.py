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
        self.random_prediction = [
            0.47450006, -0.80413032, -0.26595005, 0.17124325]
        self.random_target = [
            0.10058594, 0.07910156, 0.10555581, -0.1224325]

    def test_smooth_l1_loss(self):
        bbox_prediction_tf = tf.placeholder(tf.float32)
        bbox_target_tf = tf.placeholder(tf.float32)
        loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)
        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    bbox_prediction_tf: [self.all_zeros],
                    bbox_target_tf: [self.all_ones]
                })
            self.assertAlmostEqual(loss, 4., delta=0.4)

    def test_smooth_l1_loss_random(self):
        bbox_prediction_tf = tf.placeholder(tf.float32)
        bbox_target_tf = tf.placeholder(tf.float32)
        loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)
        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    bbox_prediction_tf: [self.random_prediction],
                    bbox_target_tf: [self.random_target]
                })
            self.assertAlmostEqual(loss, 2, delta=0.4)

    def test_focal_loss_ignore_positive_example_loss_via_alpha(self):

        with tf.Session() as sess:

            logits_tf = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
            labels_tf = tf.constant([[[2],
                                      [1],
                                      [1],
                                      [2],
                                      [0]]], tf.float32)

            loss = np.squeeze(sess.run(
                focal_loss(
                    prediction_tensor=logits_tf,
                    target_tensor=labels_tf,
                    gamma=2.0,
                    alpha=0.0)))
            print(loss)

            # self.assertAllClose(loss[:3], [0., 0., 0.])

    def test_focal_loss_perfect_score(self):

        with tf.Session() as sess:

            logits_tf = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
            labels_tf = tf.constant([[[1],
                                      [1],
                                      [1],
                                      [0],
                                      [3]]], tf.float32)

            loss = np.squeeze(sess.run(
                focal_loss(
                    prediction_tensor=logits_tf,
                    target_tensor=labels_tf,
                    gamma=2.0,
                    alpha=0.0)))
            print(loss)

            # self.assertAllClose(loss[:3], [0., 0., 0])

    def test_focal_loss_ignore_negative_example_loss_via_alpha(self):

        with tf.Session() as sess:

            logits_tf = tf.constant([[[_logit(0.55)],
                                      [_logit(0.52)],
                                      [_logit(0.50)],
                                      [_logit(0.48)],
                                      [_logit(0.45)]]], tf.float32)
            labels_tf = tf.constant([[[1],
                                      [1],
                                      [1],
                                      [0],
                                      [2]]], tf.float32)

            loss = np.squeeze(sess.run(
                focal_loss(
                    prediction_tensor=logits_tf,
                    target_tensor=labels_tf,
                    gamma=2.0,
                    alpha=1.0)))
            print(loss)

            # self.assertAllClose(loss[3:], [0., 0.])
