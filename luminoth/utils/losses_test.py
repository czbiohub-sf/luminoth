import tensorflow as tf

from luminoth.utils.losses import (
    smooth_l1_loss, focal_loss
)


class LossesTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.all_zeros = [0.0, 0.0, 0.0, 0.0]
        self.all_ones = [1.0, 1.0, 1.0, 1.0]
        self.random_prediction = [0.47450006, -0.80413032, -0.26595005, 0.17124325]
        self.random_target = [0.10058594, 0.07910156, 0.10555581, -0.1224325]

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

    def test_focal_loss(self):
        logits = tf.placeholder(tf.float32)
        target = tf.placeholder(tf.float32)
        loss_tf = focal_loss(logits, target)
        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    logits: [self.all_zeros],
                    target: [self.all_ones]
                })
            self.assertAlmostEqual(loss, 0.1732868, delta=0.001)

    def test_focal_loss_random(self):
        logits = tf.placeholder(tf.float32)
        target = tf.placeholder(tf.float32)
        loss_tf = focal_loss(logits, target)
        with tf.Session() as sess:
            loss = sess.run(
                loss_tf,
                feed_dict={
                    logits: [self.random_prediction],
                    target: [self.random_target]
                })
            self.assertAlmostEqual(loss, 0.5454119, delta=0.001)
