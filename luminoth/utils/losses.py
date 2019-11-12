import tensorflow as tf

SMOOTH_L1 = "smooth_l1"
FOCAL = "focal"


def smooth_l1_loss(bbox_prediction, bbox_target, sigma=3.0):
    """
    Return Smooth L1 Loss for bounding box prediction.

    Args:
        bbox_prediction: shape (1, H, W, num_anchors * 4)
        bbox_target:     shape (1, H, W, num_anchors * 4)


    Smooth L1 loss is defined as:

    0.5 * x^2                  if |x| < d
    abs(x) - 0.5               if |x| >= d

    Where d = 1 and x = prediction - target

    """
    sigma2 = sigma ** 2
    diff = bbox_prediction - bbox_target
    abs_diff = tf.abs(diff)
    abs_diff_lt_sigma2 = tf.less(abs_diff, 1.0 / sigma2)
    bbox_loss = tf.reduce_sum(
        tf.where(
            abs_diff_lt_sigma2,
            0.5 * sigma2 * tf.square(abs_diff),
            abs_diff - 0.5 / sigma2
        ), [1]
    )
    return bbox_loss


def focal_loss(logits, targets, alpha=None, gamma=None, normalizer=1.0):
    """Compute the focal loss between `logits` and the golden `target` values.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size
          [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size
          [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying
        alpha to the loss from positive examples
          and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating
        loss from hard and easy examples.
        normalizer: A float32 scalar normalizes
        the total loss from all examples.

    Returns:
    loss: A float32 Tensor of size
        [batch, height_in, width_in, num_predictions]
        representing normalized loss on the prediction map.
    """
    with tf.name_scope('focal_loss'):
        if alpha is None:
            alpha = 0.25
        if gamma is None:
            gamma = 2.0

        positive_label_mask = tf.math.equal(targets, 1.0)
        cross_entropy = (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=logits))
        # Below are comments/derivations for computing modulator.
        # For brevity,
        # let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
        # for positive samples and 1 - sigmoid(x) for negative examples.
        #
        # The modulator, defined as (1 - P_t)^r,
        # is a critical part in focal loss
        # computation. For r > 0,
        # it puts more weights on hard examples, and less
        # weights on easier ones. However if
        # it is directly computed as (1 - P_t)^r,
        # its back-propagation is not stable when r < 1.
        # The implementation here
        # resolves the issue.
        #
        # For positive samples (labels being 1),
        #    (1 - p_t)^r
        #  = (1 - sigmoid(x))^r
        #  = (1 - (1 / (1 + exp(-x))))^r
        #  = (exp(-x) / (1 + exp(-x)))^r
        #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
        #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
        #  = exp(- r * x - r * log(1 + exp(-x)))
        #
        # For negative samples (labels being 0),
        #    (1 - p_t)^r
        #  = (sigmoid(x))^r
        #  = (1 / (1 + exp(-x)))^r
        #  = exp(log((1 / (1 + exp(-x)))^r))
        #  = exp(-r * log(1 + exp(-x)))
        #
        # Therefore one unified form for positive (z = 1) and negative (z = 0)
        # samples is:
        #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
        neg_logits = -1.0 * logits
        modulator = tf.math.exp(
            gamma * targets * neg_logits -
            gamma * tf.math.log1p(tf.math.exp(neg_logits)))
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss,
                                 (1.0 - alpha) * loss)
        weighted_loss /= normalizer
        weighted_loss = tf.reduce_sum(weighted_loss)
    return weighted_loss
