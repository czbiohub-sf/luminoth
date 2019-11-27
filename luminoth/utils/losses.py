import tensorflow as tf

SMOOTH_L1 = "smooth_l1"
CROSS_ENTROPY = "cross_entropy"
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


def focal_loss(prediction_tensor, target_tensor, gamma=None):
    """Compute the focal loss between `logits` and the golden `target` values.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        prediction_tensor: A float tensor of shape [num_anchors, num_classes]
            representing per-label activations/logits,typically a linear output
            These activation energies are interpreted
            as unnormalized log probabilities
        target_tensor: A float tensor of shape [num_anchors, num_classes]
            representing one-hot encoded classification targets/labels
        gamma: A float32 scalar modulating
            loss from hard and easy examples.

    Returns:
    loss: A float32 Tensor of size
        [num_anchors]
        representing loss on the prediction map.
    """
    if gamma is None:
        gamma = 2.0
    epsilon = 1e-9
    y_pred = tf.nn.softmax(prediction_tensor)  # [batch_size,num_classes]

    loss = -target_tensor * \
        ((1 - y_pred) ** gamma) * \
        tf.math.log(y_pred + epsilon)
    loss = tf.reduce_sum(loss, axis=1)
    return loss
