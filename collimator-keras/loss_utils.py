import tensorflow as tf
from keras import backend as K

def K_where(condition, x, y):
    return tf.where(condition, x, y)

def K_ones(shape):
    return tf.ones(shape)

def K_zeros(shape):
    return tf.zeros(shape)

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
     
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def l2_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def l1_loss(y_true, y_pred):
    return tf.nn.l1_loss(y_true - y_pred)

#def l1_loss(predictions, targets):
#  """Implements tensorflow l1 loss.
#  Args:
#  Returns:
#  """
#  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
#      * tf.shape(targets)[3])
#  total_elements = tf.to_float(total_elements)

#  loss = tf.reduce_sum(tf.abs(predictions- targets))
#  loss = tf.div(loss, total_elements)
#  return loss


#def l2_loss(predictions, targets):
#  """Implements tensorflow l2 loss, normalized by number of elements.
#  Args:
#  Returns:
#  """
#  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
#      * tf.shape(targets)[3])
#  total_elements = tf.to_float(total_elements)

#  loss = tf.reduce_sum(tf.square(predictions-targets))
#  loss = tf.div(loss, total_elements)
#  return loss

