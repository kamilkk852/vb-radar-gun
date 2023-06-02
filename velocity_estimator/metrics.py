import tensorflow as tf


def binary_crossentropy_loss(target, output):
    max_vel = output[:, 1:]
    target_max_vel = target[:, 1:]
    mape_vals = 100*tf.math.abs(target_max_vel-max_vel)/(target_max_vel+1e-5)

    is_correct = tf.cast(tf.math.logical_and(target[:, :1] == 1., mape_vals < 25), tf.float32)
    return tf.keras.metrics.binary_crossentropy(is_correct, output[:, :1])

def mape(target, output):
    max_vel = tf.clip_by_value(output, 15, 45)
    target_max_vel = tf.clip_by_value(target, 15, 45)
    ape = 100*tf.math.abs(target_max_vel-max_vel)/target_max_vel
    
    return ape

def masked_mape(target, output):
    max_vel = tf.clip_by_value(output[:, 1:], 15, 45)
    target_max_vel = tf.clip_by_value(target[:, 1:], 15, 45)
    mape_vals = 100*tf.math.abs(target_max_vel-max_vel)/target_max_vel
    mask = (target[:, :1] == 1.)
    
    return mape_vals[mask]

class ROC_AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, *args, **kwargs):
        target, output = y_true, y_pred
        max_vel = output[:, 1:]
        target_max_vel = target[:, 1:]
        mape_vals = 100*tf.math.abs(target_max_vel-max_vel)/(target_max_vel+1e-5)

        is_correct = tf.cast(tf.math.logical_and(target[:, :1] == 1., mape_vals < 25), tf.float32)

        return super(ROC_AUC, self).update_state(is_correct, output[:, :1], *args, **kwargs)
    
class Label_ROC_AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, *args, **kwargs):
        target, output = y_true, y_pred
        is_correct = (target[:, :1] == 1.)

        return super(Label_ROC_AUC, self).update_state(is_correct, output[:, :1], *args, **kwargs)
    
class Err_ROC_AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, *args, **kwargs):
        target, output = y_true, y_pred
        max_vel = output[:, 1:]
        target_max_vel = target[:, 1:]
        mape_vals = 100*tf.math.abs(target_max_vel-max_vel)/(target_max_vel+1e-5)

        is_correct = (mape_vals < 25)

        return super(Err_ROC_AUC, self).update_state(is_correct, output[:, :1], *args, **kwargs) 