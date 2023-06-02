import tensorflow as tf
import numpy as np
from fire import Fire
from sklearn.metrics import roc_curve, roc_auc_score
from data_generator import TrajectoriesGenerator
from metrics import masked_mape

tf.keras.utils.set_random_seed(0)

def unclipped_ape(target, output):
    max_vel = output[:, 1:]
    target_max_vel = target[:, 1:]
    ape_vals = 100*np.abs(target_max_vel-max_vel)/(target_max_vel+1e-5)

    return ape_vals

def evaluate(model_path,
             n_samples=2000,
             n_frames=240,
             angle_xy_range=(-20, 20),
            angle_y_range=(-20, 20),
            slow_motion_coefs=[4, 8],
            ball_size_range=(7/640, 1/8),
            velocity_range=(15, 45),
            avg_hits=3.,
            pos_std_err=0.002,
            d_std_err=0.15,
            random_object_prob=0.15,
            no_object_prob=0.5,
            hit_max_z_angle=15,
            padding_size=0.1,
            clean=False):
    image_positions, image_diameter, velocities, slow_motion_coefs = \
        TrajectoriesGenerator(n_samples=n_samples,
                                n_frames=n_frames,
                                angle_xy_range=angle_xy_range,
                                angle_y_range=angle_y_range,
                                slow_motion_coefs=slow_motion_coefs,
                                ball_size_range=ball_size_range,
                                velocity_range=velocity_range,
                                avg_hits=avg_hits,
                                pos_std_err=pos_std_err,
                                d_std_err=d_std_err,
                                random_object_prob=random_object_prob,
                                no_object_prob=no_object_prob,
                                hit_max_z_angle=hit_max_z_angle,
                                padding_size=padding_size)(add_noise=(not clean))
    
    inputs = np.concatenate([image_positions, image_diameter, np.repeat(np.expand_dims(slow_motion_coefs, axis=1), 240, axis=1)], axis=-1).astype(np.float32)
    targets = np.linalg.norm(velocities, axis=-1).max(axis=1, keepdims=True)
    max_vel = np.linalg.norm(velocities, axis=-1).max(axis=1)
    sorted_vels = np.sort(np.linalg.norm(velocities, axis=-1), axis=1)
    coef = np.where(sorted_vels[:, -5] < 0, -1, sorted_vels[:, -5]/sorted_vels[:, -1])
    clipped_frac = np.any(inputs == -1, axis=-1).mean(axis=1)
    is_correct = np.logical_and(np.logical_and(np.logical_and(max_vel >= 15, max_vel <= 45), coef > 0.5), clipped_frac < 0.9)
    is_correct = np.expand_dims(is_correct.astype(np.float32), axis=-1)
    targets = np.concatenate([is_correct, np.clip(targets, 0, 45)], axis=-1)

    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(inputs)
    mape_vals = unclipped_ape(targets, preds)
    class_targets = np.logical_and(targets[:, 0] == 1, mape_vals[:, 0] < 2500).astype(np.int32)
    roc_auc_val = roc_auc_score(class_targets, preds[:, 0])
    fpr, tpr, thres = roc_curve(class_targets, preds[:, 0])
    mape_vals = masked_mape(tf.convert_to_tensor(targets, dtype=tf.float32),
                            tf.convert_to_tensor(preds, dtype=tf.float32)).numpy()

    print('Correct samples count:', int(is_correct.sum()))
    print("MAPE:", round(mape_vals.mean(), 1), '%')
    print("Median APE:", round(np.median(mape_vals), 1), '%')
    print("90% APE", round(np.percentile(mape_vals, 90), 1), '%')
    print("95% APE", round(np.percentile(mape_vals, 95), 1), '%')
    print('ROC AUC:', round(roc_auc_val, 3))
    print("FPR at TPR=90%:", round(100*fpr[tpr > 0.9][0]), '%')
    print("Threshold at TPR=90%:", round(thres[tpr > 0.9][0], 3))
    
if __name__ == "__main__":
    Fire(evaluate)

    
