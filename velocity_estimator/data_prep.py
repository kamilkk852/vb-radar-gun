import pickle
import numpy as np

def prep_data(data_path, test_size=5000):
    image_positions, image_diameter, velocities, slow_motion_coefs = pickle.load(open(data_path, 'rb'))

    inputs = np.concatenate([image_positions, image_diameter, np.repeat(np.expand_dims(slow_motion_coefs, axis=1), 240, axis=1)], axis=-1).astype(np.float32)
    targets = np.linalg.norm(velocities, axis=-1).max(axis=1, keepdims=True)
    max_vel = np.linalg.norm(velocities, axis=-1).max(axis=1)
    sorted_vels = np.sort(np.linalg.norm(velocities, axis=-1), axis=1)
    coef = np.where(sorted_vels[:, -5] <= 0, -1, sorted_vels[:, -5]/sorted_vels[:, -1])
    clipped_frac = np.any(inputs == -1, axis=-1).mean(axis=1)
    targets = np.clip(targets, 0, 45)

    cond = np.logical_and(np.logical_and(np.logical_and(max_vel >= 15, max_vel <= 45), coef > 0.5), clipped_frac < 0.9)
    mask = np.where(cond, np.arange(max_vel.shape[0]), -1)
    mask = mask[mask != -1]

    onlyvel_inputs, onlyvel_targets = inputs[mask], targets[mask]
    i = onlyvel_targets.shape[0]-test_size
    onlyvel_train_inputs, onlyvel_test_inputs = onlyvel_inputs[:i], onlyvel_inputs[i:]
    onlyvel_train_targets, onlyvel_test_targets = onlyvel_targets[:i], onlyvel_targets[i:]

    j = (cond.cumsum() == i).argmax()+1
    is_correct = np.expand_dims(cond.astype(np.float32), axis=-1)
    vel_targets = np.concatenate([is_correct, targets], axis=-1)
    vel_train_inputs, vel_test_inputs = inputs[:j], inputs[j:]
    vel_train_targets, vel_test_targets = vel_targets[:j], vel_targets[j:]

    return onlyvel_train_inputs, onlyvel_test_inputs, onlyvel_train_targets, onlyvel_test_targets, vel_train_inputs, vel_test_inputs, vel_train_targets, vel_test_targets