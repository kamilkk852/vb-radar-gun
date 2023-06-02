import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from fire import Fire
from data_prep import prep_data
from model import OnlyVelocityModel, VelocityModel
from metrics import binary_crossentropy_loss, mape, masked_mape, ROC_AUC, Label_ROC_AUC, Err_ROC_AUC


def train(data_path, test_size=5000, batch_size=32):
    tf.keras.utils.set_random_seed(0)

    onlyvel_train_inputs, onlyvel_test_inputs, onlyvel_train_targets, onlyvel_test_targets,\
        vel_train_inputs, vel_test_inputs, vel_train_targets, vel_test_targets = prep_data(data_path, test_size=test_size)

    print(onlyvel_train_targets.shape)
    onlyvel_model = OnlyVelocityModel()
    onlyvel_model.build((None, 240, 4))
    print(onlyvel_model.summary())

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[4000, 15000, 30000], values=[3e-4, 1e-4, 3e-5, 1e-5]
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "onlyvel_checkpoints",
        monitor="val_mape",
        save_best_only=True,
        verbose=1,
        mode='min'
    )
    onlyvel_model.compile(loss='mse',
                        metrics=[mape],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    history = onlyvel_model.fit(onlyvel_train_inputs,
                                onlyvel_train_targets,
                                batch_size=batch_size,
                                epochs=15,
                                validation_data=(onlyvel_test_inputs, onlyvel_test_targets),
                                callbacks=[checkpoint_callback])
    onlyvel_model.load_weights("onlyvel_checkpoints")

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("MAPE [%]")
    plt.plot(np.arange(history.params['epochs'])+1, history.history['mape'], '-', label='Training')
    plt.plot(np.arange(history.params['epochs'])+1, history.history['val_mape'], '-', label='Validation')
    plt.savefig("training_history.png")

    vel_model = VelocityModel()
    vel_model.build((None, 240, 4))
    for name in ['input_encoding', 'transformer_block_0',
                'transformer_block_1', 'transformer_block_2', 'transformer_block_3',
                'transformer_block_4', 'transformer_block_5', 'output_encoder',
                'output_regressor']:
        vel_model.get_layer(name).set_weights(onlyvel_model.get_layer(name).get_weights())
        vel_model.get_layer(name).trainable = False
    print(vel_model.summary())
        
    vel_model.compile(loss=binary_crossentropy_loss,
                    metrics=[masked_mape, ROC_AUC(), Label_ROC_AUC(), Err_ROC_AUC()],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    vel_model.fit(vel_train_inputs,
                vel_train_targets,
                batch_size=batch_size,
                epochs=1,
                validation_data=(vel_test_inputs, vel_test_targets))
    _, val_mape, val_rocauc, _, _ = vel_model.evaluate(x=vel_test_inputs, y=vel_test_targets)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")
    weights_filename = f"{timestamp}_mape{str(round(val_mape)).zfill(3)}_rocauc{round(1000*val_rocauc)}_weights.h5"
    model_filename = f"velocity_model_{timestamp}/"
    vel_model.save_weights(weights_filename)
    tf.keras.saving.save_model(vel_model, model_filename)

if __name__ == "__main__":
    Fire(train)
