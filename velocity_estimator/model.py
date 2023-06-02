import tensorflow as tf
import numpy as np

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, tf_dim, num_heads, ff_dim, *args, rate=0., **kwargs):
        super(TransformerBlock, self).__init__(*args, **kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, input_dim, attention_axes=(1,))
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(tf_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.emb_layer = tf.keras.layers.Dense(tf_dim)

    def call(self, inputs, training, return_attention_scores=False):
        if return_attention_scores:
            attn_output, weights = self.att(inputs, inputs, return_attention_scores=True)
        else:
            attn_output = self.att(inputs, inputs, return_attention_scores=False)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        if return_attention_scores:
            return self.layernorm2(out1 + ffn_output), weights
        return self.layernorm2(out1 + ffn_output)
    
class OnlyVelocityModel(tf.keras.models.Model):
    def __init__(self, n_blocks=6, ff_dim=256, input_dim=256, tf_dim=256, att_heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_encoding = tf.keras.layers.Conv1D(tf_dim, 1, activation='relu', name='input_encoding')
        self.transformer_blocks = [TransformerBlock(input_dim, tf_dim, att_heads, ff_dim, rate=0., name=f"transformer_block_{i}") for i in range(n_blocks)]
        self.output_encoder = tf.keras.layers.Conv1D(256, 1, activation='relu', name='output_encoder')
        self.output_regressor = tf.keras.layers.Dense(1, name='output_regressor')
        
    def call(self, x, return_attention_scores=False, training=None):
        n_samples = tf.shape(x)[0]
        im_poses = x[:, :, :2]
        im_diameter = x[:, :, 2:3]
        clipped_diameter = tf.clip_by_value(im_diameter, 7/640, 1/8)
        slow_motion_coef = x[:, :, 3:4]
        correct_mask = tf.cast(x != -1, tf.float32)
        x = tf.clip_by_value(x, 0, 1)
        x = tf.concat([im_poses, 8*im_diameter, im_poses/(8*clipped_diameter), 1/tf.math.tan(np.pi/4*clipped_diameter)/10], axis=-1)
        correct_mask = tf.cast(tf.math.reduce_any(im_poses != -1, axis=-1, keepdims=True), tf.float32)
        diff_mask = tf.math.logical_or(correct_mask[:, 1:] == 0., correct_mask[:, :-1] == 0.)
        diff = tf.concat([tf.zeros((n_samples, 1, 2)), tf.where(diff_mask, 0., im_poses[:, 1:]-im_poses[:, :-1])], axis=1) #pos diff
        diff = tf.where(im_diameter != -1, diff*(30*slow_motion_coef)/clipped_diameter*0.21, 0)
        diff = tf.concat([diff/30, tf.where(diff > 300, 0., diff/30), tf.cast(diff > 300., tf.float32)], axis=-1)
            
        prep_x = tf.concat([x, correct_mask, diff, slow_motion_coef/8], axis=-1)
        x = self.input_encoding(prep_x)
        att_weights = [None]*len(self.transformer_blocks)
        for i, layer in enumerate(self.transformer_blocks):
            if not return_attention_scores:
                x = layer(x, training=training, return_attention_scores=False)
            else:
                x, att_weights[i] = layer(x, training=training, return_attention_scores=True)
            
        vel = self.output_encoder(x)
        vel = tf.math.reduce_max(vel, axis=1)
        vel = self.output_regressor(vel)
        
        output = vel
        if not return_attention_scores:
            return output
        else:
            return output, tf.stack(att_weights, axis=0)
        
class VelocityModel(tf.keras.models.Model):
    def __init__(self, n_blocks=6, ff_dim=256, input_dim=256, tf_dim=256, att_heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_encoding = tf.keras.layers.Conv1D(tf_dim, 1, activation='relu', name='input_encoding')
        self.transformer_blocks = [TransformerBlock(input_dim, tf_dim, att_heads, ff_dim, rate=0., name=f"transformer_block_{i}") for i in range(n_blocks)]
        self.output_encoder = tf.keras.layers.Conv1D(256, 1, activation='relu', name='output_encoder')
        self.output_regressor = tf.keras.layers.Dense(1, name='output_regressor')
        
        self.class_output_encoder = tf.keras.Sequential([tf.keras.layers.Conv1D(256, 1, activation='relu', name='classification_conv1'), tf.keras.layers.Conv1D(512, 3, activation='relu', name='classification_conv2')])
        self.class_output_layer = tf.keras.layers.Dense(1, name='classifier')
        
    def call(self, x, return_attention_scores=False, training=None):
        n_samples = tf.shape(x)[0]
        im_poses = x[:, :, :2]
        im_diameter = x[:, :, 2:3]
        clipped_diameter = tf.clip_by_value(im_diameter, 7/640, 1/8)
        slow_motion_coef = x[:, :, 3:4]
        correct_mask = tf.cast(x != -1, tf.float32)
        x = tf.clip_by_value(x, 0, 1)
        x = tf.concat([im_poses, 8*im_diameter, im_poses/(8*clipped_diameter), 1/tf.math.tan(np.pi/4*clipped_diameter)/10], axis=-1)
        correct_mask = tf.cast(tf.math.reduce_any(im_poses != -1, axis=-1, keepdims=True), tf.float32)
        diff_mask = tf.math.logical_or(correct_mask[:, 1:] == 0., correct_mask[:, :-1] == 0.)
        diff = tf.concat([tf.zeros((n_samples, 1, 2)), tf.where(diff_mask, 0., im_poses[:, 1:]-im_poses[:, :-1])], axis=1) #pos diff
        diff = tf.where(im_diameter != -1, diff*(30*slow_motion_coef)/clipped_diameter*0.21, 0)
        diff = tf.concat([diff/30, tf.where(diff > 300, 0., diff/30), tf.cast(diff > 300., tf.float32)], axis=-1)
            
        prep_x = tf.concat([x, correct_mask, diff, slow_motion_coef/8], axis=-1)
        class_input = 1.*prep_x
        x = self.input_encoding(prep_x)
        att_weights = [None]*len(self.transformer_blocks)
        for i, layer in enumerate(self.transformer_blocks):
            if not return_attention_scores:
                x = layer(x, training=training, return_attention_scores=False)
            else:
                x, att_weights[i] = layer(x, training=training, return_attention_scores=True)
            class_input = tf.concat([class_input, x], axis=-1)
            
        vel = self.output_encoder(x)
        vel = tf.math.reduce_max(vel, axis=1)
        vel = self.output_regressor(vel)
        
        correct = self.class_output_encoder(class_input)
        correct = tf.math.reduce_max(correct, axis=1)
        correct = tf.math.sigmoid(self.class_output_layer(correct))
        
        output = tf.concat([correct, vel], axis=-1)
        if not return_attention_scores:
            return output
        else:
            return output, tf.stack(att_weights, axis=0)
