from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
from keras.optimizers import Adam
from base.base_model import BaseModel
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Add, Concatenate, Input, Lambda


class SuperResolutionModel(BaseModel):
    def __init__(self, config):
        super(SuperResolutionModel, self).__init__(config)
        self.build_model()
        self.set_dct_layer()

    def build_model(self):
        lr_input = Input(shape=(None, None, 1))
        s2d = Lambda(lambda t: tf.space_to_depth(t, 8), name='i')(lr_input)
        s2d_conv1 = Conv2D(64, (1, 1), padding='same', name='dct',
                           use_bias=None, trainable=False)(s2d)
        s2d_conv1 = LeakyReLU(alpha=0.1)(s2d_conv1)
        s2d_conv2 = Conv2D(64, (3, 3), padding='same',
                           activation='relu')(s2d_conv1)

        conv_1 = Conv2D(8, (3, 3), padding='same', activation='relu',
                        dilation_rate=(2, 2))(lr_input)
        conv_2 = Conv2D(8, (3, 3), padding='same', activation='relu',
                        dilation_rate=(4, 4))(lr_input)
        conv_3 = Conv2D(8, (3, 3), padding='same', activation='relu',
                        dilation_rate=(8, 8))(lr_input)
        concat_1 = Concatenate()([conv_1, conv_2, conv_3])

        conv_4 = Conv2D(16, (3, 3), strides=2, padding='same')(concat_1)
        conv_5 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_4)
        conv_6 = Conv2D(64, (3, 3), strides=2, padding='same')(conv_5)
        conv_7 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_6)
        conv_8 = Conv2D(64, (3, 3), strides=2, padding='same')(conv_7)
        add_1 = Add()([s2d_conv2, conv_8])

        conv4 = Conv2D(64, (1, 1), padding='same', name='rdct', use_bias=None,
                       trainable=False)(add_1)

        output = Lambda(lambda t: tf.depth_to_space(t, 8), name='o')(conv4)
        model = Model(lr_input, output)

        self.model = model
        self.model.compile(
            loss='mae',
            optimizer=Adam(lr=self.config.trainer.learning_rate,
                           decay=self.config.trainer.learning_rate / self.config.trainer.num_epochs),
            # metrics=['PSNR']
        )

    def set_dct_layer(self, width=8):
        matrix = np.ones((width, width), dtype=np.float32)
        for i in range(width):
            for j in range(width):
                matrix[i, j] = math.cos(
                    ((2 * j + 1) * i * math.pi) / (width * 2))
                if i == 0:
                    matrix[i, j] = matrix[i, j] * math.sqrt(1 / width)
                else:
                    matrix[i, j] = matrix[i, j] * math.sqrt(2 / width)
        matrix_t = np.transpose(matrix)
        weights = np.ones((width * width, width * width), dtype=np.float32)
        for a in range(width):
            for b in range(width):
                for j in range(width):
                    weights[width * j:width * (j + 1), width * a + b] = \
                        np.multiply(matrix_t[:, b], matrix[a, j])
        weights = np.reshape(weights, (1, 1, width * width, width * width))

        weights_reverse = np.ones((width * width, width * width),
                                  dtype=np.float32)
        for a in range(width):
            for b in range(width):
                for j in range(width):
                    weights_reverse[width * j:width * (j + 1), width * a + b] \
                        = np.multiply(matrix[:, b], matrix_t[a, j])

        weights_reverse = np.reshape(weights_reverse,
                                     (1, 1, width * width, width * width))
        self.model.get_layer('dct').set_weights([weights])
        self.model.get_layer('rdct').set_weights([weights_reverse])

