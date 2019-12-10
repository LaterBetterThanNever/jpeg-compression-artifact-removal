from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from base.base_model import BaseModel
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Add, Concatenate, Input, Lambda


class SuperResolutionModel(BaseModel):
    def __init__(self, config):
        super(SuperResolutionModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        lr_input = Input(shape=(None, None, 1))
        s2d = Lambda(lambda t: tf.space_to_depth(t, 8), name="s2d")(lr_input)
        s2d_conv1 = Conv2D(64, (1, 1), padding='same', name='dct', use_bias=None, trainable=False, activation='relu')(
            s2d)
        s2d_conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(s2d_conv1)
        conv_1 = Conv2D(8, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(lr_input)
        conv_2 = Conv2D(8, (3, 3), padding='same', activation='relu', dilation_rate=(4, 4))(lr_input)
        conv_3 = Conv2D(8, (3, 3), padding='same', activation='relu', dilation_rate=(8, 8))(lr_input)
        concate_1 = Concatenate()([conv_1, conv_2, conv_3])
        conv_4 = Conv2D(16, (3, 3), strides=2, padding='same')(concate_1)
        conv_5 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_4)
        conv_6 = Conv2D(64, (3, 3), strides=2, padding='same')(conv_5)
        conv_7 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_6)
        conv_8 = Conv2D(64, (3, 3), strides=2, padding='same')(conv_7)
        add_1 = Add()([s2d_conv2, conv_8])
        conv3 = Conv2D(64, (1, 1), padding='same', name='rdct', use_bias=None, trainable=False)(add_1)
        output = Lambda(lambda t: tf.depth_to_space(t, 8), name='output')(conv3)
        model = Model(lr_input, output)
        self.model = model
        self.model.compile(
            loss='mse',
            optimizer=self.config.model.optimizer,
            metrics=['PSNR']
        )
