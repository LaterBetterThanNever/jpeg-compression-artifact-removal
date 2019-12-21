from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

from data_loader.sr_data_loader import SuperResolutionDataLoader
from models.sr_model import SuperResolutionModel
from trainer.sr_trainer import SuperResolutionTrainer
from utils.utils import process_config, create_dirs, get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        ktf.set_session(sess)

        create_dirs([config.callbacks.tensorboard_log_dir,
                     config.callbacks.checkpoint_dir,
                     config.path.chache_path])
        print("Create the data generator.")
        data_loader = SuperResolutionDataLoader(config)
        print("Create the model.")
        model = SuperResolutionModel(config)
        print("Create the trainer.")
        trainer = SuperResolutionTrainer(model.model,
                                         data_loader.generate_train_data(),
                                         config)
        print("Start training...!")
        trainer.train()

    except Exception as err:
        print("missing or invalid arguments: {0}".format(err))
        exit(0)


if __name__ == '__main__':
    main()