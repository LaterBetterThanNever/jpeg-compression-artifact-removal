from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import json
from dotmap import DotMap
import importlib
import cv2
import numpy as np
import argparse


def add_jpeg_compression(image, compression_value):
    encoded_img = cv2.imencode(
        '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_value])[1]
    compressed_image = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return compressed_image


def pre_normalize(image):
    return image.astype(np.float32) / 255


def post_normalize(image):
    return np.around(np.clip(image * 255, 0, 255)).astype(np.uint8)


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = DotMap(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = "./log/"
    config.callbacks.checkpoint_dir = "./checkpoint/"
    config.path.chache_path = "./log/chache/"
    return config


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The configuration file name'
    )
    args = argparser.parse_args()
    return args


def create(cls):
    module_name, class_name = cls.rsplit(".", 1)
    try:
        print("importing..." + module_name)
        some_module = importlib.import_module(module_name)
        print("getattr..." + class_name)
        cls_instance = getattr(some_module, class_name)
        print(cls_instance)
        return cls_instance
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
        