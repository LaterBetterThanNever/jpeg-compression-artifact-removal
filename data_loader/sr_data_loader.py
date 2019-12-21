from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import cv2
import tqdm
import numpy as np
from base.base_data_loader import BaseDataLoader
from utils.utils import add_jpeg_compression, pre_normalize, post_normalize
from keras.preprocessing.image import img_to_array


class SuperResolutionDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SuperResolutionDataLoader, self).__init__(config)
        self.config = config
        self.images_list = []
        self.data_list = []
        self.label_list = []

    def check_data_set(self):
        remove_list = []
        self.images_list = os.listdir(self.config.path.data_path)
        for file in self.images_list:
            if file.endswith(".jpg") \
                    or file.endswith(".jpeg") \
                    or file.endswith(".png") \
                    or file.endswith(".bmp"):
                continue
            else:
                remove_list.append(file)
        for f in remove_list:
            self.images_list.remove(f)
        print("{} images loaded!".format(len(self.images_list)))
        print("{} wrong type files removed!".format(len(remove_list)))

    def images_read_process(self):
        process_check = None
        count_to_save = 10
        for i in range(len(self.images_list)):
            data_path = os.path.join(self.config.path.data_path,
                                     self.images_list[i])
            if not os.path.exists(data_path):
                print("File {} not found!".format(data_path))
                raise FileNotFoundError
            print("processing.. ", data_path)
            label_image = cv2.imread(data_path)
            label_image = cv2.cvtColor(label_image,
                                       cv2.COLOR_BGR2YCR_CB)[:, :, 0]

            data_image = add_jpeg_compression(
                label_image, self.config.process_paras.compress_value)

            data_image = pre_normalize(data_image)
            label_image = pre_normalize(label_image)
            label_image = label_image + self.config.process_paras. \
                unsharp_mask_amount * (label_image -
                                       cv2.GaussianBlur(label_image, (0, 0),
                                                        self.config.
                                                        process_paras.
                                                        unsharp_mask_sigma))
            for x in range(0,
                           label_image.shape[0] -
                           self.config.process_paras.patch_size - 1,
                           self.config.process_paras.patch_stride):
                for y in range(0,
                               label_image.shape[1] -
                               self.config.process_paras.patch_size - 1,
                               self.config.process_paras.patch_stride):

                    hr_patch = label_image[
                               x: x + self.config.process_paras.patch_size,
                               y: y + self.config.process_paras.patch_size]
                    lr_patch = data_image[
                               x: x + self.config.process_paras.patch_size,
                               y: y + self.config.process_paras.patch_size]
                    if count_to_save > 0:
                        # print("enter to save image")
                        if process_check is None:
                            # print("process_check is None")
                            process_check = np.vstack(
                                [post_normalize(lr_patch),
                                 post_normalize(hr_patch)])
                        else:
                            temp_image = np.vstack([post_normalize(lr_patch),
                                                    post_normalize(hr_patch)])
                            process_check = np.hstack([temp_image,
                                                       process_check])
                        count_to_save -= 1
                    if count_to_save == 0:
                        cv2.imwrite(os.path.join(self.config.path.chache_path,
                                                 "process_check_image.png"),
                                    process_check,
                                    (cv2.IMWRITE_PNG_COMPRESSION, 0))
                        count_to_save -= 1

                    low_pass = cv2.blur(hr_patch, (5, 5))
                    high_pass = cv2.absdiff(hr_patch, low_pass)
                    high_pass_weight = np.average(high_pass)
                    if high_pass_weight < 0.029:
                        continue
                    self.data_list.append(img_to_array(lr_patch))
                    self.label_list.append(img_to_array(hr_patch))

    def generate_train_data(self):
        self.check_data_set()
        self.images_read_process()
        input_data = np.array(self.data_list, dtype="float")
        input_label = np.array(self.label_list, dtype="float")
        return input_data, input_label
