import numpy as np
import os
import sys
import cv2
import collections
import json
import random
import tensorflow as tf
# tf.enable_eager_execution()

import re
import time
import pickle

from config import CONFIG
from tools.timer import timer
import utils
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model_config_dict = {'mobilenet_v2': {'model': tf.keras.applications.MobileNetV2,
                                      'features_shape': 1280,
                                      'attention_features_shape': 49}
                     }


@timer
def preprocess_img_and_cache(img_name_vector, cache_dir, model_config=model_config_dict['mobilenet_v2']):

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    # Here we create a Dataset object and then rehsape into batches of (16, 7, 7, 1280)
    # Why these dims? ---> (batch_size, 7x7 grid, with 1280 feature maps) Look at mobilenet_v2 architecture

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    img_feature_extractor = reconfigure_cnn(model_config)

    for img, path in tqdm(image_dataset):
        batch_features = img_feature_extractor(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        # Iterate over each item in the batch (i.e. 16 items) and serialize
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            # the np.save allows us to cache the feature map from mobilenet_v2 saved 
            # under the file naming convention f'{original_img_name}.jpg.npy - neat!
            original_img_name = path_of_feature.split('/')[-1]
            output_filename = os.path.join(cache_dir, original_img_name)
            print(output_filename)
            np.save(output_filename, bf.numpy())

@timer
def reconfigure_cnn(model_config):

    model = model_config['model'](include_top=False, weights='imagenet')

    new_input = model.input
    remaining_desired_architecture = model.layers[-1].output
    reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)

    return reconfigured_cnn

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.imagenet_utils.preprocess_input(img)
    return img, image_path


if __name__ == '__main__':

    captions_dir = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions')
    features_dir = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_features')

    if not os.path.exists(captions_dir):
        os.mkdir(captions_dir)
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)

    # CAPTIONS
    train_captions, img_name_vector = utils.organise_data()
    caption_filename_tuple = list(zip(train_captions, img_name_vector))

    with open(os.path.join(captions_dir,'caption_filename_tuple.pkl'), 'wb') as pickle_file:
        pickle.dump(caption_filename_tuple, pickle_file)

    # FEATURES
    preprocess_img_and_cache(img_name_vector,
                             features_dir,
                             model_config_dict['mobilenet_v2'])
