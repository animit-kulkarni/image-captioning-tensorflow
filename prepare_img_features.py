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
import utils
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model_config_dict = {'mobilenet_v2': {'model': tf.keras.applications.MobileNetV2}
                    }

def organise_data():
    """This function returns a flattened list of img, caption pairs. This is necessary since
    there are multiple captions per image. The work has been taken and altered from Tensorflow
    Image Captioning tutorial."""

    with open(CONFIG.ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = os.path.join(CONFIG.IMAGES_DIR,f'COCO_train2014_' + '%012d.jpg' % (val['image_id']))
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will 
    # lead to 30,000 examples.
    train_image_paths = image_paths[:6000]
    print('The number of captions in this training set is: ', len(train_image_paths))

    train_captions = []
    img_name_vector = []
    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    assert len(train_captions) == len(img_name_vector)
    
    #caption_filename_tuple = list(zip(train_captions, img_name_vector))

    return train_captions, img_name_vector

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

    # CAPTIONS
    train_captions, img_name_vector = organise_data()
    caption_filename_tuple = list(zip(train_captions, img_name_vector))

    with open(os.path.join(captions_dir,'caption_filename_tuple.pkl'), 'wb') as pickle_file:
        pickle.dump(caption_filename_tuple, pickle_file)

    # FEATURES
    preprocess_img_and_cache(img_name_vector,
                             features_dir,
                             model_config_dict['mobilenet_v2'])
