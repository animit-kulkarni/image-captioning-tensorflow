import numpy as np
import os
import sys
import cv2
import collections
import json
import random
import tensorflow as tf
tf.enable_eager_execution()

import re
import time
import pickle

from config import CONFIG
import utils
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model_config_dict = {'mobilenet_v2': {'model': tf.keras.applications.MobileNetV2,
                                    'input_dims': (244, 244)}
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
        image_path = os.path.join(CONFIG.IMAGES_DIR,f"COCO_train2014_{val['image_id']}.jpg")
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will 
    # lead to 30,000 examples.
    train_image_paths = image_paths[:6000]
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []
    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    assert len(train_captions) == len(img_name_vector)

    return train_captions, img_name_vector

def preprocess_img_and_cache(img_name_vector, model_config=model_config_dict['mobilenet_v2']):

    # model_config is a dict containing configs based on model chosen
    #utils.load_image(image_path, model_config)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(utils.load_image,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


    img_feature_extractor = reconfigure_cnn(model_config)

    for img, path in tqdm(image_dataset):
        batch_features = img_feature_extractor(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            #path_of_feature = p.numpy().decode("utf-8")
            #np.save(path_of_feature, bf.numpy())
            print(bf)


def reconfigure_cnn(model_config):

    model = model_config['model'](include_top=False, weights='imagenet')

    new_input = model.input
    remaining_desired_architecture = model.layers[-1].output
    reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)

    return reconfigured_cnn




if __name__ == '__main__':

    train_captions, img_name_vector = organise_data()

    # ground_truth_img = cv2.imread(img_name_vector[0])
    # cv2.imwrite('ground_truth_img.png', ground_truth_img)

    preprocess_img_and_cache(img_name_vector, model_config_dict['mobilenet_v2'])