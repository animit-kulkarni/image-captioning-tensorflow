import numpy as np
import os
import json
import collections
import cv2

from config import CONFIG
from prepare_img_features import load_image


seed = 42
np.random.seed(seed)  

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
    np.random.shuffle(image_paths)

    # Select the first e.g. 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will 
    # lead to 30,000 examples.
    train_image_paths = image_paths[:CONFIG.NUMBER_OF_IMAGES]
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

def calc_max_length(tensor):
    """Find the maximum length of any caption in our dataset"""
    return max(len(t) for t in tensor)

    # Load the numpy files
def map_func(img_name, caption):
    img_name_file_only = img_name.decode('utf-8').split('/')[-1]
    cached_features_to_load = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_features', img_name_file_only + '.npy')
    img_tensor = np.load(cached_features_to_load)
    return img_tensor, caption

def map_func_including_cnn(img_name, caption):
    img_name_file_only = img_name.decode('utf-8').split('/')[-1]
    path_to_file = os.path.join(CONFIG.IMAGES_DIR, img_name_file_only)
    img_tensor, _ = load_image(path_to_file)


    return img_tensor, caption



  