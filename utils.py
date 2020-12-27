import numpy as np
import os

from config import CONFIG

def calc_max_length(tensor):
    """Find the maximum length of any caption in our dataset"""
    return max(len(t) for t in tensor)

    # Load the numpy files
def map_func(img_name, caption):
    img_name_file_only = img_name.decode('utf-8').split('/')[-1]
    cahced_features_to_load = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_features', img_name_file_only + '.npy')
    img_tensor = np.load(cahced_features_to_load)
    return img_tensor, caption

  