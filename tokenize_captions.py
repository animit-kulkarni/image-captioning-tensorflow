import tensorflow as tf
import pickle
import utils
import os
import collections
import random
from tqdm import tqdm
import numpy as np

from config import CONFIG

seed = 42
np.random.seed(seed)  

# 1. Separate the captions to train:val
# 2. tokenize each caption
# 3. reformat the image name vector to match dims 
# 4. maybe cache these somewhere


def get_caption_filename_tuple(path):
    return pickle.load(open(path, 'rb'))

def train_val_split(caption_filename_tuple, split=0.8):

    np.random.shuffle(caption_filename_tuple)
    slice_index = int(len(caption_filename_tuple) * split)
    
    train_captions = caption_filename_tuple[:slice_index]
    val_captions = caption_filename_tuple[slice_index:]

    return train_captions, val_captions


def tokenize_captions(tokenizer, caption_filename_tuple, padding='post'):
    
    captions = [x[0] for x in caption_filename_tuple]
    # Here we count the number of words in the document and top 5000 is found and indexed etc
    tokenizer.fit_on_texts(captions)
    train_seqs = tokenizer.texts_to_sequences(captions)

    # Set the pad token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    seqs = tokenizer.texts_to_sequences(captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    caption_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding=padding)

    return caption_vector

def reformat_img_name_vectors(caption_filename_tuple, caption_vector):

    img_name_vector = [x[1] for x in caption_filename_tuple]
    
    # Instantiate a dictionary to hold filename and corresponding preprocessed caption tokens
    image_caption_dict = collections.defaultdict(list)
    # This is a dict that initialises a new key with an empty list value: {a: {}, b:[] ...}
    for file, caption in zip(img_name_vector, caption_vector):
        image_caption_dict[file].append(caption)

    img_names_list = [] # this will contain a list of lists i.e. 5 of same img since 5 captions
    for img_name, caption_list in image_caption_dict.items():
        num_captions = len(caption_list)
        img_names_list.extend([img_name] * num_captions)

    assert len(img_names_list) == len(caption_vector), 'Something went wrong in the img_name_vector reformatting'
    return img_names_list


if __name__ == '__main__':

    path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')
    caption_filename_tuple = get_caption_filename_tuple(path)

    # captions_train, captions_val = train_val_split(caption_filename_tuple)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=CONFIG.VOCABULARY_TOP_K,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    caption_vectors = tokenize_captions(tokenizer, caption_filename_tuple, padding='post')

    img_names_list = reformat_img_name_vectors(caption_filename_tuple, caption_vectors)

    train_captions, val_captions = train_val_split(list(zip(img_names_list, caption_vectors)))

    print(f'The length of images: train ({len(train_captions)}), val ({len(val_captions)})')

    # This is just a sanity check to see if random seed is actually working
    if seed == 30:
        assert train_captions[0][0].split('_')[-1] == '000000247874.jpg', 'For this seed, we expected img 000000247874.jpg to be top of training set' 

    caption_cache_dir = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions')
    with open(os.path.join(caption_cache_dir,'train_captions.pkl'), 'wb') as f:
        pickle.dump(train_captions, f)
    
    with open(os.path.join(caption_cache_dir,'val_captions.pkl'), 'wb') as f:
        pickle.dump(val_captions, f)



    



    

