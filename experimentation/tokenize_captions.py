import tensorflow as tf
import pickle
import utils
import os
import collections
import random
from tqdm import tqdm
import numpy as np
import logging

from config import CONFIG
from tools.timer import timer

CONFIG = CONFIG()
logger = logging.getLogger(__name__)
seed = 42
np.random.seed(seed)  

# 1. Separate the captions to train:val
# 2. tokenize each caption
# 3. reformat the image name vector to match dims 
# 4. maybe cache these somewhere

class TokensManager:

    def __init__(self):

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=CONFIG.VOCABULARY_TOP_K,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    @timer
    def prepare_imgs_tokens(self, path):

        caption_filename_tuple = self._get_caption_filename_tuple(path)
        caption_vectors = self._tokenize_captions(caption_filename_tuple, padding='post')
        img_names_list = self._reformat_img_name_vectors(caption_filename_tuple, caption_vectors)
        train_captions, val_captions = self._train_val_split(list(zip(img_names_list, caption_vectors)))

        logger.info(f'The length of images: train ({len(train_captions)}), val ({len(val_captions)})')

        return train_captions, val_captions

    def _get_caption_filename_tuple(self, path):
        return pickle.load(open(path, 'rb'))

    def _train_val_split(self, caption_filename_tuple, split=0.8):

        np.random.shuffle(caption_filename_tuple)
        slice_index = int(len(caption_filename_tuple) * split)
        
        train_captions = caption_filename_tuple[:slice_index]
        val_captions = caption_filename_tuple[slice_index:]

        return train_captions, val_captions

    @timer
    def _tokenize_captions(self, caption_filename_tuple, padding='post'):
        
        captions = [x[0] for x in caption_filename_tuple]
        
        # Here we count the number of words in the document and top 5000 is found and indexed etc
        self.tokenizer.fit_on_texts(captions)
        train_seqs = self.tokenizer.texts_to_sequences(captions)

        # Set the pad token
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

        # Create the tokenized vectors
        seqs = self.tokenizer.texts_to_sequences(captions)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        caption_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding=padding)

        self.max_length = utils.calc_max_length(seqs) # this is needed for inference

        return caption_vector

    @timer
    def _reformat_img_name_vectors(self, caption_filename_tuple, caption_vector):

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

    def save_caption_file_tuples(self, train_captions, val_captions):

        caption_cache_dir = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions')
        logger.info(f'Saving train_captions, val_captions to {caption_cache_dir}')
        
        with open(os.path.join(caption_cache_dir,f'train_captions_{CONFIG.NUMBER_OF_IMAGES}.pkl'), 'wb') as f:
            pickle.dump(train_captions, f)
        with open(os.path.join(caption_cache_dir,f'val_captions_{CONFIG.NUMBER_OF_IMAGES}.pkl'), 'wb') as f:
            pickle.dump(val_captions, f)


if __name__ == '__main__':


        path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')
        
        tokenize_captions = TokensManager()
        train_captions, val_captions = tokenize_captions.prepare_imgs_tokens(path)
        
        tokenizer_save_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'mscoco_tokenizer.pkl') 
        pickle.dump(tokenize_captions.tokenizer, open(tokenizer_save_path, 'wb'))
        #tokenize_captions.save_caption_file_tuples(train_captions, val_captions)


    



    

