import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
import logging
import sys
import matplotlib.pyplot as plt
from config import CONFIG
CONFIG = CONFIG()

from model import CNN_Encoder, RNN_Decoder
import utils as utils
from prepare_img_features import  model_config_dict
from tools.timer import timer
from tools.logging_helper import LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG.print_kwargs)
logger = logging.getLogger(__name__)
logger.info('Logging has begun!')

os.environ["CUDA_VISIBLE_DEVICES"]=""

import gradio as gr
import requests

class InstgramCaptioner:

    def __init__(self, checkpoint_path, tokenizer_path, CONFIG):
        """Load weights of encoder-decoder model from checkpoint. Load saved tokenizer.

        Args:
            checkpoint_path (str): path to directory containing checkpoints
            tokenizer_path (str): path to pickle file storing tokenizer
            CONFIG (CONFIG object): an object storing the configuration for package
        """

        self.cnn_backbone = model_config_dict[CONFIG.CNN_BACKBONE]['model']
        self.cnn_feature_model = self._reconfigure_cnn()

        self.encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE)
        self.decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)

        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        #chosen_checkpoint = ckpt_manager.checkpoints[2]
        chosen_checkpoint = ckpt_manager.latest_checkpoint
        ckpt.restore(chosen_checkpoint)

        if ckpt_manager.latest_checkpoint:
            print("******** Restored from {}".format(chosen_checkpoint))
        else:
            print("******** Initializing from scratch.")

        self.tokens_manager = pickle.load(open(tokenizer_path, 'rb'))

    @timer
    def generate_caption(self, image_path):
        """Use a CNN-GRU model to predict the caption to an image.

        Args:
            image_path (str): the path to the serialized image - png/jpg/jpeg.

        Returns:
            result: a list of strings in a sequence representing predicted caption.
        """
         
        # max_length = 47 on this dataset
        max_length = self.tokens_manager.max_length
        print('MAX LENGTH: ', max_length) 
        
        attention_plot = np.zeros((max_length, model_config_dict[CONFIG.CNN_BACKBONE]['attention_features_shape']))
        # hidden.shape = [1, 512]
        # features,shape = [1, 49, 256]
        # decoder_input.shape = [1, 1]
        hidden = self.decoder.reset_state(batch_size=1)

        img = self._load_image(image_path)
        features = self._create_img_encoding(img)
        decoder_input = tf.expand_dims([self.tokens_manager.tokenizer.word_index['<start>']], 0)

        result = []
        for i in range(max_length):
            # we could use the code below instead to generate randomness in sentence creation - useful for production
            # but not the testing here: tf.random.categorical(predictions, 1, seed=42)[0][0].numpy()
            predictions, hidden, attention_weights = self.decoder(decoder_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = np.argmax(predictions) 
            result.append(self.tokens_manager.tokenizer.index_word[predicted_id])

            if self.tokens_manager.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            decoder_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    def _create_img_encoding(self, img):
        """Encode the image using the CNN (e.g. MobileNetV2) and pass through a fully connected layer to embed the image's features.

        Args:
            image_path (str): path to the serialized img - png/jpg/jpeg

        Returns:
            features: a tensorflow Tensor object of dim [batch_size, cnn_feature_shape, embedding_dim] (e.g. [1, 49, 256])
        """
        
        temp_input = tf.expand_dims(img, 0) # this is like saying batch_size = 1
        cnn_output = self.cnn_feature_model(temp_input)
        cnn_output = tf.reshape(cnn_output, (cnn_output.shape[0], -1, cnn_output.shape[3]))
        features = self.encoder(cnn_output)

        return features

    def _reconfigure_cnn(self):
        """Reconfigures the CNN architecture, removing the final layer (and ImageNet classification layer).

        Returns:
            tf.keras.Model: the reconfigured architecture (e.g. MobileNetV2).
        """

        model = self.cnn_backbone(include_top=False, weights='imagenet')
        new_input = model.input
        remaining_desired_architecture = model.layers[-1].output
        reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)
        return reconfigured_cnn


    def _load_image(self, image_path):
        """load_image function following the convention of keras preprocessing operations for consistency with training code.

        Args:
            image_path (str): path to serialized img - png/jpg/jpeg

        Returns:
            img: Tensor of image resized to e.g. (224, 224)
        """

        if isinstance(image_path, str):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
        
        elif isinstance(image_path, np.ndarray):
            img = image_path

        img = tf.image.resize(img, model_config_dict[CONFIG.CNN_BACKBONE]['input_shape'])
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img

    def _plot_attention(self, img, attention_plot, result):

        fig, ax = plt.subplots(figsize=(10, 10))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(len_result//2, len_result//2, l+1)
            ax.set_title(result[l])
            matplotlib_img = ax.imshow(img)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=matplotlib_img.get_extent())

        plt.tight_layout()
        plt.show()
        plt.savefig('attention_plot.png')

    @timer
    def test_img_from_mscoco(self, idx, caption_filename_tuple_path, output_file='current_img.png'):
        """Test the model on an image from the downloaded dataset. This requires the caption_filename_tuple to have
            been generated and pickled using utils.organise_data(). 

            Example:

                train_captions, img_name_vector = utils.organise_data()
                caption_filename_tuple = list(zip(train_captions, img_name_vector))

                with open(os.path.join(captions_dir,'caption_filename_tuple.pkl'), 'wb') as pickle_file:
                    pickle.dump(caption_filename_tuple, pickle_file)

                tokenizer_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', 'coco_tokenizer.pkl') 
                checkpoint_path = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/checkpoints/train/02012021-183517'
                #model 31122020-180918 shows the best results so far

                caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG)
                caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', 'caption_filename_tuple.pkl')

                idx = int(sys.argv[1])
                caption_bot.test_img_from_mscoco(idx, caption_filename_tuple_path)
            
        Args:
            idx (int): index the caption_filename_tuple to select an image for inference.
            caption_filename_tuple_path (str): path to the caption_filename_tuple.
            output_file (str, optional): path to output_file location. Defaults to 'current_img.png'.
        """

        caption_filename_tuple = pickle.load(open(caption_filename_tuple_path, 'rb'))
        current_img_path = caption_filename_tuple[idx][1]

        # remove <start> and <end> tokens and convert to string
        ground_truth_caption = ' '.join(caption_filename_tuple[idx][0].split(' ')[1:-1])
        
        # forward pass on the model
        result, attention_plot = self.generate_caption(current_img_path)
        gen_caption = ' '.join(result[:-1])

        logger.info(f' The caption PREDICTED by caption_bot '.center(80, '*'))
        logger.info(gen_caption)
        logger.info(f' The LABELLED ground truth caption '.center(80, '*'))
        logger.info(ground_truth_caption)

        # cv2 operations to annotate the image with predicted and ground-truth captions
        current_img = cv2.imread(current_img_path)
        cv2.rectangle(current_img, (15, 25), (current_img.shape[1] - 15, 85), (95, 95, 95), cv2.FILLED)
        cv2.putText(current_img, gen_caption, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (60, 30, 255), 1, cv2.LINE_AA)
        cv2.putText(current_img, ground_truth_caption, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (20, 240, 10), 1, cv2.LINE_AA)
        cv2.imwrite(output_file, current_img)

        self._plot_attention(current_img, attention_plot, result)


class GardioController:

    def __init__(self, caption_bot):

        self.caption_bot = caption_bot

    def __call__(self, img):
        pass





if __name__ == '__main__':


    GRADIO = False

    tokenizer_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', f'coco_tokenizer_{CONFIG.NUMBER_OF_IMAGES}.pkl') 
    checkpoint_path = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/06022021-193842'

    caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG)
    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', f'caption_filename_tuple_{CONFIG.NUMBER_OF_IMAGES}.pkl')
    
    # some of my favourites:
    # 56 - colorful bird pic
    # 451 - a group of people are flying kites on a field
    # 479 many skiers are gathered on the snow covereed mountian.
    # 723 - a teddy bear picture

    # Models known to be working well:
    # mobilenet_v2_bahdanau/25012021-212906
    # 31122020-180918

    if not GRADIO:

        idx = int(sys.argv[1])


        caption_bot.test_img_from_mscoco(idx, caption_filename_tuple_path)

        # ALL_IMG_BASE_DIR = '/mnt/pythonfiles/training-sets/raw/MSCOCO/train2014'

        # all_img_dir = os.listdir(ALL_IMG_BASE_DIR)

        # my_img = cv2.imread(os.path.join(ALL_IMG_BASE_DIR, all_img_dir[0]))
        # caption_bot.generate_caption(my_img)

    else:

        inputs = gr.inputs.Image()

        outputs = gr.outputs.Textbox('str')

        def run_demo(img):
            result, _ = caption_bot.generate_caption(img)
            return ' '.join(result)


        gr.Interface(fn=run_demo, inputs=inputs, outputs=outputs).launch()

