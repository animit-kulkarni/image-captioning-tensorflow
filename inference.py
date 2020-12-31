import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
import sys

from config import CONFIG
from model import CNN_Encoder, RNN_Decoder
import utils
from prepare_img_features import load_image

class InstgramCaptioner:

    def __init__(self, checkpoint_path, tokenizer_path, CONFIG):
        """Load weights of encoder-decoder model from checkpoint. Load saved tokenizer.

        Args:
            checkpoint_path (str): path to directory containing checkpoints
            tokenizer_path (str): path to pickle file storing tokenizer
            CONFIG (CONFIG object): an object storing the configuration for package
        """

        self.mobilenet_v2 = tf.keras.applications.MobileNetV2
        self.cnn_feature_model = self._reconfigure_cnn()

        self.encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE)
        self.decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)

        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint)

        if ckpt_manager.latest_checkpoint:
            print("******** Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("******** Initializing from scratch.")

        self.tokens_manager = pickle.load(open(tokenizer_path, 'rb'))

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
        
        attention_plot = np.zeros((max_length, 49))
        # hidden.shape = [1, 512]
        # features,shape = [1, 49, 256]
        # decoder_input.shape = [1, 1]
        hidden = self.decoder.reset_state(batch_size=1)
        features = self._create_img_encoding(image_path)
        decoder_input = tf.expand_dims([self.tokens_manager.tokenizer.word_index['<start>']], 0)

        result = []
        for _ in range(max_length):
            # we could use the code below instead to generate randomness in sentence creation - useful for production
            # but not the testing here: tf.random.categorical(predictions, 1, seed=42)[0][0].numpy()
            predictions, hidden, attention_weights = self.decoder(decoder_input, features, hidden)
            predicted_id = np.argmax(predictions) 
            result.append(self.tokens_manager.tokenizer.index_word[predicted_id])

            if self.tokens_manager.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            decoder_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    def _create_img_encoding(self, image_path):
        """Encode the image using the CNN (e.g. MobileNetV2) and pass through a fully connected layer to embed the image's features.

        Args:
            image_path (str): path to the serialized img - png/jpg/jpeg

        Returns:
            features: a tensorflow Tensor object of dim [batch_size, cnn_feature_shape, embedding_dim] (e.g. [1, 49, 256])
        """
        
        img = self._load_image(image_path)
        temp_input = tf.expand_dims(img, 0) # this is like saying batch_size = 1
        cnn_output = self.cnn_feature_model(temp_input)
        cnn_output = tf.reshape(cnn_output, (cnn_output.shape[0], -1, cnn_output.shape[3]))
        features = self.encoder(cnn_output)

        return features

    def _reconfigure_cnn(self):
        """Reconfigures the MobileNetV2 architectire, removing the final layer (and ImageNet classification layer).

        Returns:
            tf.keras.Model: the reconfigured MobileNetV2 architecture.
        """

        model = self.mobilenet_v2(include_top=False, weights='imagenet')
        new_input = model.input
        remaining_desired_architecture = model.layers[-1].output
        reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)
        return reconfigured_cnn


    def _load_image(self, image_path):
        """load_image function following the convention of keras preprocessing operations for consistency with training code.

        Args:
            image_path (str): path to serialized img - png/jpg/jpeg

        Returns:
            img: Tensor of image resized to (244, 244)
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img

    def test_img_from_mscoco(self, idx, caption_filename_tuple_path, output_file='current_img.png'):
        """Test the model on an image from the downloaded dataset. This requires the caption_filename_tuple to have
            been generated and pickled using utils.organise_data(). 

            Example:

                train_captions, img_name_vector = utils.organise_data()
                caption_filename_tuple = list(zip(train_captions, img_name_vector))

                with open(os.path.join(captions_dir,'caption_filename_tuple.pkl'), 'wb') as pickle_file:
                    pickle.dump(caption_filename_tuple, pickle_file)

                caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG)
                caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')

                idx = 0 # take first img in dataset
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
        result, _ = self.generate_caption(current_img_path)
        gen_caption = ' '.join(result[:-1])

        print(f'The caption PREDICTED by caption_bot'.center(80, '*'))
        print(gen_caption)
        print(f'The original ground truth caption'.center(80, '*'))
        print(ground_truth_caption)

        # cv2 operations to annotate the image with predicted and ground-truth captions
        current_img = cv2.imread(current_img_path)
        cv2.rectangle(current_img, (15, 25), (current_img.shape[1] - 15, 85), (95, 95, 95), cv2.FILLED)
        cv2.putText(current_img, gen_caption, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (60, 30, 255), 1, cv2.LINE_AA)
        cv2.putText(current_img, ground_truth_caption, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (20, 240, 10), 1, cv2.LINE_AA)
        cv2.imwrite(output_file, current_img)

if __name__ == '__main__':

    tokenizer_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'coco_tokenizer.pkl') 
    checkpoint_path = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/checkpoints/train/31122020-180918'

    caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG)
    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')

    idx = int(sys.argv[1])
    caption_bot.test_img_from_mscoco(idx, caption_filename_tuple_path)







