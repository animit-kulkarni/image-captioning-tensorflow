import tensorflow as tf
import numpy as np
import os
import pickle
import cv2

from config import CONFIG
from model import CNN_Encoder, RNN_Decoder
import utils
from prepare_img_features import load_image

class InstgramCaptioner:

    def __init__(self, checkpoint_path, tokenizer_path, CONFIG, encoder=None, decoder=None):

        self.mobilenet_v2 = tf.keras.applications.MobileNetV2
        self.cnn_feature_model = self._reconfigure_cnn()
        model_weights_root = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/checkpoints/train/30122020-230207'

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE)
            self.encoder.load_weights(os.path.join(model_weights_root, 'encoder_weights.tf'))

        if decoder is not None:
            self.decoder = decoder
        
        else:
            self.decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)
            self.decoder.load_weights(os.path.join(model_weights_root, 'decoder_weights.tf'))


        optimizer = optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE,
                                                         beta_1=0.9,
                                                         beta_2=0.999,
                                                         epsilon=1e-7,)

        # ckpt = tf.train.Checkpoint(encoder=self.encoder,
        #                            decoder=self.decoder,
        #                            optimizer=optimizer)

        # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # ckpt.restore(ckpt_manager.latest_checkpoint)

        # if ckpt_manager.latest_checkpoint:
        #     print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        # else:
        #     print("Initializing from scratch.")

        self.tokens_manager = pickle.load(open(tokenizer_path, 'rb'))

    def generate_caption(self, image_path):

        max_length = self.tokens_manager.max_length
        print('MAX LENGTH: ', max_length) 
        
        attention_plot = np.zeros((max_length, 49))
        hidden = self.decoder.reset_state(batch_size=1)

        features = self._create_img_encoding(image_path)
        decoder_input = tf.expand_dims([self.tokens_manager.tokenizer.word_index['<start>']], 0)

        result = []
        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(decoder_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokens_manager.tokenizer.index_word[predicted_id])

            if self.tokens_manager.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            decoder_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    def _create_img_encoding(self, image_path):
        
        img = self._load_image(image_path)
        temp_input = tf.expand_dims(img, 0) # this is like saying batch_size = 1
        cnn_output = self.cnn_feature_model(temp_input)
        cnn_output = tf.reshape(cnn_output, (cnn_output.shape[0], -1, cnn_output.shape[3]))
        features = self.encoder(cnn_output)

        return features


    def _reconfigure_cnn(self):

        model = self.mobilenet_v2(include_top=False, weights='imagenet')
        new_input = model.input
        remaining_desired_architecture = model.layers[-1].output
        reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)
        return reconfigured_cnn


    def _load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img

if __name__ == '__main__':

    tokenizer_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'coco_tokenizer.pkl') 
    checkpoint_path = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/checkpoints/train/29122020-113613'
    image_path = os.path.join(CONFIG.IMAGES_DIR, os.listdir(CONFIG.IMAGES_DIR)[0])

    current_img = cv2.imread(image_path)
    cv2.imwrite('current_img.png', current_img)

    caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG)

    result, attention_plot = caption_bot.generate_caption(image_path)

    print(result)





