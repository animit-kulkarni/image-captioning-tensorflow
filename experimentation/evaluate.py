from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
import numpy as np
import pickle
import os
import sys

from metrics_evaluation import compute_scores
from loss import loss_function
from tools.timer import timer
from model import CNN_Encoder, RNN_Decoder
from tokenize_captions import TokensManager
import utils

from config import CONFIG
CONFIG = CONFIG()

class EvaluationHandler:
    def __init__(self, loss_object, tokenizer, checkpoint_path=None):

        print('Setting up Evaluation Handler')

        self.tokenizer = tokenizer
        self.loss_object = loss_object
        self.special_tokens = ['<unk>', '<pad>', '<end>', '<start>']       
        
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None:
            self.encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE)
            self.decoder = RNN_Decoder(
                CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)
            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                       decoder=self.decoder)

            ckpt_manager = tf.train.CheckpointManager(
                ckpt, self.checkpoint_path, max_to_keep=5)
            #chosen_checkpoint = ckpt_manager.checkpoints[2]
            chosen_checkpoint = ckpt_manager.latest_checkpoint
            ckpt.restore(chosen_checkpoint)



    @timer
    def evaluate_data(self, validation_dataset, val_steps, encoder=None, decoder=None):

        if self.checkpoint_path is None:
            assert encoder is not None
            assert decoder is not None
            self.encoder = encoder
            self.decoder = decoder

        print('Begin evaluation')
        avg_bleu = np.array([0, 0, 0, 0], dtype=float)
        avg_rouge = 0.0
        for batch_idx, (img_tensor, target) in enumerate(validation_dataset):
            score = self._evaluate_batch(img_tensor, target)
            avg_bleu += np.array(score['BLEU'], dtype=float)/float(val_steps)
            avg_rouge += score['ROUGE']/float(val_steps)

        avg_bleu = avg_bleu.round(2)
        avg_rouge = avg_rouge.round(2)
        avg_scores = {'BLEU':avg_bleu, 'ROUGE': avg_rouge}
        print('The average BLEU:', avg_scores)
        return avg_scores


    def _evaluate_batch(self, img_tensor, target):
        
        self.loss, self.total_loss, predicted_ids = self._forward_pass(img_tensor, target)
        self.loss = self.loss/(int(target.shape[1]))

        predicted_ids = np.array(predicted_ids).reshape(-1) #TODO: remove 46 hardcoding

        cleaned_target = self._tokens_to_captions(target, self.special_tokens)
        cleaned_predicted_tokens = self._tokens_to_captions(predicted_ids, self.special_tokens)
    
        ground_truth_captions = {f'{k}':v for (k, v) in enumerate(cleaned_target)}
        predicted_captions = {f'{k}':v for (k, v) in enumerate(cleaned_predicted_tokens)}

        score, scores = compute_scores(ground_truth_captions, predicted_captions)
        score = self._clean_coco_scores_output(score)
               
        # for gt, pred in zip(ground_truth_captions.values(), predicted_captions.values()):            
        #     score = self.bleu_score(gt, pred, verbose=False)
        
        return score
    
    @tf.function
    def _forward_pass(self, img_tensor, target):
        """Training step as tf.function to allow for gradient updates in tensorflow.

        Args:
            img_tensor -- this is output of CNN
            target -- caption vectors of dim (units, max_length) where units is num GRUs and max_length is size of caption with most tokens
        """
        loss = 0

        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        features = self.encoder(img_tensor)
        result_ids = []
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            predicted_ids = tf.math.argmax(predictions, axis=1)
            result_ids.append(predicted_ids) 
            loss += loss_function(target[:, i], predictions, self.loss_object)
            dec_input = tf.expand_dims(predicted_ids, 1) # take the ith word in target not pred i.e. teacher forcing method

        total_loss = (loss / int(target.shape[1]))

        return loss, total_loss, result_ids

    def _tokens_to_captions(self, tokens_batch, tokens_to_remove):

        predicted_captions = self.tokenizer.sequences_to_texts(np.array(tokens_batch))

        cleaned_captions_batch =[]
        for caption in predicted_captions:
            # 47 is max seqence length remember ... (6000 dataset)
            clean_caption = caption.split(' ')[:47]
            if '<end>' in clean_caption:
                clean_caption = [item for i, item in enumerate(clean_caption) if '<end>' in clean_caption[i:]]
            clean_caption = [item for item in clean_caption if item not in tokens_to_remove]
            if clean_caption == []:
                clean_caption = [' ']
            clean_caption_str = ' '.join(clean_caption)
            cleaned_captions_batch.append([clean_caption_str])            

        return cleaned_captions_batch


    def _clean_coco_scores_output(self, scores_dict):

        score_names = ['BLEU', 'ROUGE']

        cleaned_scores_dict = {}
        for i, (key, val) in enumerate(scores_dict.items()):
            cleaned_scores_dict[score_names[i]] = val

        return cleaned_scores_dict
            
    def bleu_score(self, predicted, actual, verbose=False):

        b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        b3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
        b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

        if verbose:
            print('BLEU-1: %f' % b1)
            print('BLEU-2: %f' % b2)
            print('BLEU-3: %f' % b3)
            print('BLEU-4: %f' % b4)

        return np.array([round(b1, 5), round(b2, 5), round(b3, 5), round(b4, 5)])
            


if __name__ == '__main__':

    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', f'caption_filename_tuple_{CONFIG.NUMBER_OF_IMAGES}.pkl')
        
    tokens_manager = TokensManager()
    train_captions, val_captions = tokens_manager.prepare_imgs_tokens(caption_filename_tuple_path)

    train_dataset = tf.data.Dataset.from_tensor_slices(([t[0] for t in train_captions], [t[1] for t in train_captions]))
    val_dataset = tf.data.Dataset.from_tensor_slices(([v[0] for v in val_captions], [v[1] for v in val_captions]))

    if CONFIG.INCLUDE_CNN_IN_TRAINING:
        loading_data_fn = utils.map_func_including_cnn
    else:
        loading_data_fn = utils.map_func


    val_dataset = val_dataset.map(lambda file, cap: tf.numpy_function(loading_data_fn,
                                                                [file, cap],
                                                                [tf.float32, tf.int32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    train_dataset = train_dataset.map(lambda file, cap: tf.numpy_function(loading_data_fn,
                                            [file, cap],
                                            [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)


    train_dataset = train_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.EVAL_BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    train_steps = (len(train_captions)// CONFIG.EVAL_BATCH_SIZE)
    val_steps = (len(val_captions)// CONFIG.EVAL_BATCH_SIZE)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    evaluation_handler = EvaluationHandler(loss_object,
                                           tokens_manager.tokenizer,
                                           checkpoint_path='/mnt/pythonfiles/models/mobilenet_v2_bahdanau/25012021-212906')

    evaluation_handler.evaluate_data(val_dataset, val_steps)
