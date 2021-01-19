from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
import numpy as np

from metrics_evaluation import compute_scores
from loss import loss_function

from tools.timer import timer

from config import CONFIG

CONFIG = CONFIG()


def bleu_score(predicted, actual, verbose=True):

    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    if verbose:
        print('BLEU-1: %f' % b1)
        print('BLEU-2: %f' % b2)
        print('BLEU-3: %f' % b3)
        print('BLEU-4: %f' % b4)

    return [b1, b2, b3, b4]

class EvaluationHandler:
    def __init__(self, tokenizer, loss_object):

        print('Setting up Evaluation Handler')

        self.tokenizer = tokenizer
        self.loss_object = loss_object
        self.special_tokens = ['<unk>', '<pad>', '<end>', '<start>']       


    @timer
    def evaluate_data(self,  encoder, decoder, validation_dataset, val_steps):

        self.encoder = encoder
        self.decoder = decoder

        print('Begin evaluation')
        avg_scores = np.array([0, 0, 0, 0], dtype=float)
        for batch_idx, (img_tensor, target) in enumerate(validation_dataset):
            score = self._evaluate_batch(batch_idx, img_tensor, target, self.tokenizer, self.loss_object)
            avg_scores += score/float(val_steps)

        #cleaned_scores = self._clean_coco_scores_output(score)
        avg_scores = {'BLEU':avg_scores}
        print('The average BLEU:', avg_scores)
        return avg_scores


    def _evaluate_batch(self, batch_idx, img_tensor, target, tokenizer, loss_object):
        
        self.loss, self.total_loss, predicted_ids = self._forward_pass(img_tensor, target, tokenizer, self.loss_object)

        predicted_ids = np.array(predicted_ids).reshape(CONFIG.EVAL_BATCH_SIZE, 47)

        print('finished forward pass')

        cleaned_target = self._tokens_to_captions(target, self.special_tokens)
        cleaned_predicted_tokens = self._tokens_to_captions(predicted_ids, self.special_tokens)
    
        ground_truth_captions = {f'{k}':v for (k, v) in enumerate(cleaned_target)}
        predicted_captions = {f'{k}':v for (k, v) in enumerate(cleaned_predicted_tokens)}

        print('computing scores')

        # score, scores = compute_scores(ground_truth_captions, predicted_captions)

        # cleaned_scores = self._clean_coco_scores_output(score)
        # print(cleaned_scores)
        
        for gt, pred in zip(ground_truth_captions.values(), predicted_captions.values()):
            score = self.bleu_score(gt, pred, verbose=False)

            
        return score
    
    #@tf.function
    def _forward_pass(self, img_tensor, target, tokenizer, loss_object):
        """Training step as tf.function to allow for gradient updates in tensorflow.

        Args:
            img_tensor -- this is output of CNN
            target -- caption vectors of dim (units, max_length) where units is num GRUs and max_length is size of caption with most tokens
        """
        loss = 0

        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        features = self.encoder(img_tensor)
        result_ids = []
        for i in range(0, target.shape[1]):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            predicted_ids = tf.math.argmax(predictions, axis=1)
            result_ids.append(predicted_ids) 
            loss += loss_function(target[:, i], predictions, self.loss_object)
            dec_input = tf.expand_dims(target[:, i], 1) # take the ith word in target not pred i.e. teacher forcing method

        total_loss = (loss / int(target.shape[1]))

        return loss, total_loss, result_ids

    def _tokens_to_captions(self, tokens_batch, tokens_to_remove):

        predicted_captions = self.tokenizer.sequences_to_texts(np.array(tokens_batch))

        cleaned_captions_batch =[]
        for caption in predicted_captions:
            # 47 is max seqence length remember ... (6000 dataset)
            clean_caption = caption.split(' ')[:47]
            clean_caption = [item for i, item in enumerate(clean_caption) if '<end>' in clean_caption[i:]]
            clean_caption = [item for item in clean_caption if item not in tokens_to_remove]
            if clean_caption == []:
                clean_caption = [' ']
            clean_caption_str = ' '.join(clean_caption)
            print(clean_caption_str)
            cleaned_captions_batch.append([clean_caption_str])            

        return cleaned_captions_batch


    def _clean_coco_scores_output(self, scores_dict):

        score_names = ['BLEU', 'METEOR', 'ROUGE']

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
            


