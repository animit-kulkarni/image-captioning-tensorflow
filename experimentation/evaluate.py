from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf

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




