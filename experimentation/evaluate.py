from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf

def eval_step(img_tensor, target, tokenizer, eval_batch_size):

    batch_hidden = decoder.reset_state(batch_size=eval_batch_size)
    features = encoder(img_tensor)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    max_length = target.shape[1]
    convert_tok = lambda x: tokenizer.index_word[(x)]


    predicted_words = [['<start>'] for i in range(eval_batch_size)]
    for _ in range(max_length):
        # we could use the code below instead to generate randomness in sentence creation - useful for production
        # but not the testing here: tf.random.categorical(predictions, 1, seed=42)[0][0].numpy()
        batch_predictions, batch_hidden, _ = decoder(dec_input, features, batch_hidden)
        batch_predictions = np.matrix(batch_predictions)
        predicted_ids = batch_predictions.argmax(axis=1)
        predicted_ids = np.array([int(id_) for id_ in predicted_ids])
        words = [tokenizer.index_word[id_] for id_ in predicted_ids]

        for i, word in enumerate(words):
            if predicted_words[i][-1] != '<end>':
                predicted_words[i].append(word)

        dec_input = tf.expand_dims(predicted_ids, 1)
    
    target_words = []
    for caption in target:
        target_words.append([tokenizer.index_word[int(token)] for token in caption])

    total_bleu = 0
    for target, predicted in zip(target_words, predicted_words):
        bleu_score = bleu_score(predicted, target)
        total_bleu += bleu_score
    
    batch_bleu_score = total_bleu / eval_batch_size

    return batch_bleu_score

def bleu_score(predicted_words, target_words):
    return sentence_bleu(target_words, predicted_words)


