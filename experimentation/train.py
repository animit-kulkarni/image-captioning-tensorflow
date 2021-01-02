import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pickle
import os
import time
from datetime import datetime
import logging
import cv2
from tqdm import tqdm

from config import CONFIG
from tools.logging_helper import LOGGING_CONFIG
from tools.timer import timer
import utils
from model import CNN_Encoder, RNN_Decoder
from loss import loss_function
from prepare_img_features import model_config_dict
from tokenize_captions import TokensManager
import evaluate

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

logging.basicConfig(**LOGGING_CONFIG.print_kwargs)
logger = logging.getLogger(__name__)
logger.info('Logging has begun!')


@tf.function
def train_step(img_tensor, target, tokenizer, loss_object):
    """Training step as tf.function to allow for gradient updates in tensorflow.

    Args:
        img_tensor -- this is output of CNN
        target -- caption vectors of dim (units, max_length) where units is num GRUs and max_length is size of caption with most tokens
    """
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
         # img_tensor.shape = [512, 49, 1280] (batch_size, feature_map_size)
         # features.shape = [512, 49, 256] (batch_size, fc_layer_output)
        features = encoder(img_tensor)

        # Iterate over the sequence (each caption was tokenized and padded up to 47 tokens (max_length). These 'numbers' are embedded as tokens.    
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            # predictions.shape = [512, 5001] (batch_size, vocab_size)
            # hidden.shape = [512, 512] (batch_size, units)
            # dec_input.shape = [512, 1] (batch_size, 1)
            # features.shape = [512, 49, 256] (batch_size, embedded_cnn_output_shape)

            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions, loss_object)
            
            # using teacher forcing. See 'https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/' 
            # for more information - it's a simple concept
            # expand dims is like unsqueeze in pytorch.

            dec_input = tf.expand_dims(target[:, i], 1) # take the ith word in target not pred i.e. teacher forcing method

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


#@timer
def eval_step(img_tensor, target, tokenizer, eval_batch_size):

    batch_hidden = decoder.reset_state(batch_size=eval_batch_size)
    features = encoder(img_tensor)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    max_length = target.shape[1]

    predicted_words = [['<start>'] for i in range(eval_batch_size)]
    for _ in range(max_length):
        # we could use the code below instead to generate randomness in sentence creation - useful for production
        # but not the testing here: tf.random.categorical(predictions, 1, seed=42)[0][0].numpy()
        batch_predictions, batch_hidden, _ = decoder(dec_input, features, batch_hidden)
        predicted_ids = K.argmax(batch_predictions, axis=1)
        words = [tokenizer.index_word[int(id_)] for id_ in predicted_ids]

        for i, word in enumerate(words):
            if predicted_words[i][-1] != '<end>':
                predicted_words[i].append(word)
            else: 
                predicted_words[i].append('<pad>')

        dec_input = tf.expand_dims(predicted_ids, 1)
    
    target_words = []
    for caption in target:
        target_words.append([tokenizer.index_word[int(token)] for token in caption])

    total_bleu = np.array([0, 0, 0, 0], dtype = 'float64')
    for target, predicted in zip(target_words, predicted_words):
        bleu_score = evaluate.bleu_score(predicted[1:], target, verbose=False)
        total_bleu += bleu_score
    
    batch_bleu_score = total_bleu / eval_batch_size

    return batch_bleu_score

def run_eval():
    print(f'EVALUATING {eval_steps} steps')
    total_bleu_score = 0
    for (v_batch, (v_img_tensor, v_target)) in enumerate(val_dataset):
        batch_bleu_score = eval_step(v_img_tensor, v_target, tokens_manager.tokenizer, CONFIG.EVAL_BATCH_SIZE)
        logger.info(f"Eval step {v_batch}/{eval_steps} || BLEU-scores: {batch_bleu_score}")
        total_bleu_score += batch_bleu_score
        average_bleu_score = total_bleu_score/(eval_steps)

    with train_summary_writer.as_default():
        tf.summary.scalar('AVG-BLEU-1', average_bleu_score[0], step=step)
        tf.summary.scalar('AVG-BLEU-2', average_bleu_score[1], step=step)
        tf.summary.scalar('AVG-BLEU-3', average_bleu_score[2], step=step)
        tf.summary.scalar('AVG-BLEU-4', average_bleu_score[3], step=step)


if __name__ == '__main__':

    #TODO: I think a TrainingManager class could be made where all this model_id, path creation etc is made and reduces the 
    # clunkiness of the training code.
    if CONFIG.INCLUDE_CNN_IN_TRAINING:
        model_id = datetime.now().strftime("%d%m%Y-%H%M%S") + '_CNN'
    else:
        model_id = datetime.now().strftime("%d%m%Y-%H%M%S")

    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', f'caption_filename_tuple_{CONFIG.NUMBER_OF_IMAGES}.pkl')
        
    logger.info('Preparing tokens')
    tokens_manager = TokensManager()
    train_captions, val_captions = tokens_manager.prepare_imgs_tokens(caption_filename_tuple_path)
    tokens_manager.save_caption_file_tuples(train_captions, val_captions) # this isn't necessary for training but useful for analytical work
    tokenizer_save_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', f'coco_tokenizer_{CONFIG.NUMBER_OF_IMAGES}.pkl') 
    pickle.dump(tokens_manager, open(tokenizer_save_path, 'wb')) # save the tokenizer for inference

    # separate the filenames and captions to get correct format for dataset work

    train_dataset = tf.data.Dataset.from_tensor_slices(([t[0] for t in train_captions], [t[1] for t in train_captions]))
    val_dataset = tf.data.Dataset.from_tensor_slices(([v[0] for v in val_captions], [v[1] for v in val_captions]))

    if CONFIG.INCLUDE_CNN_IN_TRAINING:
        loading_data_fn = utils.map_func_including_cnn
    else:
        loading_data_fn = utils.map_func

    # Use map to load the numpy files in parallel
    train_dataset = train_dataset.map(lambda file, cap: tf.numpy_function(loading_data_fn,
                                                                 [file, cap],
                                                                 [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(lambda file, cap: tf.numpy_function(loading_data_fn,
                                                                 [file, cap],
                                                                 [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    train_dataset = train_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.EVAL_BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    # Model
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE, include_cnn_backbone=CONFIG.INCLUDE_CNN_IN_TRAINING)
    decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-7)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Admin
    checkpoint_path = os.path.join(CONFIG.CHECKPOINT_PATH, 'checkpoints/train', model_id)
    if not os.path.exists(checkpoint_path):
        logger.info(f'Creating directory: {checkpoint_path}')
        os.makedirs(checkpoint_path)

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # Tensorboard
    train_log_dir = os.path.join(CONFIG.LOGS_DIR, model_id)
    if not os.path.exists(train_log_dir):
        logger.info(f'Creating directory: {train_log_dir}')
        os.makedirs(train_log_dir)
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Train!
    logger.info('       *********** BEGIN TRAINING LOOP ***********')
    loss_plot = []
    step = 0
    num_steps = len(train_captions)//CONFIG.BATCH_SIZE
    eval_steps = (len(val_captions)// CONFIG.EVAL_BATCH_SIZE)

    for epoch in range(start_epoch, CONFIG.EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(train_dataset):
            batch_loss, t_loss = train_step(img_tensor, target, tokens_manager.tokenizer, loss_object)
            step += 1
            total_loss += t_loss

            if batch % 5 == 0:
                logger.info(f'Epoch: {epoch+1}/{CONFIG.EPOCHS} | Batch {batch+1}/{num_steps} | Loss: {batch_loss.numpy() / int(target.shape[1])}')
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', batch_loss, step=step)
                tf.summary.scalar('learning_rate', CONFIG.LEARNING_RATE, step=step)

            # if batch % 1000 == 0 and batch != 0:
            #     run_eval()

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            logger.info(f'... saving checkpoint after {epoch} epochs')
            ckpt_manager.save()

        logger.info('Epoch: {} | Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    logger.info(' **** Saving final model weights ****')

    # encoder.save_weights(os.path.join(checkpoint_path, 'encoder_weights.tf'))
    # logger.info(f'      Encoder weights saved to {checkpoint_path}')
    # decoder.save_weights(os.path.join(checkpoint_path, 'decoder_weights.tf'))
    # logger.info(f'      Decoder weights saved to {checkpoint_path}')

    # encoder.save(os.path.join(checkpoint_path, 'encoder'))
    # logger.info(f'      Encoder weights saved to {checkpoint_path}')
    # decoder.save(os.path.join(checkpoint_path, 'decoder'))
    # logger.info(f'      Decoder weights saved to {checkpoint_path}')


    # tokenizer_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'coco_tokenizer.pkl') 
    # checkpoint_path = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau/checkpoints/train/29122020-113613'
    # image_path = os.path.join(CONFIG.IMAGES_DIR, os.listdir(CONFIG.IMAGES_DIR)[0])

    # current_img = cv2.imread(image_path)
    # cv2.imwrite('current_img.png', current_img)

    # caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG, encoder = encoder, decoder = decoder)

    # result, attention_plot = caption_bot.generate_caption(image_path)

    # caption_bot = InstgramCaptioner(checkpoint_path, tokenizer_path, CONFIG, encoder = encoder, decoder = decoder)
    # result2, _ = caption_bot.generate_caption(image_path)

    # print(result)
    # print(result2)







