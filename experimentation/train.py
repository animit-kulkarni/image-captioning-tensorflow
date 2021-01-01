import tensorflow as tf
import numpy as np
import pickle
import os
import time
from datetime import datetime
import logging
import cv2

from config import CONFIG
from tools.logging_helper import LOGGING_CONFIG
from tools.timer import timer
import utils
from model import CNN_Encoder, RNN_Decoder
from loss import loss_function
from prepare_img_features import model_config_dict
from tokenize_captions import TokensManager

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

        # Iterate over the sequence (each caption was tokenized and padded up to 49 tokens. These 'numbers' are embedded as  )        
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


if __name__ == '__main__':

    #TODO: I think a TrainingManager class could be made where all this model_id, path creation etc is made and reduces the 
    # clunkiness of the training code.
    if CONFIG.INCLUDE_CNN_IN_TRAINING:
        model_id = datetime.now().strftime("%d%m%Y-%H%M%S") + '_CNN'
    else:
        model_id = datetime.now().strftime("%d%m%Y-%H%M%S")

    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')
        
    logger.info('Preparing tokens')
    tokens_manager = TokensManager()
    train_captions, val_captions = tokens_manager.prepare_imgs_tokens(caption_filename_tuple_path)
    tokens_manager.save_caption_file_tuples(train_captions, val_captions) # this isn't necessary for training but useful for analytical work
    tokenizer_save_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'coco_tokenizer.pkl') 
    pickle.dump(tokens_manager, open(tokenizer_save_path, 'wb')) # save the tokenizer for inference

    # separate the filenames and captions to get correct format for dataset work
    img_name_train = [i[0] for i in train_captions]
    cap_train = [i[1] for i in train_captions]
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    if CONFIG.INCLUDE_CNN_IN_TRAINING:
        loading_data_fn = utils.map_func_including_cnn
    else:
        loading_data_fn = utils.map_func

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(loading_data_fn,
                                                                 [item1, item2],
                                                                 [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
    num_steps = len(img_name_train)//CONFIG.BATCH_SIZE
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, tokens_manager.tokenizer, loss_object)
            step += 1
            total_loss += t_loss

            if batch % 5 == 0:
                logger.info(f'Epoch: {epoch+1}/{CONFIG.EPOCHS} | Batch {batch+1}/{num_steps} | Loss: {batch_loss.numpy() / int(target.shape[1])}')
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', batch_loss, step=step)
                tf.summary.scalar('learning_rate', CONFIG.LEARNING_RATE, step=step)

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







