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
import wandb

from config import CONFIG
from tools.logging_helper import LOGGING_CONFIG
from tools.timer import timer
import utils
from model import CNN_Encoder, RNN_Decoder
from loss import loss_function
from prepare_img_features import model_config_dict
from tokenize_captions import TokensManager

from evaluate import EvaluationHandler

CONFIG = CONFIG()

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

logging.basicConfig(**LOGGING_CONFIG.print_kwargs)
logger = logging.getLogger(__name__)
logger.info('Logging has begun!')

if CONFIG.WANDB:
    wandb.init(project="instagramcaptioner")
    wandb.config.update(CONFIG.__dict__)

@tf.function
def train_step(img_tensor, target, tokenizer, loss_object, validation=False):
    """Training step as tf.function to allow for gradient updates in tensorflow.

    Args:
        img_tensor -- this is output of CNN
        target -- caption vectors of dim (units, max_length) where units is num GRUs and max_length is size of caption with most tokens
    """
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:

        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions, loss_object)
            dec_input = tf.expand_dims(target[:, i], 1) # take the ith word in target not pred i.e. teacher forcing method

    total_loss = (loss / int(target.shape[1]))
    
    if not validation:
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss
    
    else:
        return loss, total_loss, predictions


if __name__ == '__main__':

    # ************ Data ************

    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', f'caption_filename_tuple_{CONFIG.NUMBER_OF_IMAGES}.pkl')
        
    logger.info('Preparing tokens')
    tokens_manager = TokensManager()
    train_captions, val_captions = tokens_manager.prepare_imgs_tokens(caption_filename_tuple_path)
    tokens_manager.save_caption_file_tuples(train_captions, val_captions) # this isn't necessary for training but useful for analytical work
    tokenizer_save_path = os.path.join(CONFIG.CACHE_DIR_ROOT, f'{CONFIG.CNN_BACKBONE}_captions', f'coco_tokenizer_{CONFIG.NUMBER_OF_IMAGES}.pkl') 
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


    
    # ************ Shuffle and batch ************

    train_dataset = train_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.EVAL_BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



    # ************ Model ************

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE, include_cnn_backbone=CONFIG.INCLUDE_CNN_IN_TRAINING)
    decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)
    


    # ************ Optimizer ************

    initial_learning_rate = CONFIG.LEARNING_RATE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                 decay_steps=200,
                                                                 decay_rate=0.96,
                                                                 staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-7)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    # ************ Evaluation ************

    evaluation_handler = EvaluationHandler(tokens_manager.tokenizer, loss_object)
    


    # ************ Checkpoints ************

    if CONFIG.RESUME_TRAINING['RESUME']:
        logger.info('RESTORING CHECKPOINT ......')
        checkpoint_path =  CONFIG.RESUME_TRAINING['CHECKPOINT_AND_MODEL_ID_PATH']
        model_id = checkpoint_path.split('/')[-1]
    else:
        #TODO: I think a TrainingManager class could be made where all this model_id, path creation etc is made and reduces the 
        # clunkiness of the training code.
        if CONFIG.INCLUDE_CNN_IN_TRAINING:
            model_id = datetime.now().strftime("%d%m%Y-%H%M%S") + '_CNN'
        else:
            model_id = datetime.now().strftime("%d%m%Y-%H%M%S")
        checkpoint_path = os.path.join(CONFIG.CHECKPOINT_PATH, CONFIG.CNN_BACKBONE + '_bahdanau', model_id)
    
    if CONFIG.WANDB:
        wandb.config.model_id = model_id

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

    # ************ Experiment logging ************
    
    if CONFIG.TENSORBOARD:
        train_log_dir = os.path.join(CONFIG.LOGS_DIR, model_id)
        if not os.path.exists(train_log_dir):
            logger.info(f'Creating directory: {train_log_dir}')
            os.makedirs(train_log_dir)
        
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    # ************ Train! ************

    logger.info('       *********** BEGIN TRAINING LOOP ***********       ')
    loss_plot = []
    step = 0
    num_steps = len(train_captions)//CONFIG.BATCH_SIZE
    val_steps = (len(val_captions)// CONFIG.EVAL_BATCH_SIZE)

    for epoch in range(start_epoch, CONFIG.EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(train_dataset):
            batch_loss, t_loss = train_step(img_tensor, target, tokens_manager.tokenizer, loss_object)
            step += 1
            total_loss += t_loss

            if batch % 5 == 0:
                logger.info(f'Epoch: {epoch+1}/{CONFIG.EPOCHS} || Batch {batch+1}/{num_steps} || Loss: {batch_loss.numpy() / int(target.shape[1])}')
            
           
            if CONFIG.TENSORBOARD:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', batch_loss, step=step)
                    tf.summary.scalar('learning_rate', CONFIG.LEARNING_RATE, step=step)

            if CONFIG.WANDB:
                wandb.log({'Batch Loss' : batch_loss.numpy()/int(target.shape[1])})


        if CONFIG.EVALUATE_DURING_TRAINING:
            if epoch % CONFIG.EVAL_STEPS == 0:
                avg_scores  = evaluation_handler.evaluate_data(encoder, decoder, val_dataset, val_steps)

            if CONFIG.WANDB:
                wandb.log({'Val Loss': evaluation_handler.loss.numpy()}) # / int(v_target.shape[1])} )
                wandb.log({'BLEU-1': avg_scores['BLEU'][0],
                           'BLEU-2': avg_scores['BLEU'][1],
                           'BLEU-3': avg_scores['BLEU'][2],
                           'BLEU-4': avg_scores['BLEU'][3], 
                           #'METEOR': score['METEOR'], 
                           #'ROUGE': score['ROUGE']
                           })

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        if CONFIG.WANDB:
                wandb.log({'Epoch Loss' : total_loss/num_steps})

        if epoch % 5 == 0:
            logger.info(f'... saving checkpoint after {epoch} epochs')
            ckpt_manager.save()

        logger.info('Epoch: {} | Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    logger.info(f' **** A gentle reminder that checkpoint path is: {checkpoint_path} ****')









