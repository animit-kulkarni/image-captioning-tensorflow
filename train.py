import tensorflow as tf
import numpy as np
import pickle
import os
import time

from config import CONFIG
import utils
from model import CNN_Encoder, RNN_Decoder
from loss import loss_function
from prepare_img_features import model_config_dict
from tokenize_captions import TokensManager

@tf.function
def train_step(img_tensor, target, tokenizer, loss_object):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions, loss_object)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


if __name__ == '__main__':

    caption_filename_tuple_path = os.path.join(CONFIG.CACHE_DIR_ROOT, 'mobilenet_v2_captions', 'caption_filename_tuple.pkl')
        
    tokens_manager = TokensManager()
    train_captions, val_captions = tokens_manager.prepare_imgs_tokens(caption_filename_tuple_path)
    # tokens_manager.save_caption_file_tuples(train_captions, val_captions)

    img_name_train = [i[0] for i in train_captions]
    cap_train = [i[1] for i in train_captions]

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        utils.map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Model
    encoder = CNN_Encoder(CONFIG.EMBEDDING_SIZE)
    decoder = RNN_Decoder(CONFIG.EMBEDDING_SIZE, CONFIG.UNITS, CONFIG.VOCAB_SIZE)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Admin
    checkpoint_path = os.path.join(CONFIG.CHECKPOINT_PATH, 'checkpoints/train')
    if not os.path.exists(checkpoint_path):
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

    # Train!
    loss_plot = []
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, tokens_manager.tokenizer, loss_object)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
