#from tensorflow import python as tf 
# if the model has been trained with python fork then inference needs python fork otherwise 
# you get the following error:
# ValueError: Tensor's shape (1536,) is not compatible with supplied shape (2, 1536)
import tensorflow as tf
from config import CONFIG
CONFIG = CONFIG()
from prepare_img_features import model_config_dict

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 49, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 49, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 49, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 49, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, include_cnn_backbone=False):
        super(CNN_Encoder, self).__init__()

        # shape after fc == (batch_size, 49, embedding_dim=256)
        self.image_dropout_layer = tf.keras.layers.Dropout(rate=CONFIG.DROPOUT['IMAGE'])
        self.image_embedding_dropout_layer = tf.keras.layers.Dropout(rate=CONFIG.DROPOUT['IMAGE_EMBEDDING'])

        self.include_cnn_backbone = include_cnn_backbone
        if self.include_cnn_backbone:
          self.cnn = model_config_dict[CONFIG.CNN_BACKBONE]['model']
          self.cnn_backbone = self._reconfigure_cnn()
        
        self.fc = tf.keras.layers.Dense(embedding_dim)


    def call(self, x):
        if self.include_cnn_backbone:
          x = self.cnn_backbone(x)
          x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))

        x = self.image_dropout_layer(x)  # dOn't think this does anything here because there's no weights to dropout ...
        x = self.fc(x)
        if CONFIG.RELU_ENCODER:
          x = tf.nn.relu(x)
        x = self.image_embedding_dropout_layer(x)
        return x

    def _reconfigure_cnn(self):
        model = self.cnn(include_top=False, weights='imagenet')
        new_input = model.input
        remaining_desired_architecture = model.layers[-1].output
        reconfigured_cnn = tf.keras.Model(new_input, remaining_desired_architecture)
        return reconfigured_cnn


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units #512
    self.word_embedding_dropout_layer = tf.keras.layers.Dropout(rate=CONFIG.DROPOUT['WORD_EMBEDDING'])

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #vocab_size = 5001, embedding_dim=256
    self.gru = tf.keras.layers.GRU(self.units, # units=512
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # TODO: Can defo apply dropout here too ...                               
    self.fc1 = tf.keras.layers.Dense(self.units) # units=512
    self.fc2 = tf.keras.layers.Dense(vocab_size) # vocab_size=5001

    self.attention = BahdanauAttention(self.units) # units=512

  def call(self, x, features, hidden):

    # x.shape = [batch_size, 1] (this is just a number i.e. input id)
    # features.shape = [batch_size, 49, 256]
    # hidden.shape = [batch_size, 512]

    # defining attention as a separate model
    # context_vector = [512, 256] (batch_size, embedding_size)
    # attention_weights = TensorShape([512, 64, 1])
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim=256)
    x = self.embedding(x)
    x = self.word_embedding_dropout_layer(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim=256 + hidden_size=512)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
