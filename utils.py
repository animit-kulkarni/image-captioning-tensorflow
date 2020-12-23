import tensorflow as tf

def load_image(image_path, model_config):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, model_config['input_dims'])
    img = tf.keras.applications.imagenet_utils.preprocess_input(img)
    return img