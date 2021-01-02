class CONFIG:
        # Paths
        ANNOTATION_FILE = '/mnt/pythonfiles/training-sets/raw/MSCOCO/annotations/captions_train2014.json'
        IMAGES_DIR = '/mnt/pythonfiles/training-sets/raw/MSCOCO/train2014'
        CACHE_DIR_ROOT = '/mnt/pythonfiles/training-sets/interim'
        CHECKPOINT_PATH = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau'
        LOGS_DIR = '/mnt/pythonfiles/users/animit/InstagramCaptioner/.logs'

        # Prepare image features parameters
        NUMBER_OF_IMAGES = 15000
        CNN_BACKBONE = 'inception_v3' #mobilenet_v2'
        INCLUDE_CNN_IN_TRAINING = False

        # Training parameters
        VOCABULARY_TOP_K = 5000
        BATCH_SIZE = 512
        BUFFER_SIZE = 1000
        EMBEDDING_SIZE = 256
        UNITS = 512
        VOCAB_SIZE = VOCABULARY_TOP_K + 1
        EPOCHS = 50
        LEARNING_RATE = 0.0005

        # Evaluation parameters
        EVAL_BATCH_SIZE = 16

