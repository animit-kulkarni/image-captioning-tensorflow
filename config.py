class CONFIG:
        # Paths
        ANNOTATION_FILE = '/mnt/pythonfiles/training-sets/raw/MSCOCO/annotations/captions_train2014.json'
        IMAGES_DIR = '/mnt/pythonfiles/training-sets/raw/MSCOCO/train2014'
        CACHE_DIR_ROOT = '/mnt/pythonfiles/training-sets/interim'
        CHECKPOINT_PATH = '/mnt/pythonfiles/models/mobilenet_v2_bahdanau'
        LOGS_DIR = '.logs/gradient_tape'

        # Training parameters
        VOCABULARY_TOP_K = 5000
        BATCH_SIZE = 512
        BUFFER_SIZE = 1000
        EMBEDDING_SIZE = 256
        UNITS = 512
        VOCAB_SIZE = VOCABULARY_TOP_K + 1
        EPOCHS = 2
        LEARNING_RATE = 0.005

