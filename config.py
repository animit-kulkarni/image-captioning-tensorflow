class CONFIG:
        # Paths
        ANNOTATION_FILE = '/Users/animitkulkarni/Python/InstagramCaptioner/data_management/annotations/captions_train2014.json'
        IMAGES_DIR = '/Users/animitkulkarni/Python/training-sets/raw/train2014'
        CACHE_DIR_ROOT = '/Users/animitkulkarni/Python/training-sets/interim/'
        CHECKPOINT_PATH = "/Users/animitkulkarni/Python/models/mobilnet_v2_bahdanau"


        # Training parameters
        VOCABULARY_TOP_K = 5000
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000
        EMBEDDING_SIZE = 256
        UNITS = 512
        VOCAB_SIZE = VOCABULARY_TOP_K + 1
        EPOCHS = 1
