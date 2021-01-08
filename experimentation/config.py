class CONFIG:

        def __init__(self):
                # Paths
                self.ANNOTATION_FILE = '/mnt/pythonfiles/training-sets/raw/MSCOCO/annotations/captions_train2014.json'
                self.IMAGES_DIR = '/mnt/pythonfiles/training-sets/raw/MSCOCO/train2014'
                self.CACHE_DIR_ROOT = '/mnt/pythonfiles/training-sets/interim'
                self.CHECKPOINT_PATH = '/mnt/pythonfiles/models'
                self.LOGS_DIR = '/mnt/pythonfiles/users/animit/InstagramCaptioner/.logs'

                # Prepare image features parameters
                self.NUMBER_OF_IMAGES = 6000
                self.CNN_BACKBONE = 'mobilenet_v2'
                self.INCLUDE_CNN_IN_TRAINING = False

                # Training parameters
                self.VOCABULARY_TOP_K = 5000
                self.BATCH_SIZE = 512
                self.BUFFER_SIZE = 1000
                self.EMBEDDING_SIZE = 256
                self.UNITS = 256 # size of GRU hidden
                self.VOCAB_SIZE = self.VOCABULARY_TOP_K + 1
                self.EPOCHS = 20
                self.LEARNING_RATE = 0.001
                self.RESUME_TRAINING = {'RESUME': False,
                                        'CHECKPOINT_AND_MODEL_ID_PATH': '/mnt/pythonfiles/models/inception_v3_bahdanau/03012021-000041'}

                self.DROPOUT = {'IMAGE': 0,
                                'IMAGE_EMBEDDING': 0,
                                'WORD_EMBEDDING': 0}

                self.L2_REGULARIZATION = {'REG': False, 'WEIGHTING_CONSTANT': 1e-8}
                self.RELU_ENCODER = False

                # Evaluation parameters
                self.EVAL_BATCH_SIZE = self.BATCH_SIZE

                # Experiment tracking
                self.WANDB = True
                self.TENSORBOARD = False
                

