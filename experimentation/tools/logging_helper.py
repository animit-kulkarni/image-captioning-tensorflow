import logging
import sys

class LOGGING_CONFIG:
    LOG_FILENAME = 'training_logs.log'
    logging_kwargs = {'filename': LOG_FILENAME,
                      'filemode': 'w',
                      'format': '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                      'datefmt': '%H:%M:%S',
                      'level': logging.DEBUG}

    print_kwargs = {'stream' : sys.stdout, 
                    'level' : logging.DEBUG}