import os
import pickle


file = '/Users/animitkulkarni/Python/training-sets/interim/mobilenet_v2_captions/caption_filename_tuple.pkl'
caption_filename_tuple = pickle.load(open(file, 'rb'))

print(caption_filename_tuple)

    