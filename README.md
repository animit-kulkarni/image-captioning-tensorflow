# Image Captioner

Ever struggled to find a caption for your latest instagram pic? Worry no more!

## Table of Contents
* [Summary](#summary)
* [Technologies](#technologies)
* [Setup and Usage](#setup-and-usage)

## Summary
Whilst object detection is a mature and heavily researched topic within the computer vision and deep learning spaces, identifying the names of objects alone is insubstantial in many applications. In a world where everyday consumers are interacting with sophistcated technologies, it is imperative AI can do more to interact with humans than just single words and instead need the ability to form accurate and grammatically sound descriptions of scenes in an image. The applications are endless, ranging from supporting child education to enabling health assistants for the elderly or visually impaired.  

With this piece of work, an attempt is made to solve a frequent (yet somewhat less empathetic to society compared to the applications mentioned above...) problem in social media - deciding on a caption for your latest instagram post. 

Here a CNN architecture (MobileNetV2 and InceptionV3, but tehcnically any CNN backbone supported in keras.applications) is used to encode the image as input features to a GRU sequence model utilising Bahdanau attention. The model was trained under teacher forcing.

More information on the model and experiments can be found here [placeholder].  

The model has been trained on the MSCOCO 2014 dataset. Whilst 2 x Nvidia GeForce RTX 2080 Ti GPUs were used in training, this implementation can be trained on CPU subject to library requirements.

## Technologies
* python 3.6.8
* cuda 10.1
* For information on packages see: [requirements.txt](requirements.txt)

Note on dependencies:
* When using cuda 10.1, tensorflow 2.3.1 didn't pick up GPU. Instead tensorflow 2.1. and 2.2 worked with GPU

Training and Inference:
* wandb integration
* tensorboard integration
* gradio - accommodates frontend for inference

## Setup and Usage

~~~
pip install virtualenv

virtualenv -p [path-to-python] venv

source venv/bin/activate
~~~

1. Clone the repository

~~~
git clone https://github.com/animit-kulkarni/InstagramCaptioner.git

pip install -r requirements.txt
~~~
2. Configure [config](experimentation/config.py) file, ensuring paths for inputs and output are valid

3. Download data using tensorflow api

~~~
python data_management/download_data.py
~~~

2. Prepare image features by performing a forward pass through the CNN architecture

~~~
python prepare_img_features.py
~~~

3. Train model. Training parameters are found in [config.py](config.py). Tensorboard can be launched to track training at `http://localhost:6006/`

~~~
python train.py
~~~

4. Inference

~~~
python inference.py [int]
~~~

