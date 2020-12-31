# InstagramCaptioner

Ever struggled to find a caption for your latest instagram pic? Worry no more!

## Table of Contents
* [Summary](#summary)
* [Technologies](#technologies)
* [Setup and Usage](#setup-and-usage)

## Summary
Whilst object detection is a mature and heavily researched topic within the computer vision and deep learning spaces, identifying the names of objects alone is insubstantial in many applications. In a world where everyday consumers are interacting with sophistcated technologies, it is imperative AI can do more to interact with humans tha just single words, instead having the ability to form accurate and grammatically sound descriptions of scenes in image. The applications are endles, including child education and health assistants for elderly or visually disabled.  

With this piece of work, an attempt is made to solve a frequent (yet somehwat less empathic to society compared to the applications mentioned above...) problem in social media - deciding on a caption for your latest instagram post. 

Here a MobileNetV2 architecture is used to encode the image as features to a GRU sequence model with Bahdanau attention.

The model has been trained on the MSCOCO 2014 dataset. Whilst 2x RTX 2080 Ti GPUs were used to train the model, this implementation can be trained on CPU subject to library requirements.

## Technologies
* python 3.6.8
* cuda 10.1
* For information on packages see: [requirements.txt](requirements.txt)

Note on dependencies:
* When using cuda 10.1, tensorflow 2.3.1 didn't pick up GPU. Instead tensorflow 2.1. and 2.2 worked with GPU

## Setup and Usage
`pip install virtualenv'

`virtualenv -p [path-to-python] venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

1. Clone the repository

`git clone https://github.com/animit-kulkarni/InstagramCaptioner.git`

2. Configure congig.py paths

3. Download data

`python data_management/download_data.py`

2. Prepapre image features

`python prepare_img_features.py`

3. Train model

`python train.py`

4. Inference

`python inference.py [int]`

