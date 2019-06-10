# Gesture_Recognition

Given a collection of hand gesture training images, we aim to design a classification model that distinguishes different gestures from new images. This problem has widespread application in areas such as ASL translation and driver/pedestrian hand recognition for safe and smart driving. As of recently, deep learning has seen tremendous growth in different computer vision and machine learning applications, so we attempt the problem of hand gesture recognition by designing and training a three-layered Convolutional Neural Network. The network takes in a set of images from 10 different gestures and is able to classify them with around 90.2% accuracy.

# Requirements

Install package 'keras ' as follows : $ pip install --user keras

demo notebook along with its required files tested on DSMLP server with 'launch-tf-gpu.sh'


# Files
=================
<pre>
demo_HAM10000_VGG13.ipynb -- Run a demo of our code 
        |
        |----- imports VGG-13 model trained from the training notebook below 
        |----- evalutes a example testset of 10 images picked from the complete dataset 
        
HAM10000_VGG13_training.ipynb -- Run the training of our VGG-13 model on complete HAM10000 dataset
        |
        |----- dataset preprocessing (reshaping, nomalization, one-hot labels)
        |----- VGG-13 model training (fitting) and plotting of training acc/loss history
        |----- LeNet-5 model training (fitting) and plotting of training acc/loss history as comparison
        
utility.py -- Implements some helper functions for training and displaying
        |
        |----- train-test separation on data and labels 
        |----- image normalization to (0,1)
        |----- convert to one-hot labels
        |----- plotting tools such as training accuracy line plots and example image displays

assets/vgg13_model.json -- Our VGG-13 network architecture definition
      /vgg13_model.h5   -- Trained parameters of VGG-13 network on HAM10000 datasets
      /images_test.npy  -- Zipped numpy binary file of arrays of test images
      /labels_test.npy  -- Zipped numpy binary file of arrays of test labels
</pre>

# DEMO
The Gesture_Recognition_DEMO.ipynb notebook takes a test set 1,000 images 
