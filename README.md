# Gesture_Recognition

Given a collection of hand gesture training images, we aim to design a classification model that distinguishes different gestures from new images. This problem has widespread application in areas such as ASL translation and driver/pedestrian hand recognition for safe and smart driving. As of recently, deep learning has seen tremendous growth in different computer vision and machine learning applications, so we attempt the problem of hand gesture recognition by designing and training a three-layered Convolutional Neural Network. The network takes in a set of images from 10 different gestures and is able to classify them with around 90.2% accuracy.

# Requirements

Install package 'keras ' as follows : $ pip install --user keras

demo notebook along with its required files tested on DSMLP server with 'launch-tf-gpu.sh'

Files ran on Python 3 on virtual GPU with 16GB RAM, library version information:
- keras: v.2.1.6
- tensorflow: v. 1.12.0
- sklearn: v.0.20.3

# Files
<pre>
Gesture_Recognition.ipynb -- Run the training of our CNN model on complete Hand Gesture dataset
        |
        |----- dataset preprocessing (reshaping, nomalization, one-hot labels)
        |----- CNN model training (fitting) and plotting of training acc/loss history
        |----- model results (correct/incorrect images, confusion matrix, classificaiton report)
        
Gesture_Recognition_DEMO.ipynb -- Run a demo of our code 
        |
        |----- imports trained CNN model trained from the training notebook below 
        |----- evaluates on example testset of 1,000 images picked from the complete dataset 
        
utility.py -- Implements some helper functions for training and displaying reults
        |
        |----- data loader
        |----- train-test separation on data and labels 
        |----- converting to one-hot labels
        |----- plotting tools such as training accuracy and example image displays

data_DEMO -- folder containes data for the demo 
        |
        |----- data_DEMO.npy = numpy file of test data (1,000 images)
        |----- labels_DEMO.npy = numpy file for test labels

my_CNN.h5 -- trained network parameters/weights 

dataset.png -- image of dataset
</pre>

# DEMO
The Gesture_Recognition_DEMO.ipynb notebook takes a test set 1,000 images and uses the network parameters trained and saved as 'my_CNN.h5' and classfies them.
Make sure all files listed are in current working directory, and run demo notebook.
