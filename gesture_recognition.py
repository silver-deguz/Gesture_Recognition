import numpy as np
import tensorflow as tf
import keras
import os

from keras import layers
from keras import models


class GestureRecognition():
    def __init__(self, train_config):
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = train_config['batch_size']
        self.img_height = train_config['img_height']
        self.img_width = train_config['img_width']
        self.channel = 1
        
        self.train_images = train_config['train_images']
        self.train_labels = train_config['train_labels']
        self.test_images = train_config['test_images']
        self.test_images = train_config['test_labels']

    def train(self):
        print('Start training...')
#        epoch_loss = []

        ''' TO DO '''


#         merged = tf.summary.merge_all()
#         train_writer = tf.summary.FileWriter(os.path.join('./log','train'))

#         saver = tf.train.Saver(max_to_keep=10)
#         output_file = open(self.output_path,'w')
#         output_file.close()
        model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(self.train_images, self.train_labels, epochs=5, batch_size=64)
        
        

#    def get_loss(self):
#        ''' TO DO '''
#
#
#        return


    def write_data_to_txt(self, file_path, data):
        output_file = open(file_path, 'a')
        to_write = str(data)+'\n'
        output_file.write(to_write)
        output_file.close()
        return


    def build_model(self):
        input_shape = [self.batch_size, self.img_height, self.img_width, self.channel]
        model = models.Sequential()

        # Encoder
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None, input_shape=input_shape))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(filters=64, (3, 3), padding='same', activation=None))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(filters=64, (3, 3), padding='same', activation=None))
        model.add(layers.LeakyReLU(alpha=0.3))

        model.add(layers.Conv2D(filters=128, (3, 3), padding='same', activation=None))
        model.add(layers.LeakyReLU(alpha=0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation=None))
        model.add(layers.LeakyReLU(alpha=0.3))
        dense = tf.nn.dropout(dense, keep_prob=0.5)
        model.add(layers.Dense(10, activation='softmax'))


         # Decoder
        ''' TO DO '''
        #not sure how to do this first part
        model.add(layers.Dense(10, activation='softmax'))
        dense = tf.nn.dropout(dense, keep_prob=0.5)
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(1024, activation=None))
        #model.add(layers.Flatten()) RESHAPE??
        
        
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D_transpose(filters=128, (3, 3), padding='same', activation=None))
        
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D_transpose(filters=64, (3, 3), padding='same', activation=None))
        
        #model.add(layers.MaxPooling2D((2, 2))) RESIZING
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D_transpose(filters=64, (3, 3), padding='same', activation=None))
        
        #model.add(layers.MaxPooling2D((2, 2))) RESIZING
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D_transpose(filters=32, kernel_size=(3, 3), padding='same', activation=None, input_shape=input_shape))

        return


#    def predict(self):
#        ''' TO DO '''
#
#        return 0


    def test(self):
        ''' TO DO '''
        
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels)

        return test_loss, test_acc
