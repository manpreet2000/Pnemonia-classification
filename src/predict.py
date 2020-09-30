#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class pnemonia_class:
    def __init__(self,filename):
        self.filename =filename


    def predictpnemonia(self):
        # load model
        base_model=tf.keras.applications.densenet.DenseNet121(include_top=False)
        x=base_model.output
        x=tf.keras.layers.GlobalAveragePooling2D()(x)

        x=tf.keras.layers.Dropout(0.5)(x)

        pred=tf.keras.layers.Dense(1,activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)

        model.load_weights("./weights/my_model_weights.h5")

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] > 0.9999435:
            prediction = 'Pnemonia'
            return [{ "image" : prediction}]
            
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]
            


