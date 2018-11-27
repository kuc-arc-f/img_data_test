# -*- coding: utf-8 -*-
# モデルの定義

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

class Img_Model:
        mNum_categ=0;
        def __init__(self, num_categ):
                param=0
                self.mNum_categ =num_categ;

        def get_model(self ):
                model = Sequential()
                model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(64, (3, 3))  )  
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                model.add(Dense(64))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
#                model.add(Dense(2))
                model.add(Dense( self.mNum_categ ))
                model.add(Activation('softmax'))
                #model.summary()
                #
#                model.compile(loss='categorical_crossentropy',
#                        optimizer='adam',
#                        metrics=['accuracy'])
                return model



#        def test(self):
#                print("test")
