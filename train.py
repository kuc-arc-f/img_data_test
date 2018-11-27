# -*- coding: utf-8 -*-
# Kerasで自前のデータから学習と予測
# https://qiita.com/agumon/items/ab2de98a3783e0a93e66


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from img_model import Img_Model

batch_size = 32
epochs = 10
#epochs = 1
#
#ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
#
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical')
label_dict = train_generator.class_indices
#print( len(train_generator) )
print( label_dict )
num_categ =len(label_dict)
#print( len(label_dict) )
#quit()
#
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(128, 128),
        batch_size=batch_size,
         class_mode='categorical')
#model
model_cls= Img_Model(num_categ)
model= model_cls.get_model()
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

#fit
history = model.fit_generator(
        train_generator,
        samples_per_epoch=800,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=200)
model.save_weights('img_test.h5')  

