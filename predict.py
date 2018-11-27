# -*- coding: utf-8 -*-
# 画像検証の処理
# 起動: python predit.py filename
# (ex: python predit.py cat.jpg )

import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from img_model import Img_Model


#main
arg =sys.argv[1]
filename=arg
print(filename)
#quit()
#
batch_size = 32

#
#img-load
img_height, img_width = 128, 128
#img = image.load_img('data_in/train/cat/neko_13.jpg')  # this is a PIL image
# 画像を読み込んで4次元テンソルへ変換
#filename="data_in/train/cat/neko_13.jpg"
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
x = x / 255.0
#print(x)
#print(x.shape)



#
#ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical')
label_dict = train_generator.class_indices
#print( len(train_generator) )
#print(type(label_dict) )
print( label_dict )
num_categ =len(label_dict)
#
model_cls= Img_Model(num_categ)
model= model_cls.get_model()
# 学習済みの重みをロード
model.load_weights( 'img_test.h5')

model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

# クラスを予測, 入力は1枚の画像なので[0]のみ
pred  = model.predict(x)[0]
max_num =pred.max()
print( "predict:") 
print( pred ) 
print( "max=",max_num ) 
#print(type(pred) ) 

ct=0
idx_num=0
for item in pred:
    if(item==max_num ):
        idx_num=ct
#        print(item)
    ct +=1
print( "dict-value:") 
print(idx_num)

#　クラス出力
keys = [k for k, v in label_dict.items() if v ==idx_num ]
print("class=" + keys[0])
