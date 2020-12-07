import os
import numpy as np
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from keras import optimizers
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint
from skimage import  io, util
import json
from PIL import Image


image_size=(480,480)

def  gaussian():
    print("Adding Gaussian noise")
    noise_path='./Test/T1/'
    img=io.imread('./my1.png')
    noise_img_gaussian=util.random_noise(img,mode='gaussian')
    io.imsave(noise_path+"gau_1"+".png",noise_img_gaussian)
def preprocessing():

    input_file = open('./Example_Test/Test_1/packetbeat.json')
    i = 0
    q = 0
    k = 0
    w,h = 500 , 500
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for in_data in input_file:
        if i >= 250000:
            break
        j = json.loads(in_data)
        r = j['destination']['port']%128
        g = j['destination']['port']/128%128
        b = j['destination']['port']/128/128%128
        data[q,k] = [r,g,b]
        k = k+1;
        if k == 500:
            q = q+1
            k = 0

        i =i + 1
    img = Image.fromarray(data, 'RGB')
    img.save('./my1.png')
    img.show()

def load_data():
    train_data_path='./Test/'
    files=os.listdir(train_data_path)
    images=[]
    labels=[]
    with open('./noise.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./noise.h5')
    for f in files:
        file_path=train_data_path+f+'/'
        data=os.listdir(file_path)
        for fp in data:
            img=image.load_img(file_path+fp,target_size=image_size, color_mode='grayscale')
            img_array=image.img_to_array(img)
            x = np.expand_dims(img_array, axis=0)
            x /= 255
            result = model.predict_classes(x,verbose=0)
            print(f,result)

preprocessing()
gaussian()
load_data()