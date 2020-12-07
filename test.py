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
import sys
import re


argv = './'
image_size=(480,480)

def gaussian(testcase):
    print("Adding Gaussian noise")
    noise_path='./Test/'
    img=io.imread('./my86.png')
    noise_img_gaussian=util.random_noise(img,mode='gaussian')
    io.imsave( noise_path + testcase+".png",noise_img_gaussian)
def preprocessing():
    files = os.listdir(sys.argv[1])
    print(files)
    fuck = 1 
    for f in files:
        dirpath = sys.argv[1] +"/" +f
        dir = os.listdir(dirpath)
        for jsonfile in dir:
            if 'packetbeat.json' in jsonfile:
                path = sys.argv[1] + "/"+ f + "/" + jsonfile
                print(path)
                input_file = open(path)
                i = 0
                q = 0
                k = 0
                w,h = 500 , 500
                data = np.zeros((h, w, 3), dtype=np.uint8)
                for in_data in input_file:
                    if i >= 250000:
                        break
                    j = json.loads(in_data)
                    try:
                        r = j['destination']['port']%255
                        g = j['destination']['port']/255%255
                        b = j['destination']['port']/128/128%128
                        data[q,k] = [r,g,b]
                        k = k+1;
                        if k == 500:
                            q = q+1
                            k = 0
                        i =i + 1
                    except KeyError:
                        continue
                img = Image.fromarray(data, 'RGB')
                img.save('./my86.png')
                #img.show()
                gaussian(f)

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
        img=image.load_img(train_data_path+f,target_size=image_size, color_mode='grayscale')
        img_array=image.img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        x /= 255
        result = model.predict_classes(x,verbose=0)
        t_list  = re.findall(r'\d+', f)
        if result[0]+1 == 3:
            print("testcase " +t_list[0] +": attack " ,5)
        else:
            print("testcase " +t_list[0] +": attack " ,result[0]+1)
#print(sys.argv[1])
preprocessing()
load_data()