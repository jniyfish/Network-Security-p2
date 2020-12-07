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

image_size=(480,480)

train_data_path='./data/'

def load_data():
    files=os.listdir(train_data_path)
    images=[]
    labels=[]
    for f in files:
        file_path=train_data_path+f+'/'
        data=os.listdir(file_path)
        for fp in data:
            img=image.load_img(file_path+fp,target_size=image_size, color_mode='grayscale')
            img_array=image.img_to_array(img)
            images.append(img_array)
            b=np.array(images)       
            if 'A1' in f:
                labels.append(0)
            elif 'A2' in f:
                labels.append(1)
            elif 'A5' in f:
                labels.append(2)
        a=np.array(labels)
    return b,a

def pred_data():
    with open('./noise.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('./noise.h5')
    path='./data/A2/'
    i=0
    j=0
    m=0
    l=0
    for f in os.listdir(path):
        img = image.load_img(path + f, target_size=image_size,color_mode='grayscale')
        img_array = image.img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        x /= 255
        j=j+1
        result = model.predict_classes(x,verbose=0)
        print(f,result)
        print('---------')

def main():
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(480,480,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.summary()

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    images,labels =load_data()
    labels = np_utils.to_categorical(labels, 3)
    images /= 255
    history=model.fit(images, labels, batch_size=32, epochs=5, verbose=2,validation_split=0.1,callbacks=[ModelCheckpoint('my_model_{val_loss:.8f}.h5',monitor='val_loss',save_best_only=True,mode='min')])
    
    yaml_string = model.to_yaml()
    with open('./noise.yaml', 'w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('./noise.h5')


if __name__ == '__main__':
    main()
    pred_data()