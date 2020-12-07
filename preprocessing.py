import json
import numpy as np
from PIL import Image
import os
from skimage import  io, util

def  gaussian():
    print("Adding Gaussian noise")
    noise_path='./data/A5/'
    img=io.imread('./my1.png')
    for i in range(100):
        noise_img_gaussian=util.random_noise(img,mode='gaussian')
        io.imsave(noise_path+"gau_"+str(i)+".png",noise_img_gaussian)

input_file = open('./Train/Attack_5/packetbeat.json')
i = 0
q = 0
k = 0
w,h = 500 , 500
data = np.zeros((h, w, 3), dtype=np.uint8)
for in_data in input_file:
    print(i)
    if i >= 250000:
        break
    j = json.loads(in_data)
    if j['destination']['ip'] == '10.0.2.2' or j['destination']['ip'] == '10.0.2.15' : 
        r = j['destination']['port']%128
        g = j['destination']['port']/128%128
        b = j['destination']['port']/128/128%128
        data[q,k] = [r,g,b]
        i =i + 1
        k = k+1;
        if k == 500:
            q = q+1
            k = 0


img = Image.fromarray(data, 'RGB')
img.save('./my1.png')
img.show()
gaussian()

