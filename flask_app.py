import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pdz
app = Flask(__name__)
from tensorflow.keras.models import load_model
import re
import base64
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter

# from scipy.imageio import imsave, imread, imresize


model = load_model('model_1.h5')





def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva




def convertImage(imageData1):
    imgstr = re.search(r'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))
        
@app.route('/')
def home():
    return render_template('index.html')

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.jpeg','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict',methods=['POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = np.array(imageprepare('output.jpeg')).reshape(28, 28)
    
    # x = x.reshape(28,28,1)
    # print(x.shape)
    x = x.reshape(1,28,28,1)
    
    prediction = model.predict(x)
    return(str(np.argmax(prediction)))
    
    

	
    
if __name__ == "__main__":
    app.run(debug=False, threaded = False)

       
        
    
        
        


	

    
	
	
	
		
    
