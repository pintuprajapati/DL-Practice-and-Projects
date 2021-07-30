from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

# Keras Libraries
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils libraries
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Defining a flask app
app = Flask(__name__)

# Model which is saved with Keras model.save()
MODEL_PATH ='car_model_resnet50.h5'

# Loading the trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    ## Preprocessing the image, converting image into array
    x = image.img_to_array(img)
    ## Scaling. Converting between 0 to 1 (as we did for training data)
    x=x/255
    x = np.expand_dims(x, axis=0) ## Expanding an image. adding a new dimension.
   
    ## Output to be displayed on web page
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Car is Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car is Mercedes"   
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
