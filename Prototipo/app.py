from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from flask_caching import Cache
import matplotlib.pyplot as plt


import os
from flask import make_response


from tensorflow.python.framework import ops


app = Flask(__name__, template_folder='templates')

app.config["IMAGE_UPLOADS"] = "D:/sumaFlask/Repositorio"
cache = Cache(app)

def init():
    global model,graph
    # load the pre-trained Keras model    
	#model = load_model('model/modeloFinalCV5.h5')
	#model = load_model('model/modeloDPFinalCV5.h5')
    #model = load_model('model/modelo2LYFinalCV5.h5')
	#model = load_model('model/modeloDPFinalCV5TLFT.h5')	
	#model = load_model('model/modeloDP2LYFinalCV5.h5')
    model = load_model('model/modeloDPFinalCV5TL.h5')	
    model._make_predict_function()
    ops.reset_default_graph()
    graph = ops.Graph()

@app.route('/')
def upload_file():
   with app.app_context():
        cache.clear()
   resp = make_response(render_template('index.html'))    
   resp.delete_cookie('sessionID')
   cache.clear()
   #return render_template('index.html')
   return resp
    
@app.route('/home', methods = ['POST'])
def come_back():
   with app.app_context():
        cache.clear()
   resp = make_response(render_template('index.html'))    
   resp.delete_cookie('sessionID')
   cache.clear()
   if request.method == 'POST':
        return resp

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
   resp = make_response(render_template('nelsonortiz.html'))     
   resp.delete_cookie('sessionID')
   cache.clear()   
   if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("L")
        imagen4 = img.convert('RGB').save("static/Imagen/image_name.jpg","JPEG")        
        img = img.resize((320,240))
        im2arr = np.array(img)        
        im2arr = im2arr.reshape(1,320,240,1)   
        
        with graph.as_default():
            print("Hola Nelson, entramos")
            y_pred = model.predict_classes(im2arr)            
            print("Ortizin pasamos esta PARTE")
            print("El valor de predicion es ",y_pred[0])
            nombre = numbers_to_strings(y_pred[0])      
        
        return  render_template('nelsonortiz.html', **locals())
        
def numbers_to_strings(argument): 
    switcher = { 
        0: "ATARDECER", 
        1: "BAÑO", 
        2: "CASA",
        3: "COLOR",
        4: "ESCUCHAR",
        5: "GRACIAS",
        6: "HOLA1",
        7: "HOLA2",
        8: "HOY",
        9: "MAMA",
        10:"MUCHOGUSTO",
        11:"NOMBRE",
        12:"NOVIO",
        13:"PAPA",
        14:"PROFESOR",
        15:"QUEPASO",
        16:"SIENTESE",
        17:"TELEVISION",
        18:"TENERCURIOSIDAD",
        19:"TENERPOSESION",
        20:"UNIVERSIDAD",
        21:"YO",

    } 
    return switcher.get(argument, "DESCONOCIDO")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
		
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    init()
    app.run(debug = True)