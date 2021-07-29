from flask import Flask,render_template,request,redirect,url_for
import os
from re import findall
import os
import streamlit as st
import pandas as pd 
import numpy as np
import ogdiabret
from PIL import Image 

app=Flask("__name__")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/home")
def home1():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/post")
def post(): 
    return render_template('post.html')

@app.route("/True")
def true(): 
    return render_template('True.html')

@app.route("/False")
def false(): 
    return render_template('False.html')

def load_image(image_file):
	img = Image.open(image_file)
	return img
    
app.config["IMAGE_UPLOADS"]="/home/mahesh/Mahesh/Python/BEproject/diaretdb1/diaretdb1v1/diaretdb1v1v1/resources/images/db1fundusimages"

@app.route("/upload-files",methods=['GET','POST'])
def upload_image():
    if request.method=='POST':
        if request.files:
            image=request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"],image.filename))
            print("Image Saved")
            
            ogdiabret.Image_Processing()
            img = load_image(image)
            test_str = image.filename

            file_no=findall(r'\d+',test_str)
            file_no=int(file_no[0])
            ogdiabret.Preprocessing()
            
            test_image=ogdiabret.scaler_means[file_no]
            #ogdiabret.scaler_means=np.array(ogdiabret.scaler_means)
            ogdiabret.scaler_means.pop(file_no)
            ogdiabret.scaler_means=np.array(ogdiabret.scaler_means)
            ogdiabret.scaler_means=ogdiabret.scaler_means.reshape(-1,1)
            test_result=ogdiabret.y_result[file_no]
            test_image=test_image.reshape(-1,1)
            test_result=np.array(test_result)

            ogdiabret.y_result.pop(file_no)
            ogdiabret.y_result=np.array(ogdiabret.y_result)
            out=ogdiabret.Model_KNN(test_image,test_result)
            if out==1:
                return redirect(url_for("true"))

            else:
                return redirect(url_for("false"))
        
    return render_template('post.html')

app.run(debug=True)